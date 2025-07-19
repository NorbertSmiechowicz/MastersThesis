import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class EnergyGNNForecaster(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        out = self.lin(x)
        return out  # shape: (num_nodes, n_timestamps)


def train_gnn(
    model: torch.nn.Module,
    data: Data,
    lr: float = 0.01,
    patience: int = 20,
    min_delta: float = 1e-4,
    max_epochs: int = 1000
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    epochs_no_improve = 0
    epoch = 0
    loss_history = []
    while epoch < max_epochs:
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_history.append(loss_value)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value
            }, f"checkpoint_epoch_{epoch}.pt")
        # Early stopping
        if best_loss - loss_value > min_delta:
            best_loss = loss_value
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Loss stagnated for {patience} epochs.")
            break
        epoch += 1
    # Save final model
    torch.save(model.state_dict(), "energy_gnn_final.pt")

def evaluate_gnn(
    model: torch.nn.Module,
    data: Data
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        forecast = model(data)
    return forecast