

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from Models.GCN import EnergyGNNForecaster, train_gnn, evaluate_gnn

def load_eu_grid_data(n_timestamps: int = 5) -> Data:
    data_dir = os.environ.get("EU_GRID_DATA_DIR", "")
    train_path = os.path.join(data_dir, "train-set.csv")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    train_data = train_df.values.astype(np.float32)
    nan_mask = np.isnan(train_data)
    col_means = np.nanmean(train_data, axis=0)
    train_data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    n_nodes = train_data.shape[1]
    x = torch.tensor(train_data[-1], dtype=torch.float).unsqueeze(1)
    y = torch.tensor(train_data[-n_timestamps:], dtype=torch.float).transpose(0,1)
    edge_index = torch.tensor(np.array([(i,j) for i in range(n_nodes) for j in range(n_nodes) if i!=j]).T, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def main() -> None:
    n_timestamps = 5
    data = load_eu_grid_data(n_timestamps=n_timestamps)
    model = EnergyGNNForecaster(
        in_channels=data.x.shape[1],
        hidden_channels=16,
        out_channels=data.y.shape[1],
        num_layers=2
    )
    train_gnn(model, data, lr=0.01, patience=20, min_delta=1e-4, max_epochs=1000)
    forecast = evaluate_gnn(model, data)
    print("Forecasted vectors (first node):", forecast[0].detach().cpu().numpy())

if __name__ == "__main__":
    main()

