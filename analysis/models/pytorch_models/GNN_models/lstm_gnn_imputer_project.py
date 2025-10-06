"""
LSTM + GAT / GCN Imputer
---------------------------------
This single-file project contains:
 - utility functions for loading and scaling CSV data
 - functions to compute correlation/distance and build graph (edge_index, edge_weight)
 - TimeWindowsDataset for sliding windows with NaNs preserved
 - EdgeGATConv (edge-aware single-head attention MessagePassing)
 - LSTM_GAT_Imputer and LSTM_GCN_Imputer classes (no variable shadowing of `F`)
 - training & imputation demo using the uploaded file

Usage:
  - Put your CSV at /mnt/data/Industrial_fault_detection.csv (this file is provided in the environment)
  - Run: python lstm_gnn_imputer_project.py

Notes:
  - Requires: torch, torch_geometric, pandas, numpy, scikit-learn
  - If torch_geometric isn't installed, see its installation docs (it must match your PyTorch+CUDA version).

Author: assistant (example)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Attempt to import torch_geometric components
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
except Exception as e:
    raise RuntimeError("This script requires torch_geometric. Install it for your PyTorch/CUDA version. Error: {}".format(e))

# ---------------------- Utilities ----------------------

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    arr = df.values.astype(float)
    cols = list(df.columns)
    return df, arr, cols


def compute_feature_minmax(X):
    """X: (T, F) numpy array possibly with NaNs"""
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    # avoid zero-range
    zero_range = np.isclose(maxs, mins)
    maxs[zero_range] = mins[zero_range] + 1.0
    return mins, maxs


def minmax_scale_with_nan(X, mins, maxs):
    Xs = (X - mins) / (maxs - mins)
    Xs[np.isnan(X)] = np.nan
    return Xs


def inverse_minmax_scale(Xs, mins, maxs):
    return Xs * (maxs - mins) + mins


def compute_correlation_matrix(X, min_periods=3):
    df = pd.DataFrame(X)
    corr = df.corr(method='pearson', min_periods=min_periods).fillna(0.0).values
    return corr


def compute_distance_matrix(X, fill_with='mean'):
    Xf = X.copy()
    if fill_with == 'mean':
        col_mean = np.nanmean(Xf, axis=0)
        inds = np.where(np.isnan(Xf))
        Xf[inds] = np.take(col_mean, inds[1])
    Fi = Xf.shape[1]
    d = np.zeros((Fi, Fi))
    for i in range(Fi):
        for j in range(Fi):
            d[i, j] = np.linalg.norm(Xf[:, i] - Xf[:, j])
    return d


def build_edge_index_from_similarity(sim, k=4, threshold=None):
    # sim: (Fi,Fi) similarity matrix (higher = closer)
    Fi = sim.shape[0]
    edges = []
    weights = []
    for i in range(Fi):
        row = sim[i].copy()
        row[i] = 0.0
        idxs = np.argsort(-np.abs(row))
        selected = []
        for j in idxs:
            if j == i:
                continue
            if threshold is not None and abs(row[j]) < threshold:
                continue
            selected.append(j)
            if len(selected) >= k:
                break
        for j in selected:
            edges.append((i, j))
            edges.append((j, i))
            w = float(abs(row[j]))
            weights.append(w)
            weights.append(w)
    if len(edges) == 0:
        for i in range(Fi):
            for j in range(Fi):
                if i != j:
                    edges.append((i, j)); weights.append(0.1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    if edge_weight.max() > 0:
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


# функция построения рёбер
def build_edge_index_from_corr_matrix(corr,
                                     k=4,
                                     threshold=None,
                                     keep_sign=False,
                                     transform='abs',   # 'abs'|'linear'|'signed'
                                     power=1.0,
                                     alpha=1.0):
    """
    Build edge_index, edge_weight from correlation matrix.
    - keep_sign: if True, preserve sign in weights (weights can be negative)
    - transform:
        'abs'    -> w = |corr|^power
        'linear' -> w = ((corr + 1)/2)^power  (maps [-1,1] -> [0,1])
        'signed' -> w = corr (optionally scaled)
    - alpha: global multiplier for weights (float)
    Returns: edge_index (2,E), edge_weight (E,)
    """
    Fi = corr.shape[0]
    edges = []
    weights = []
    for i in range(Fi):
        row = corr[i].copy()
        row[i] = 0.0
        idxs = np.argsort(-np.abs(row))
        selected = []
        for j in idxs:
            if j == i:
                continue
            if threshold is not None and abs(row[j]) < threshold:
                continue
            selected.append(j)
            if len(selected) >= k:
                break
        for j in selected:
            w_raw = row[j]
            if transform == 'abs':
                w = (abs(w_raw) ** power) * alpha
            elif transform == 'linear':
                w = (((w_raw + 1.0) / 2.0) ** power) * alpha
            elif transform == 'signed':
                w = (w_raw ** power) * alpha
            else:
                # fallback: abs
                w = (abs(w_raw) ** power) * alpha
            # add directed edges both directions if you want symmetric info
            edges.append((i, j))
            weights.append(w)
            # if you want symmetric, also add (j,i) with same weight; if using top-k per node we may already be adding both
    if len(edges) == 0:
        # fallback: fully connect small weights
        for i in range(Fi):
            for j in range(Fi):
                if i != j:
                    edges.append((i,j)); weights.append(0.1*alpha)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    # Optionally normalize to [0,1] (if transform produced big values)
    if edge_weight.numel() and edge_weight.max() > 0:
        # keep sign if negative values allowed
        if keep_sign:
            # normalize magnitudes to [0,1], preserve sign
            mags = edge_weight.abs()
            mags = mags / mags.max()
            edge_weight = mags * torch.sign(edge_weight)
        else:
            edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


# ---------------------- Dataset ----------------------

class TimeWindowsDataset(Dataset):
    """Return sliding windows (seq_len, num_features) preserving NaNs."""
    def __init__(self, X, seq_len=8):
        # X: numpy (T, F)
        self.X = X
        self.seq_len = seq_len
        self.indices = list(range(0, X.shape[0] - seq_len + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        w = self.X[i:i+self.seq_len, :]
        return torch.tensor(w, dtype=torch.float)


def collate_windows(batch):
    return torch.stack(batch, dim=0)

# ---------------------- Edge-aware GATConv ----------------------

# class EdgeGATConv(MessagePassing):
#     """Single-head edge-aware GAT-like convolution (MessagePassing).
#     This class is intentionally simple for clarity and educational purposes.
#     The raw attention e_ij is multiplied by edge_weight (if provided) before softmax.
#     """
#     def __init__(self, in_channels, out_channels, negative_slope=0.2, bias=True):
#         super().__init__(aggr='add')
#         self.lin = nn.Linear(in_channels, out_channels, bias=False)
#         self.att = nn.Parameter(torch.Tensor(2 * out_channels))
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.lin.weight)
#         nn.init.xavier_uniform_(self.att.view(1, -1))
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, edge_index, edge_weight=None):
#         # x: (N, in_channels)
#         x_l = self.lin(x)  # (N, out)
#         row = edge_index[0]
#         col = edge_index[1]
#         x_row = x_l[row]
#         x_col = x_l[col]
#         feat_cat = torch.cat([x_row, x_col], dim=-1)
#         e = (feat_cat * self.att).sum(dim=-1)
#         if edge_weight is not None:
#             ew = edge_weight.view(-1).to(e.dtype)
#             e = e * ew
#         alpha = softmax(self.leaky_relu(e), col)
#         out = self.propagate(edge_index, x=x_l, alpha=alpha)
#         if self.bias is not None:
#             out = out + self.bias
#         return out

#     def message(self, x_j, alpha):
#         return x_j * alpha.view(-1, 1)


class EdgeGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, bias=True, edge_mode='mul', edge_coef=1.0):
        """
        edge_mode: 'mul' — multiply raw attention by edge_weight
                   'add' — add edge_bias (edge_coef * edge_weight) to raw attention
        edge_coef: scalar factor (float) applied to edge_weight when mode='add'
        """
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * out_channels))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.edge_mode = edge_mode
        self.edge_coef = float(edge_coef)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        x_l = self.lin(x)
        row, col = edge_index
        x_row = x_l[row]
        x_col = x_l[col]
        feat_cat = torch.cat([x_row, x_col], dim=-1)
        e = (feat_cat * self.att).sum(dim=-1)  # raw unnormalized attention (E,)
        if edge_weight is not None:
            ew = edge_weight.view(-1).to(e.dtype)
            if self.edge_mode == 'mul':
                # prevent zeroing out completely if ew ~ 0: use 1 + alpha*ew or simply multiply if ew in [0,1]
                e = e * ew
            elif self.edge_mode == 'add':
                e = e + self.edge_coef * ew
        alpha = softmax(self.leaky_relu(e), col)
        out = self.propagate(edge_index, x=x_l, alpha=alpha)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, alpha):
        return x_j * alpha.view(-1, 1)




# ---------------------- Models ----------------------

class LSTM_GAT_Imputer(nn.Module):
    def __init__(self, seq_len, num_nodes, lstm_hidden=64, gnn_hidden=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.lstm_hidden = lstm_hidden
        self.gnn_hidden = gnn_hidden
        self.lstm = nn.LSTM(input_size=2, hidden_size=lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, gnn_hidden)
        self.gat = EdgeGATConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, windows, edge_index, edge_weight):
        # windows: (B, S, num_nodes) with NaNs
        B, S, num_nodes = windows.shape
        assert num_nodes == self.num_nodes
        values = windows.clone()
        observed = ~torch.isnan(values)
        values_filled = torch.nan_to_num(values, nan=0.0)
        ch0 = values_filled.permute(0, 2, 1).unsqueeze(-1)
        ch1 = observed.permute(0, 2, 1).float().unsqueeze(-1)
        node_seq = torch.cat([ch0, ch1], dim=-1)  # (B, num_nodes, S, 2)
        node_seq = node_seq.view(B * num_nodes, S, 2)
        lstm_out, _ = self.lstm(node_seq)
        last = lstm_out[:, -1, :]
        last = last.view(B, num_nodes, -1)
        x_gnn = self.proj(last)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)
        x_cat = x_gnn.view(B * num_nodes, -1)
        device = x_cat.device
        ei = edge_index.to(device)
        ew = edge_weight.to(device)
        edge_indices = [ei + b * num_nodes for b in range(B)]
        edge_index_batch = torch.cat(edge_indices, dim=1)
        edge_weight_batch = ew.repeat(B)
        gnn_out = self.gat(x_cat, edge_index_batch, edge_weight_batch)
        gnn_out = F.relu(gnn_out)
        preds = self.out(gnn_out).view(B, num_nodes)
        return preds


class LSTM_GCN_Imputer(nn.Module):
    def __init__(self, seq_len, num_nodes, lstm_hidden=64, gnn_hidden=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.lstm_hidden = lstm_hidden
        self.gnn_hidden = gnn_hidden
        self.lstm = nn.LSTM(input_size=2, hidden_size=lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, gnn_hidden)
        self.gcn = GCNConv(gnn_hidden, gnn_hidden)
        self.out = nn.Linear(gnn_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, windows, edge_index, edge_weight):
        B, S, num_nodes = windows.shape
        assert num_nodes == self.num_nodes
        values = windows.clone()
        observed = ~torch.isnan(values)
        values_filled = torch.nan_to_num(values, nan=0.0)
        ch0 = values_filled.permute(0, 2, 1).unsqueeze(-1)
        ch1 = observed.permute(0, 2, 1).float().unsqueeze(-1)
        node_seq = torch.cat([ch0, ch1], dim=-1)
        node_seq = node_seq.view(B * num_nodes, S, 2)
        lstm_out, _ = self.lstm(node_seq)
        last = lstm_out[:, -1, :]
        last = last.view(B, num_nodes, -1)
        x_gnn = self.proj(last)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)
        x_cat = x_gnn.view(B * num_nodes, -1)
        device = x_cat.device
        ei = edge_index.to(device)
        ew = edge_weight.to(device)
        edge_indices = [ei + b * num_nodes for b in range(B)]
        edge_index_batch = torch.cat(edge_indices, dim=1)
        edge_weight_batch = ew.repeat(B)
        gnn_out = self.gcn(x_cat, edge_index_batch, edge_weight_batch)
        gnn_out = F.relu(gnn_out)
        preds = self.out(gnn_out).view(B, num_nodes)
        return preds

# ---------------------- Training & Demo ----------------------

# def train_demo(csv_path='/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv',
#                seq_len=8, batch_size=32, epochs=8, use_gat=True,
#                k_neighbors=3, device=None):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Device:', device)

#     # === Load & scale ===
#     df, X_raw, cols = load_csv(csv_path)
#     T, Fi = X_raw.shape
#     print(f'Loaded CSV with T={T}, F={Fi} columns')

#     # === Split train/test by time ===
#     train_size = int(0.8 * T)
#     train_size = 855

#     train_data_raw = X_raw[:train_size]
#     test_data_raw = X_raw[train_size:]

#     mins_tr, maxs_tr = compute_feature_minmax(train_data_raw)
#     train_data = minmax_scale_with_nan(train_data_raw, mins_tr, maxs_tr)

#     test_data = minmax_scale_with_nan(test_data_raw, mins_tr, maxs_tr)
#     print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

#     # === Build graph only on train data ===
#     corr_train = compute_correlation_matrix(train_data)
#     edge_index_train, edge_weight_train = build_edge_index_from_similarity(corr_train, k=k_neighbors)
#     print('Built graph:', edge_index_train.shape, edge_weight_train.shape)

#     # === Datasets and loaders ===
#     train_dataset = TimeWindowsDataset(train_data, seq_len=seq_len)
#     test_dataset = TimeWindowsDataset(test_data, seq_len=seq_len)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_windows)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)

#     # === Model ===
#     if use_gat:
#         model = LSTM_GAT_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
#     else:
#         model = LSTM_GCN_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#     mask_prob = 0.2

#     # === Training ===
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         n_batches = 0
#         for batch in train_loader:
#             batch = batch.to(device)
#             B = batch.shape[0]
#             last = batch[:, -1, :]
#             known_last = ~torch.isnan(last)
#             rnd = torch.rand_like(last)
#             mask_pred = (rnd < mask_prob) & known_last
#             masked_batch = batch.clone()
#             masked_batch[:, -1, :][mask_pred] = float('nan')
#             preds = model(masked_batch, edge_index_train, edge_weight_train)
#             if mask_pred.sum() == 0:
#                 continue
#             y_true = last[mask_pred]
#             y_pred = preds[mask_pred]
#             mse = F.mse_loss(y_pred, y_true)
#             over = F.relu(y_pred - 1.0)
#             under = F.relu(0.0 - y_pred)
#             loss = mse + 5.0 * (over.mean() + under.mean())
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             total_loss += loss.item()
#             n_batches += 1
#         avg_loss = total_loss / max(1, n_batches)
#         print(f'Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f}')

#         # === Validation ===
#         model.eval()
#         with torch.no_grad():
#             val_loss = 0.0
#             n_val = 0
#             for batch in test_loader:
#                 batch = batch.to(device)
#                 B = batch.shape[0]
#                 last = batch[:, -1, :]
#                 known_last = ~torch.isnan(last)
#                 rnd = torch.rand_like(last)
#                 mask_pred = (rnd < mask_prob) & known_last
#                 masked_batch = batch.clone()
#                 masked_batch[:, -1, :][mask_pred] = float('nan')
#                 preds = model(masked_batch, edge_index_train, edge_weight_train)
#                 if mask_pred.sum() == 0:
#                     continue
#                 y_true = last[mask_pred]
#                 y_pred = preds[mask_pred]
#                 mse = F.mse_loss(y_pred, y_true)
#                 val_loss += mse.item()
#                 n_val += 1
#             avg_val_loss = val_loss / max(1, n_val)
#         print(f'           val_loss={avg_val_loss:.6f}')

#     # === Imputation across full series ===
#     model.eval()
#     X_test_filled_scaled = test_data.copy()
#     with torch.no_grad():
#         for t in range(seq_len, test_data.shape[0]):
#             window = X_test_filled_scaled[t-seq_len:t, :].copy()
#             window_tensor = torch.tensor(window, dtype=torch.float).unsqueeze(0).to(device)
#             preds = model(window_tensor, edge_index_train, edge_weight_train)
#             preds = preds.squeeze(0).cpu().numpy()
#             for f in range(Fi):
#                 if np.isnan(X_test_filled_scaled[t, f]):
#                     X_test_filled_scaled[t, f] = preds[f]

#     X_test_imputed = inverse_minmax_scale(X_test_filled_scaled, mins_tr, maxs_tr)
#     out_test_path = '/Users/kathrinebovkun/PycharmProjects/experiments/data/imputed_test_result.csv'
#     pd.DataFrame(X_test_imputed, columns=cols).to_csv(out_test_path, index=False)
#     print('Saved imputed test CSV to', out_test_path)


def train_demo(csv_path='/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv',
               seq_len=8, batch_size=32, epochs=8, use_gat=True,
               k_neighbors=3, device=None, trend_window=72):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # === Load raw data ===
    df, X_raw, cols = load_csv(csv_path)              # X_raw shape (T, Fi), may contain NaN
    T, Fi = X_raw.shape
    print(f'Loaded CSV with T={T}, F={Fi} columns')

    # === Split train/test by time (deterministic) ===
    train_size = 855  # as you requested (first 855 rows for train)
    assert train_size < T, "train_size must be < total rows"
    # We'll keep an overlap of seq_len for building first test window:
    # test_data_raw contains last seq_len of train + actual test portion
    test_data_raw = X_raw[train_size - seq_len:]     # shape = seq_len + (T - train_size)
    train_data_raw = X_raw[:train_size]              # shape = train_size

    # === Compute rolling trend on the full raw series (uses only past values for each t) ===
    # We compute rolling mean on the full X_raw to ensure trend at each timestamp uses only past/current values.
    df_full = pd.DataFrame(X_raw, columns=cols)
    # rolling(window=trend_window, min_periods=1) — first values use smaller window
    trend_full = df_full.rolling(window=trend_window, min_periods=1).mean().values
    # Split trend arrays consistently with train/test split (and overlap)
    trend_train = trend_full[:train_size]                      # (train_size, Fi)
    trend_test_full = trend_full[train_size - seq_len:]        # (seq_len + test_size, Fi)

    # === Detrend (raw - trend) ===
    # subtract elementwise; NaNs propagate (if a value is NaN, result stays NaN)
    train_detrended_raw = train_data_raw - trend_train
    test_detrended_raw = test_data_raw - trend_test_full

    # === Compute min/max on detrended TRAIN only, then scale both train/test by those mins/maxs ===
    mins_tr, maxs_tr = compute_feature_minmax(train_detrended_raw)
    train_data = minmax_scale_with_nan(train_detrended_raw, mins_tr, maxs_tr)
    test_data = minmax_scale_with_nan(test_detrended_raw, mins_tr, maxs_tr)

    print(f"Train detrended shape: {train_data.shape}, Test detrended shape (with overlap): {test_data.shape}")

    # === Build graph on detrended TRAIN data (correlation-based) ===
    corr_train = compute_correlation_matrix(train_detrended_raw)
    edge_index_train, edge_weight_train = build_edge_index_from_similarity(corr_train, k=k_neighbors)
    print('Built graph on train:', edge_index_train.shape, edge_weight_train.shape)

    # === Datasets & loaders ===
    train_dataset = TimeWindowsDataset(train_data, seq_len=seq_len)
    test_dataset = TimeWindowsDataset(test_data, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_windows)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)

    # === Model ===
    if use_gat:
        model = LSTM_GAT_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
    else:
        model = LSTM_GCN_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mask_prob = 0.2

    # === Training loop ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)                        # (B, seq_len, Fi) scaled detrended
            last = batch[:, -1, :]
            known_last = ~torch.isnan(last)
            rnd = torch.rand_like(last)
            mask_pred = (rnd < mask_prob) & known_last
            masked_batch = batch.clone()
            masked_batch[:, -1, :][mask_pred] = float('nan')
            preds = model(masked_batch, edge_index_train, edge_weight_train)
            if mask_pred.sum() == 0:
                continue
            y_true = last[mask_pred]
            y_pred = preds[mask_pred]
            mse = F.mse_loss(y_pred, y_true)
            over = F.relu(y_pred - 1.0)
            under = F.relu(0.0 - y_pred)
            loss = mse + 5.0 * (over.mean() + under.mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        print(f'Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f}')

        # === Validation on test windows (no shuffle) ===
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for batch in test_loader:
                batch = batch.to(device)
                last = batch[:, -1, :]
                known_last = ~torch.isnan(last)
                rnd = torch.rand_like(last)
                mask_pred = (rnd < mask_prob) & known_last
                masked_batch = batch.clone()
                masked_batch[:, -1, :][mask_pred] = float('nan')
                preds = model(masked_batch, edge_index_train, edge_weight_train)
                if mask_pred.sum() == 0:
                    continue
                y_true = last[mask_pred]
                y_pred = preds[mask_pred]
                mse = F.mse_loss(y_pred, y_true)
                val_loss += mse.item()
                n_val += 1
            avg_val_loss = val_loss / max(1, n_val)
        print(f'           val_loss={avg_val_loss:.6f}')

    # === Imputation across TEST series (we operate on detrended+scaled test_data) ===
    model.eval()
    X_test_filled_scaled = test_data.copy()   # working copy (shape = seq_len + test_size, Fi)
    with torch.no_grad():
        # iterate over windows inside test_data (first window includes last train rows via overlap)
        for t in range(seq_len, X_test_filled_scaled.shape[0]):
            window = X_test_filled_scaled[t-seq_len:t, :].copy()
            window_tensor = torch.tensor(window, dtype=torch.float).unsqueeze(0).to(device)
            preds = model(window_tensor, edge_index_train, edge_weight_train)
            preds = preds.squeeze(0).cpu().numpy()   # predicted scaled detrended values for time t (Fi,)
            # fill only NaNs at time t (in scaled detrended space)
            for f in range(Fi):
                if np.isnan(X_test_filled_scaled[t, f]):
                    X_test_filled_scaled[t, f] = preds[f]

    # === Inverse min-max (detrended -> original detrended scale) ===
    detrended_imputed = inverse_minmax_scale(X_test_filled_scaled, mins_tr, maxs_tr)  # shape seq_len + test_size

    # === Add trend back to retrieve original-scale values ===
    # trend_test_full corresponds to the same rows as test_data_raw (seq_len + test_size)
    # final_test = detrended_imputed + trend_test_full
    final_test_with_trend = detrended_imputed + trend_test_full  # (seq_len + test_size, Fi)

    # === Extract actual test-time slice (exclude the initial overlap of seq_len rows) ===
    final_test_actual = final_test_with_trend[seq_len:, :]   # shape (test_size, Fi)
    original_test_raw = X_raw[train_size:, :]               # (test_size, Fi)

    # === Compose final test output: keep original observed values, replace NaNs by imputed ===
    imputed_test_output = original_test_raw.copy()
    nan_mask = np.isnan(imputed_test_output)
    imputed_test_output[nan_mask] = final_test_actual[nan_mask]

    out_test_path = '/Users/kathrinebovkun/PycharmProjects/experiments/data/imputed_test_result_with_trend.csv'
    pd.DataFrame(imputed_test_output, columns=cols).to_csv(out_test_path, index=False)
    print('Saved imputed test CSV with trend re-added to', out_test_path)


if __name__ == '__main__':
    train_demo(csv_path='/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv', seq_len=16, batch_size=64, epochs=150, use_gat=True)
