"""
LSTM + GAT / GCN Imputer (single-file)
- fixes, improvements and an ONLINE (streaming) imputation wrapper
- Usage: adjust csv_path and run. See OnlineImputer for online usage.
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
from collections import deque

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
    """X: (T, F) numpy array possibly with NaNs
    returns mins, maxs (1d arrays length F). If all-NaN column -> mins=0, maxs=1
    """
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    # handle all-NaN columns
    all_nan = np.isnan(mins) | np.isnan(maxs)
    mins[all_nan] = 0.0
    maxs[all_nan] = 1.0
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


def compute_correlation_matrix(X, min_periods=3, method='pearson'):
    # uses pairwise correlation with pandas (pairwise complete obs)
    df = pd.DataFrame(X)
    corr = df.corr(method=method, min_periods=min_periods).fillna(0.0).values
    return corr


def compute_distance_matrix(X, fill_with='mean'):
    # vectorized pairwise Euclidean between columns; impute NaNs by column mean
    Xf = X.copy()
    if fill_with == 'mean':
        col_mean = np.nanmean(Xf, axis=0)
        inds = np.where(np.isnan(Xf))
        Xf[inds] = np.take(col_mean, inds[1])
    Fi = Xf.shape[1]
    # compute pairwise distances using broadcasting
    # shape (Fi, Fi)
    A = Xf.T  # (Fi, T)
    # squared distances
    norms = (A ** 2).sum(axis=1, keepdims=True)
    d2 = norms + norms.T - 2 * (A @ A.T)
    d2[d2 < 0] = 0.0
    d = np.sqrt(d2)
    return d


def build_edge_index_from_similarity(sim, k=4, threshold=None):
    # sim: (Fi,Fi) similarity matrix (higher = closer)
    Fi = sim.shape[0]
    edges = []
    weights = []
    for i in range(Fi):
        row = sim[i].copy()
        row[i] = 0.0
        # top-k by absolute value
        if k < Fi:
            idxs = np.argpartition(-np.abs(row), k)[:k]
            # sort those by descending abs
            idxs = idxs[np.argsort(-np.abs(row[idxs]))]
        else:
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
    if edge_weight.numel() and edge_weight.max() > 0:
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


def build_edge_index_from_corr_matrix(corr,
                                     k=4,
                                     threshold=None,
                                     keep_sign=False,
                                     transform='abs',   # 'abs'|'linear'|'signed'
                                     power=1.0,
                                     alpha=1.0):
    Fi = corr.shape[0]
    edges = []
    weights = []
    for i in range(Fi):
        row = corr[i].copy()
        row[i] = 0.0
        if k < Fi:
            idxs = np.argpartition(-np.abs(row), k)[:k]
            idxs = idxs[np.argsort(-np.abs(row[idxs]))]
        else:
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
                # preserve sign
                w = (abs(w_raw) ** power) * alpha * np.sign(w_raw)
            else:
                w = (abs(w_raw) ** power) * alpha
            edges.append((i, j))
            weights.append(w)
    if len(edges) == 0:
        for i in range(Fi):
            for j in range(Fi):
                if i != j:
                    edges.append((i,j)); weights.append(0.1*alpha)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    if edge_weight.numel() and edge_weight.max() > 0:
        if keep_sign:
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
        # input features per node: (value, observed_flag)
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
        # build batched edge_index by offsetting node ids for each sample
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
        # GCNConv expects edge_index shape (2, E) and edge_weight shape (E,)
        gnn_out = self.gcn(x_cat, edge_index_batch, edge_weight_batch)
        gnn_out = F.relu(gnn_out)
        preds = self.out(gnn_out).view(B, num_nodes)
        return preds

# ---------------------- Online Imputer Wrapper ----------------------

class OnlineImputer:
    """A lightweight online wrapper that keeps a causal trend estimate (EMA) and a short buffer of past rows
    to build windows and call a trained model for single-step imputation.

    Design choices & rationale:
    - Use exponential moving average (EMA) as a causal, memory-light trend estimator. A simple moving average
      with a deque could be used instead if exact windowed mean is desired.
    - Expect a trained model (LSTM_GAT_Imputer or LSTM_GCN_Imputer) and precomputed mins/maxs and graph.
    - The online API accepts one raw row (1D numpy array with NaNs) at a time and returns the row with NaNs imputed.
    """
    def __init__(self, model, edge_index, edge_weight, mins, maxs, seq_len=8, trend_alpha=0.05, device=None):
        self.model = model
        self.model.eval()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.mins = mins
        self.maxs = maxs
        self.seq_len = seq_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # initialize buffer with NaNs until filled
        self.buffer = deque(maxlen=seq_len)
        self.Fi = len(mins)
        for _ in range(seq_len):
            self.buffer.append(np.full(self.Fi, np.nan, dtype=float))
        # EMA trend
        self.trend = np.zeros(self.Fi, dtype=float)
        self.trend_inited = False
        self.alpha = float(trend_alpha)

    def update_trend(self, row):
        # row: 1D numpy raw values (may contain NaNs) -- EMA ignores NaNs
        mask = ~np.isnan(row)
        if not self.trend_inited:
            # initialize EMA with first non-NaN per feature when available
            self.trend[mask] = row[mask]
            if mask.any():
                self.trend_inited = True
            return
        # EMA update only for observed features
        self.trend[mask] = self.alpha * row[mask] + (1.0 - self.alpha) * self.trend[mask]

    def push_row(self, row):
        # push raw row (1D numpy array), update trend and buffer
        self.update_trend(row)
        self.buffer.append(row.copy())

    def impute_row(self, raw_row):
        """Main entry: provide a raw row (1D numpy array with NaNs). Returns imputed row (same shape).
        The method is causal: the imputed values at time t use only past information + current observed values.
        Steps:
          1. push current raw_row into buffer and update EMA trend
          2. produce a window of length seq_len from buffer (most recent last)
          3. detrend: window - trend (use current EMA as trend estimate for the last row, for earlier rows use stored trend)
          4. scale (minmax using mins/maxs)
          5. run model to predict t's values (scaled detrended)
          6. fill only NaNs in the current row
          7. inverse-scale and add trend back
        """
        assert raw_row.shape[0] == self.Fi
        # push and update trend (trend updated with current observed values so it's causal)
        self.push_row(raw_row)
        # construct window array (seq_len, Fi)
        window = np.stack(list(self.buffer), axis=0)
        # detrend: subtract current EMA trend for simplicity
        # note: for better accuracy, one might store historical per-timestamp trends
        window_detr = window - self.trend
        # scale
        window_scaled = minmax_scale_with_nan(window_detr, self.mins, self.maxs)
        # build tensor (1, seq_len, Fi)
        window_tensor = torch.tensor(window_scaled, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(window_tensor, self.edge_index.to(self.device), self.edge_weight.to(self.device))
            preds = preds.squeeze(0).cpu().numpy()  # scaled detrended predictions for time t
        # only fill NaNs in current raw_row
        out = raw_row.copy()
        mask = np.isnan(out)
        if mask.any():
            # inverse scale and add trend
            pred_inv = inverse_minmax_scale(preds, self.mins, self.maxs)
            pred_with_trend = pred_inv + self.trend
            out[mask] = pred_with_trend[mask]
        return out

    def save_state(self, path):
        # minimal state save (model state_dict, mins/maxs, edge_index/weights)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        np.save(os.path.join(path, 'mins.npy'), self.mins)
        np.save(os.path.join(path, 'maxs.npy'), self.maxs)
        torch.save(self.edge_index, os.path.join(path, 'edge_index.pt'))
        torch.save(self.edge_weight, os.path.join(path, 'edge_weight.pt'))

    @staticmethod
    def load_state(model_class, path, seq_len=8, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mins = np.load(os.path.join(path, 'mins.npy'))
        maxs = np.load(os.path.join(path, 'maxs.npy'))
        edge_index = torch.load(os.path.join(path, 'edge_index.pt'))
        edge_weight = torch.load(os.path.join(path, 'edge_weight.pt'))
        model = model_class(seq_len=seq_len, num_nodes=len(mins))
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device))
        return OnlineImputer(model, edge_index, edge_weight, mins, maxs, seq_len=seq_len, device=device)

# ---------------------- Training & Demo (kept mostly as before with small fixes) ----------------------

def train_demo(csv_path,
               seq_len=8, batch_size=32, epochs=8, use_gat=True,
               k_neighbors=3, device=None, trend_window=72, save_model_dir=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # === Load raw data ===
    df, X_raw, cols = load_csv(csv_path)
    T, Fi = X_raw.shape
    print(f'Loaded CSV with T={T}, F={Fi} columns')

    # === Split train/test by time (deterministic) ===
    train_size = int(0.8 * T)
    # train_size = 855  # user-specific previous value removed in favor of 80/20 split
    assert train_size < T, "train_size must be < total rows"
    test_data_raw = X_raw[train_size - seq_len:]
    train_data_raw = X_raw[:train_size]

    # === Compute rolling trend on the full raw series (causal rolling mean with min_periods=1)
    df_full = pd.DataFrame(X_raw, columns=cols)
    trend_full = df_full.rolling(window=trend_window, min_periods=1).mean().values
    trend_train = trend_full[:train_size]
    trend_test_full = trend_full[train_size - seq_len:]

    # === Detrend (raw - trend) ===
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
            # keep predictions near [0,1] after scaling to stabilize; consider removing or changing regularizer
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

    # === Optionally save model + preprocessing for online use ===
    if save_model_dir is not None:
        os.makedirs(save_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'model.pt'))
        np.save(os.path.join(save_model_dir, 'mins.npy'), mins_tr)
        np.save(os.path.join(save_model_dir, 'maxs.npy'), maxs_tr)
        torch.save(edge_index_train, os.path.join(save_model_dir, 'edge_index.pt'))
        torch.save(edge_weight_train, os.path.join(save_model_dir, 'edge_weight.pt'))
        print('Saved model and preprocessing to', save_model_dir)

    # === Imputation across TEST series (operate on detrended+scaled test_data) ===
    model.eval()
    X_test_filled_scaled = test_data.copy()
    with torch.no_grad():
        for t in range(seq_len, X_test_filled_scaled.shape[0]):
            window = X_test_filled_scaled[t-seq_len:t, :].copy()
            window_tensor = torch.tensor(window, dtype=torch.float).unsqueeze(0).to(device)
            preds = model(window_tensor, edge_index_train, edge_weight_train)
            preds = preds.squeeze(0).cpu().numpy()
            for f in range(Fi):
                if np.isnan(X_test_filled_scaled[t, f]):
                    X_test_filled_scaled[t, f] = preds[f]

    detrended_imputed = inverse_minmax_scale(X_test_filled_scaled, mins_tr, maxs_tr)
    final_test_with_trend = detrended_imputed + trend_test_full
    final_test_actual = final_test_with_trend[seq_len:, :]
    original_test_raw = X_raw[train_size:, :]
    imputed_test_output = original_test_raw.copy()
    nan_mask = np.isnan(imputed_test_output)
    imputed_test_output[nan_mask] = final_test_actual[nan_mask]

    out_test_path = os.path.join(os.getcwd(), 'imputed_test_result_with_trend.csv')
    pd.DataFrame(imputed_test_output, columns=cols).to_csv(out_test_path, index=False)
    print('Saved imputed test CSV with trend re-added to', out_test_path)


if __name__ == '__main__':
    # example train (adjust csv_path as needed)
    csv_path = '/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv'
    print('we here!')
    train_demo(csv_path=csv_path, seq_len=16, batch_size=64, epochs=50, use_gat=True, save_model_dir='./saved_imputer')
