# utils_and_dataset.py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

def load_csv_to_numpy(path):
    df_ = pd.read_csv(path)
    df = df_.copy()
    return df, df.values.astype(float), list(df.columns)

def compute_feature_minmax(X):
    # X: (T, num_features) numpy with np.nan
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    # handle constant columns
    maxs = np.where(np.isfinite(maxs) & (maxs == mins), mins + 1.0, maxs)
    return mins, maxs

def minmax_scale_with_nan(X, mins, maxs):
    # scale each feature to [0,1], keep NaN as NaN
    Xs = (X - mins) / (maxs - mins)
    # keep NaN positions
    Xs[np.isnan(X)] = np.nan
    return Xs

def inverse_minmax_scale(Xs, mins, maxs):
    return Xs * (maxs - mins) + mins

def compute_correlation_matrix(X):
    # X: (T, F) numpy with NaN. Use pandas to compute pairwise correlations ignoring NaNs.
    df = pd.DataFrame(X)
    corr = df.corr(method='pearson', min_periods=3).fillna(0.0).values
    return corr

def compute_distance_matrix(X, fill_with='mean'):
    # pairwise euclidean distance between columns (features), handling NaNs by fill
    Xf = X.copy()
    if fill_with == 'mean':
        col_mean = np.nanmean(Xf, axis=0)
        inds = np.where(np.isnan(Xf))
        Xf[inds] = np.take(col_mean, inds[1])
    F = Xf.shape[1]
    d = np.zeros((F, F))
    for i in range(F):
        for j in range(F):
            d[i, j] = np.linalg.norm(Xf[:, i] - Xf[:, j])
    return d

def build_edge_index_from_corr(corr, k=4, threshold=None):
    # corr: (F,F) numpy
    F = corr.shape[0]
    edges = []
    weights = []
    for i in range(F):
        row = corr[i].copy()
        row[i] = 0.0
        # sort by absolute correlation, descending
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
            # add both directions (GNNs usually use directed edge_index with both directions)
            edges.append((i, j))
            edges.append((j, i))
            weights.append(abs(row[j]))
            weights.append(abs(row[j]))
    if len(edges) == 0:
        # fallback fully connected small weight
        for i in range(F):
            for j in range(F):
                if i!=j:
                    edges.append((i,j)); weights.append(0.1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    # normalize weights to [0,1]
    if edge_weight.max() > 0:
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight

def build_edge_index_from_distance(dist, k=4, sigma=None):
    # dist: (F,F) numpy distances. Convert to similarity via Gaussian kernel.
    if sigma is None:
        # median of upper triangle (nonzero)
        triu = dist[np.triu_indices(dist.shape[0], k=1)]
        sigma = max(np.median(triu[triu > 0.0]) , 1e-6)
    sim = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
    np.fill_diagonal(sim, 0.0)
    # then reuse top-k selection logic
    return build_edge_index_from_corr(sim, k=k, threshold=None)

class TimeWindowsDataset(Dataset):
    """
    Returns raw windows with NaNs preserved.
    Each item: torch.tensor(window) shape (seq_len, num_nodes) with NaNs allowed.
    """
    def __init__(self, X, seq_len=8):
        # X: numpy (T, F)
        self.X = X
        self.seq_len = seq_len
        self.indices = list(range(0, X.shape[0] - seq_len + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        w = self.X[i:i+self.seq_len, :]  # shape (seq_len, F)
        return torch.tensor(w, dtype=torch.float)  # keeps NaNs

def collate_windows(batch):
    # batch: list of tensors (seq_len, F)
    return torch.stack(batch, dim=0)  # (batch, seq_len, F)
