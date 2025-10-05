# train_demo.py
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from utils_and_dataset import (
    load_csv_to_numpy, compute_feature_minmax, minmax_scale_with_nan,
    compute_correlation_matrix, build_edge_index_from_corr,
    TimeWindowsDataset, collate_windows
)
from lstm_gat_imputer import LSTM_GAT_Imputer
from lstm_gcn_imputer import LSTM_GCN_Imputer
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch
import torch.nn as nn

# from utils_and_dataset import inverse_minmax_scale

def inverse_minmax_scale(X_scaled, mins, maxs):
    """
    Обратное преобразование после масштабирования min-max (в диапазон [0,1]).
    
    Аргументы:
        X_scaled : np.ndarray
            Масштабированные данные с возможными NaN.
        mins : np.ndarray
            Минимумы признаков (размерность F).
        maxs : np.ndarray
            Максимумы признаков (размерность F).
    
    Возвращает:
        X_restored : np.ndarray
            Восстановленные значения в исходном масштабе.
    """
    X_restored = X_scaled.copy().astype(float)
    for i in range(X_scaled.shape[1]):
        mask = ~np.isnan(X_scaled[:, i])
        X_restored[mask, i] = X_scaled[mask, i] * (maxs[i] - mins[i]) + mins[i]
    return X_restored

# ========== Пример: сгенерируем synthetic CSV ==========
T = 400
Fi = 10
t = np.arange(T)
data = np.zeros((T, Fi))
for i in range(Fi):
    data[:, i] = np.sin(2 * np.pi * (i+1) * t / 50.0) + 0.2 * np.random.randn(T)
# введём пропуски случайно
rng = np.random.RandomState(0)
mask_missing = rng.rand(*data.shape) < 0.12
data[mask_missing] = np.nan
df_ = pd.DataFrame(data, columns=[f"f{i}" for i in range(Fi)])
csv_path = "/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv"
df_.to_csv(csv_path, index=False)
print("Saved synthetic CSV:", csv_path)

# ========== Load ==========
df_, X_raw, cols = load_csv_to_numpy(csv_path)  # X_raw shape (T, Fi)
df = df_.copy()

# ========== compute min/max and scale to [0,1] ==========
scaler = MinMaxScaler()
scaler.fit(X_raw)   # обучаешь на исходных данных
X_scaled = scaler.transform(X_raw)

# mins, maxs = compute_feature_minmax(X_raw)
# X_scaled = minmax_scale_with_nan(X_raw, mins, maxs)  # keeps NaNs

# ========== build graph (use correlation-based) ==========
corr = compute_correlation_matrix(X_scaled)  # (Fi,Fi)
edge_index, edge_weight = build_edge_index_from_corr(corr, k=3)  # top-3 neighbors

# ========== dataset & dataloader ==========
SEQ = 8
dataset = TimeWindowsDataset(X_scaled, seq_len=SEQ)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_windows)

# ========== choose model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_GAT_Imputer(seq_len=SEQ, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
# or use GCN:
# model = LSTM_GCN_Imputer(seq_len=SEQ, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
mask_prob = 0.2  # fraction of known last-step entries that we will mask for training

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        # batch: (B, SEQ, Fi) with NaNs
        batch = batch.to(device)
        B = batch.shape[0]
        last = batch[:, -1, :]  # ground-truth for last time step (may contain NaNs)
        known_last = ~torch.isnan(last)  # (B, Fi)
        # choose random subset of known_last entries to mask (train targets)
        rnd = torch.rand_like(last)
        mask_pred = (rnd < mask_prob) & known_last  # True where we will try to predict
        # create masked input: set those positions at last time step to NaN
        masked_batch = batch.clone()
        masked_batch[:, -1, :][mask_pred] = float('nan')
        # forward
        preds = model(masked_batch, edge_index, edge_weight)  # (B, Fi), scaled values in [~]
        # compute loss only on mask_pred positions
        if mask_pred.sum() == 0:
            continue
        y_true = last[mask_pred]
        y_pred = preds[mask_pred]
        mse = F.mse_loss(y_pred, y_true)
        # range penalty (scaled ranges are [0,1])
        # clamp predicted outside [0,1]
        lower = 0.0; upper = 1.0
        over = F.relu(y_pred - upper)
        under = F.relu(lower - y_pred)
        range_penalty = (over.mean() + under.mean())
        loss = mse + 5.0 * range_penalty  # hyper param for penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    print(f"Epoch {epoch+1}/{epochs} loss={total_loss/(n_batches+1e-9):.6f}")

# ========== Imputation on the whole series ==========
model.eval()
with torch.no_grad():
    # we'll impute the last-step of each window and fill the original NaNs at t = seq_len-1 .. T-1
    X_filled_scaled = X_scaled.copy()  # numpy with NaNs
    for idx in range(len(dataset)):
        w = dataset[idx]  # (SEQ, Fi) tensor with NaNs
        w_tensor = w.unsqueeze(0).to(device)  # (1, SEQ, Fi)
        preds = model(w_tensor, edge_index, edge_weight)  # (1,Fi)
        preds = preds.squeeze(0).cpu().numpy()
        preds_clamped = np.clip(preds, 0.0, 1.0)
        t_last = idx + SEQ - 1
        # fill only if original is NaN
        for f in range(Fi):
            if np.isnan(X_filled_scaled[t_last, f]):
                X_filled_scaled[t_last, f] = preds_clamped[f]
# inverse transform to original scale
# X_imputed = inverse_minmax_scale(X_filled_scaled, mins, maxs)
X_imputed = scaler.inverse_transform(X_filled_scaled)
# save to CSV
pd.DataFrame(X_imputed, columns=cols).to_csv("/Users/kathrinebovkun/PycharmProjects/experiments/data/imputed_result.csv", index=False)
print("Imputed saved to imputed_result.csv")
