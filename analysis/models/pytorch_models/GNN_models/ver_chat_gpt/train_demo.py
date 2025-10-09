# =============================================
# train_demo.py — GNN Imputer (с ручными и корреляционными связями)
# =============================================
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

# --- Импорт утилит и моделей ---
from utils_and_dataset import (
    load_csv_to_numpy, compute_feature_minmax, minmax_scale_with_nan,
    compute_correlation_matrix, build_edge_index_from_corr,
    TimeWindowsDataset, collate_windows, inverse_minmax_scale
)
from lstm_gat_imputer import LSTM_GAT_Imputer
from lstm_gcn_imputer import LSTM_GCN_Imputer

# =============================================
# Функции для ручного графа
# =============================================

def build_manual_graph(weight_matrix: np.ndarray):
    """
    Создаёт edge_index и edge_weight из заданной матрицы влияния признаков.
    weight_matrix[i, j] — сила влияния признака j на i.
    """
    src, dst, w = [], [], []
    Fi = weight_matrix.shape[0]
    for i in range(Fi):
        for j in range(Fi):
            if weight_matrix[i, j] > 0:
                src.append(j)
                dst.append(i)
                w.append(weight_matrix[i, j])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


def visualize_graph(edge_index, edge_weight, Fi):
    """Визуализация графа через NetworkX"""
    G = nx.DiGraph()
    for (u, v, w) in zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist()):
        G.add_edge(u, v, weight=round(w, 2))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})
    plt.title("Graph of Feature Relations")
    plt.show()


# =============================================
# Основная функция обучения
# =============================================

def train_demo(csv_path, seq_len=8, batch_size=16, epochs=20, use_gat=True, use_manual_graph=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ========== Загрузка данных ==========
    df_, X_raw, cols = load_csv_to_numpy(csv_path)
    T, Fi = X_raw.shape
    print(f"Loaded CSV with shape {T}×{Fi}")

    # ========== Масштабирование ==========
    mins, maxs = compute_feature_minmax(X_raw)
    X_scaled = minmax_scale_with_nan(X_raw, mins, maxs)

    # ========== Построение графа ==========
    if use_manual_graph:
        print("Using manual (expert-defined) graph...")
        manual_weights = np.array([
            # f0  f1   f2   f3   f4   f5   f6   f7   f8   f9
            [0.0, 0.8, 0.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # f0 <- f1,f2,f3
            [0.8, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # f1 <- f0,f2
            [0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # f2 <- f0,f1
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],  # f3 <- f4
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # f4 <- -
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0, 0.0],  # f5 <- f6,f7
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],  # f6 <- f5
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0],  # f7 <- f5,f6
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],  # f8 <- f9
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0],  # f9 <- f8
        ])
        edge_index, edge_weight = build_manual_graph(manual_weights)
    else:
        print("Using correlation-based graph...")
        corr = compute_correlation_matrix(X_scaled)
        edge_index, edge_weight = build_edge_index_from_corr(corr, k=3)

    print("Graph built:", edge_index.shape, edge_weight.shape)
    visualize_graph(edge_index, edge_weight, Fi)

    # ========== Подготовка датасета ==========
    dataset = TimeWindowsDataset(X_scaled, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_windows)

    # ========== Инициализация модели ==========
    if use_gat:
        model = LSTM_GAT_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)
    else:
        model = LSTM_GCN_Imputer(seq_len=seq_len, num_nodes=Fi, lstm_hidden=64, gnn_hidden=64).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    epochs = epochs
    mask_prob = 0.2  # сколько значений будем маскировать для обучения

    # ========== Обучение ==========
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            B = batch.shape[0]
            last = batch[:, -1, :]
            known_last = ~torch.isnan(last)
            rnd = torch.rand_like(last)
            mask_pred = (rnd < mask_prob) & known_last

            masked_batch = batch.clone()
            masked_batch[:, -1, :][mask_pred] = float('nan')

            preds = model(masked_batch, edge_index, edge_weight)
            if mask_pred.sum() == 0:
                continue
            y_true = last[mask_pred]
            y_pred = preds[mask_pred]
            mse = F.mse_loss(y_pred, y_true)

            # штраф за выход за диапазон [0,1]
            lower, upper = 0.0, 1.0
            over = F.relu(y_pred - upper)
            under = F.relu(lower - y_pred)
            range_penalty = (over.mean() + under.mean())

            loss = mse + 5.0 * range_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/(n_batches+1e-9):.6f}")

    # ========== Импутация ==========
    model.eval()
    with torch.no_grad():
        X_filled_scaled = X_scaled.copy()
        for idx in range(len(dataset)):
            w = dataset[idx]
            w_tensor = w.unsqueeze(0).to(device)
            preds = model(w_tensor, edge_index, edge_weight).squeeze(0).cpu().numpy()
            preds_clamped = np.clip(preds, 0.0, 1.0)
            t_last = idx + seq_len - 1
            for f in range(Fi):
                if np.isnan(X_filled_scaled[t_last, f]):
                    X_filled_scaled[t_last, f] = preds_clamped[f]

    X_imputed = inverse_minmax_scale(X_filled_scaled, mins, maxs)
    out_path = csv_path.replace(".csv", "_imputed.csv")
    pd.DataFrame(X_imputed, columns=cols).to_csv(out_path, index=False)
    print("Imputed data saved to:", out_path)


# =============================================
# Запуск
# =============================================
if __name__ == "__main__":
    csv_path = "/Users/kathrinebovkun/PycharmProjects/experiments/data/Industrial_fault_detection_with_nans.csv"
    train_demo(csv_path, seq_len=8, batch_size=32, epochs=50, use_gat=True, use_manual_graph=True)
