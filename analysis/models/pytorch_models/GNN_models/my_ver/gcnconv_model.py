"""
simple_gnn_imputer.py

Простейший GNN-imputer на основе GCNConv (PyTorch Geometric).

Идея (упрощённо):
 - У нас есть граф (N узлов) и у каждого узла вектор признаков длины F.
 - Некоторые элементы матрицы признаков отсутствуют (NaN). Задача — их заполнить (impute).
 - Мы даём пользователю возможность вручную задать:
     1) матрицу корреляций между узлами `node_corrs` (NxN) — будет преобразована в `edge_index` и `edge_weight`;
     2) матрицу корреляций между признаками `feat_corr` (FxF) — используется для линейного смешивания признаков перед агрегацией;
     3) вектор весов по признакам `feat_weights` (F,) — умножение по признакам (importance).
 - Модель очень простая: encoder (Linear) -> GCNConv -> decoder (Linear).
 - Потеря (loss) считается только по наблюдаемым элементам (mask).

Файл содержит:
 - минимальные тулкиты для преобразования матрицы корреляций в edges;
 - класс `SimpleGNNImputer`;
 - небольшой пример генерации данных и обучения (в `if __name__ == "__main__"`).

Зависимости: torch, torch_geometric

Примечание: это минимальная обучающая версия. Для продакшна рекомендуется:
 - учитывать нормализацию признаков, регуляризацию, более глубокую архитектуру;
 - аккуратно обрабатывать отсутствие признаков в neighbour-агрегации (например, нормировать по реальному количеству наблюдений);
 - сделать модель способной работать батчами / с динамическими графами.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


def build_edge_index_and_weight(node_corrs: torch.Tensor, threshold: float = 0.0):
    """
    Преобразует плотную матрицу корреляций (NxN) в edge_index и edge_weight.
    - node_corrs: симметричная матрица (torch.Tensor, shape [N,N]).
    - threshold: порог фильтрации мелких весов (по абсолютному значению).
    Возвращает (edge_index, edge_weight) — в формате torch_geometric.
    """
    assert node_corrs.dim() == 2 and node_corrs.size(0) == node_corrs.size(1), "node_corrs must be square"
    adj = node_corrs.clone().float()
    if threshold > 0.0:
        mask = (adj.abs() > threshold)
        adj = adj * mask.float()
    # Уберём диагональ — GCNConv обычно сам добавляет self-loops
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)
    return edge_index, edge_weight


class SimpleGNNImputer(nn.Module):
    """
    Очень простая модель imputer на GCNConv.

    forward(x, edge_index=None, edge_weight=None, node_corrs=None,
            feat_corr=None, feat_weights=None, mask=None)
    - x: [N, F] torch.Tensor, может содержать NaN для отсутствующих значений
    - edge_index/edge_weight: если заданы, используются напрямую
    - node_corrs: если задана (NxN), используется для построения edge_index/edge_weight
    - feat_corr: (F,F) матрица корреляций между признаками (опционально)
    - feat_weights: (F,) вектор весов по признакам (опционально)
    - mask: булев массив того же размера что x: True для наблюдаемых значений
    """

    def __init__(self, in_feats: int, hidden: int = 32):
        super().__init__()
        self.enc = nn.Linear(in_feats, hidden)
        self.gcn = GCNConv(hidden, hidden)
        self.dec = nn.Linear(hidden, in_feats)

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None,
                node_corrs: torch.Tensor = None, feat_corr: torch.Tensor = None,
                feat_weights: torch.Tensor = None, mask: torch.Tensor = None):
        # x: [N, Fi]
        if mask is None:
            mask = ~torch.isnan(x)
        # Заполним NaN нулями — модель увидит отсутствующие, но не "отравит" численными NaN
        x_filled = torch.where(mask, x, torch.zeros_like(x))

        # Смешивание признаков через матрицу корреляций признаков
        if feat_corr is not None:
            # ожидаем shape (Fi, Fi)
            x_filled = x_filled @ feat_corr.T

        # Весы по признакам
        if feat_weights is not None:
            # feat_weights: (Fi,)
            x_filled = x_filled * feat_weights.view(1, -1)

        # Построим ребра из матрицы корреляций между узлами, если нужно
        if node_corrs is not None and edge_index is None:
            edge_index, edge_weight = build_edge_index_and_weight(node_corrs)

        if edge_index is None:
            raise ValueError("edge_index (or node_corrs) must be provided")

        # Encoder -> GCNConv -> Decoder
        h = F.relu(self.enc(x_filled))
        h = F.relu(self.gcn(h, edge_index, edge_weight))
        recon = self.dec(h)
        return recon


# ---------------------- Пример использования ----------------------
if __name__ == "__main__":
    # Генерируем маленький пример для демонстрации
    torch.manual_seed(0)
    N = 6   # число узлов
    Fi = 4   # число признаков

    # Истинные данные (их мы хотим восстановить)
    X_true = torch.randn(N, Fi)

    # Случайно делаем некоторые значения пропущенными
    observed_mask = (torch.rand(N, Fi) > 0.3)
    X_obs = X_true.clone()
    X_obs[~observed_mask] = float('nan')

    # Простая матрица корреляций между узлами: чем ближе индексы, тем сильнее связь
    idx = torch.arange(N)
    d = torch.abs(idx.view(-1, 1) - idx.view(1, -1)).float()
    node_corrs = 1.0 / (1.0 + d)  # значения в диапазоне (0,1]
    # Отфильтруем очень маленькие значения (сделаем граф разреженным)
    node_corrs = node_corrs * (node_corrs > 0.2).float()

    # Матрица корреляций между признаками (можно задать вручную)
    feat_corr = torch.eye(Fi) * 1.0 + 0.1 * (torch.ones(Fi, Fi) - torch.eye(Fi))

    # Веса по признакам — укажем явно
    feat_weights = torch.tensor([1.0, 0.5, 0.2, 2.0], dtype=torch.float32)

    # Построим edge_index и edge_weight
    edge_index, edge_weight = build_edge_index_and_weight(node_corrs, threshold=0.0)

    # Модель
    model = SimpleGNNImputer(in_feats=Fi, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Тренируем всего несколько эпох — это демонстрация, не реализация для продакшна
    X_obs_tensor = X_obs
    mask_tensor = observed_mask

    for epoch in range(1, 401):
        model.train()
        recon = model(X_obs_tensor, edge_index=edge_index, edge_weight=edge_weight,
                      feat_corr=feat_corr, feat_weights=feat_weights, mask=mask_tensor)
        # loss: MSE только по наблюдаемым элементам
        mse = ((recon - X_true)**2 * mask_tensor.float()).sum() / mask_tensor.float().sum()
        opt.zero_grad()
        mse.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d}  masked MSE = {mse.item():.6f}")

    # Получаем итоговую импутацию
    model.eval()
    with torch.no_grad():
        recon_final = model(X_obs_tensor, edge_index=edge_index, edge_weight=edge_weight,
                            feat_corr=feat_corr, feat_weights=feat_weights, mask=mask_tensor)

    X_imputed = X_obs_tensor.clone()
    X_imputed[~mask_tensor] = recon_final[~mask_tensor]

    print("\nX_true:")
    print(X_true)
    print("\nX_obs (with NaNs):")
    print(X_obs)
    print("\nX_imputed (replaced NaNs with model output):")
    print(X_imputed)

    # Простой подсчёт MSE на ранее пропущенных элементах (контроль)
    missing_mask = ~mask_tensor
    if missing_mask.float().sum() > 0:
        missing_mse = ((X_imputed - X_true)**2 * missing_mask.float()).sum() / missing_mask.float().sum()
        print(f"\nMSE on previously missing elements = {missing_mse.item():.6f}")
    else:
        print("No missing elements to evaluate.")


# -----------------------------------------------------------------
# Комментарии и возможности расширения:
# - Сейчас мы заменяем NaN нулём перед подачей в сеть. В реальных задачах полезнее
#   заменять на средние по признаку или использовать mask-aware aggregation (чтобы
#   соседние узлы не передавали нули как реальные значения).
# - Можно расширить модель: несколько GCN-слоёв, attention (GAT), нормализация, skip-connections.
# - Для больших графов стоит контролировать разреженность edge_weight и порог фильтрации.
# - Если вы хотите, я могу добавить: per-feature normalization, обработку батчей, поддержку динамических графов,
#   или версию, которая строит граф автоматически от признаков (feature-similarity graph).
# -----------------------------------------------------------------
