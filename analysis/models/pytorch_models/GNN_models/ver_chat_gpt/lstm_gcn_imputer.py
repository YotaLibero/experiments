# lstm_gcn_imputer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
        B, S, F = windows.shape
        assert F == self.num_nodes
        values = windows.clone()
        observed = ~torch.isnan(values)
        values_filled = torch.nan_to_num(values, nan=0.0)
        ch0 = values_filled.permute(0, 2, 1).unsqueeze(-1)
        ch1 = observed.permute(0, 2, 1).float().unsqueeze(-1)
        node_seq = torch.cat([ch0, ch1], dim=-1)
        node_seq = node_seq.view(B * F, S, 2)
        lstm_out, _ = self.lstm(node_seq)
        last = lstm_out[:, -1, :]
        last = last.view(B, F, -1)
        x_gnn = self.proj(last)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)
        x_cat = x_gnn.view(B * F, -1)
        # batch edges
        device = x_cat.device
        ei = edge_index.to(device)
        ew = edge_weight.to(device)
        edge_indices = []
        for b in range(B):
            edge_indices.append(ei + b * F)
        edge_index_batch = torch.cat(edge_indices, dim=1)
        edge_weight_batch = ew.repeat(B)
        gnn_out = self.gcn(x_cat, edge_index_batch, edge_weight_batch)
        gnn_out = F.relu(gnn_out)
        preds = self.out(gnn_out).view(B, F)
        return preds
