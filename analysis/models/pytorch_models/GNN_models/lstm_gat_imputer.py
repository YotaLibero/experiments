# lstm_gat_imputer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from edge_gat_conv import EdgeGATConv

class LSTM_GAT_Imputer(nn.Module):
    """
    Input:
      windows: (batch, seq_len, num_nodes) with NaNs preserved
    Flow:
      - build input for LSTM per node: (value_filled, observed_flag) as 2 channels
      - LSTM processes each node's sequence -> node embedding
      - project node embeddings -> GNN input, batch several windows together by concatenation
      - Edge-aware GAT aggregates and outputs scalar prediction per node (last time)
    """
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
        """
        windows: (B, S, Fi) torch tensor with NaNs
        edge_index: (2, E) long
        edge_weight: (E,) float
        returns:
          preds: (B, Fi) predicted values (scaled) at last time step
        """
        B, S, Fi = windows.shape
        assert Fi == self.num_nodes
        # Build LSTM input: for each sample and for each node we create a (S, 2) sequence
        values = windows.clone()
        observed = ~torch.isnan(values)  # bool tensor (B,S,Fi)
        # Fill NaNs with 0.0 for numerical input
        values_filled = torch.nan_to_num(values, nan=0.0)
        # Prepare channels: (B, Fi, S, 2) -> then reshape to (B*Fi, S, 2) for LSTM processing
        ch0 = values_filled.permute(0, 2, 1).unsqueeze(-1)  # (B, Fi, S, 1)
        ch1 = observed.permute(0, 2, 1).float().unsqueeze(-1)  # (B, Fi, S, 1)
        node_seq = torch.cat([ch0, ch1], dim=-1)  # (B, Fi, S, 2)
        node_seq = node_seq.view(B * Fi, S, 2)  # (B*Fi, S, 2)
        # LSTM
        lstm_out, _ = self.lstm(node_seq)  # (B*Fi, S, lstm_hidden)
        last = lstm_out[:, -1, :]  # (B*Fi, lstm_hidden)
        last = last.view(B, Fi, -1)  # (B, Fi, lstm_hidden)
        # project
        x_gnn = self.proj(last)  # (B, Fi, gnn_hidden)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)
        # prepare batched graph: concat nodes from all B graphs
        x_cat = x_gnn.view(B * Fi, -1)  # (B*Fi, gnn_hidden)
        # replicate edge_index and edge_weight B times with offsets
        device = x_cat.device
        ei = edge_index.to(device)
        ew = edge_weight.to(device)
        edge_indices = []
        for b in range(B):
            edge_indices.append(ei + b * Fi)
        edge_index_batch = torch.cat(edge_indices, dim=1)  # (2, E*B)
        edge_weight_batch = ew.repeat(B)  # (E*B,)
        # run GAT
        gnn_out = self.gat(x_cat, edge_index_batch, edge_weight_batch)  # (B*Fi, gnn_hidden)
        gnn_out = F.relu(gnn_out)
        preds = self.out(gnn_out).view(B, Fi)  # (B, Fi), scalar per node
        return preds
