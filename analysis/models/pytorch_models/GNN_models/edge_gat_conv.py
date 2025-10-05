# edge_gat_conv.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class EdgeGATConv(MessagePassing):
    """
    Simplified edge-aware single-head GAT:
    e_ij = a^T [W x_i || W x_j]
    e_ij <- e_ij * edge_weight_ij (if provided)
    alpha_ij = softmax_j( LeakyReLU(e_ij) )
    message = alpha_ij * (W x_j)
    """
    def __init__(self, in_channels, out_channels, negative_slope=0.2, bias=True):
        super().__init__(aggr='add')  # sum aggregation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * out_channels))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (N, in_channels)
        edge_index: (2, E)
        edge_weight: (E,) optional (should be >=0 and small)
        """
        # Linear transform
        x_l = self.lin(x)  # (N, out)
        # We will compute e for each edge
        row, col = edge_index  # convention: row=source, col=target
        x_row = x_l[row]  # source embeddings (E, out)
        x_col = x_l[col]  # target embeddings (E, out)
        # concat
        feat_cat = torch.cat([x_row, x_col], dim=-1)  # (E, 2*out)
        e = (feat_cat * self.att).sum(dim=-1)  # (E,)
        if edge_weight is not None:
            # Ensure same dtype & shape
            ew = edge_weight.view(-1).to(e.dtype)
            e = e * ew
        # attention coefficients per target node (softmax over incoming edges)
        alpha = softmax(self.leaky_relu(e), col)  # normalize across edges with same target (col)
        # propagate messages (source node features x_row)
        out = self.propagate(edge_index, x=x_l, alpha=alpha)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, alpha):
        # x_j: (E, out), alpha: (E,)
        return x_j * alpha.view(-1, 1)
