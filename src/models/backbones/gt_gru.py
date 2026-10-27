# Graph-conditioned, time-aware GRU cell.

import torch
import torch.nn as nn
from typing import Optional

class GTGRUCell(nn.Module):
    """
    Graph-Conditioned, Time-Aware GRU Cell.
    The target embedding is injected into each gate.
    """
    def __init__(self, input_size: int, hidden_size: int, context_size: int, time_emb_dim: int):
        super(GTGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.time_emb_dim = time_emb_dim

        c_target_size = 32
        self.T_r = nn.Linear(c_target_size, hidden_size, bias=False)
        self.T_z = nn.Linear(c_target_size, hidden_size, bias=False)
        self.T_h = nn.Linear(c_target_size, hidden_size, bias=False)

        # for reset gate (r_t)
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_r = nn.Linear(context_size, hidden_size, bias=False)
        self.B_r = nn.Linear(time_emb_dim, hidden_size, bias=False)
        
        # for update gate (z_t)
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_z = nn.Linear(context_size, hidden_size, bias=False)
        self.B_z = nn.Linear(time_emb_dim, hidden_size, bias=False)

        # for candidate hidden state (h_tilde_t)
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_h = nn.Linear(context_size, hidden_size, bias=False)
        self.B_h = nn.Linear(time_emb_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, context: torch.Tensor, time_embed: torch.Tensor, c_target: torch.Tensor) -> torch.Tensor:
        r_t = torch.sigmoid(self.W_r(x) + self.U_r(h_prev) + self.V_r(context) + self.B_r(time_embed) + self.T_r(c_target))
        z_t = torch.sigmoid(self.W_z(x) + self.U_z(h_prev) + self.V_z(context) + self.B_z(time_embed) + self.T_z(c_target))
        h_tilde_t = torch.tanh(self.W_h(x) + self.U_h(r_t * h_prev) + self.V_h(context) + self.B_h(time_embed) + self.T_h(c_target))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        return h_t
