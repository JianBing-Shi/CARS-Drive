from __future__ import annotations
import torch
from torch import Tensor, nn

class ActionEncoder(nn.Module):
    """Encode causal local actions aligned with historical BEV frames."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 3 or actions.shape[1] < 1:
            raise ValueError('history_actions must have shape [batch, history_step, 3]')
        _, hidden = self.gru(self.input(torch.nan_to_num(actions.float())))
        return self.norm(hidden[-1])
