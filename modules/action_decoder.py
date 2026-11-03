from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionDecoder(nn.Module):
    """Predict the next action from current visual memory and past actions."""

    def __init__(self, hidden_dim: int, future_action: int, heads: int) -> None:
        super().__init__()
        self.future_action = future_action
        self.step = nn.Embedding(future_action, hidden_dim)
        self.action = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.state = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, memory: Tensor, valid: Tensor, previous: Tensor,
                hidden: Tensor | None = None, step: int = 0) -> tuple[Tensor, Tensor]:
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(-1)
        if hidden is None:
            hidden = (memory * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        query = self.action(previous) + self.step.weight[step % self.future_action]
        context, _ = self.cross(query[:, None], memory, memory, key_padding_mask=~safe)
        hidden = self.state(context[:, 0], hidden)
        return self.output(hidden), hidden

