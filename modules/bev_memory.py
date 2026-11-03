from __future__ import annotations

import torch
from torch import Tensor, nn


class BEVMemory(nn.Module):
    """Update every spatial token independently while retaining region identity."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, hidden: Tensor, frame: Tensor, valid: Tensor) -> Tensor:
        batch, points, dim = frame.shape
        updated = self.update(frame.reshape(batch * points, dim), hidden.reshape(batch * points, dim)).reshape(batch, points, dim)
        return torch.where(valid[..., None], updated, hidden)

