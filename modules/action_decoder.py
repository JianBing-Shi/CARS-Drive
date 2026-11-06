from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionDecoder(nn.Module):
    """Predict the next action from current visual memory and past actions."""

    def __init__(self, hidden_dim: int, future_action: int, heads: int,
                 architecture_version: int = 1, layers: int = 2, dropout: float = 0.) -> None:
        super().__init__()
        if architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        if layers < 1:
            raise ValueError("decoder layers must be positive")
        self.architecture_version = architecture_version
        self.future_action = future_action
        if architecture_version == 2:
            self.action = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            self.visual_norm = nn.LayerNorm(hidden_dim)
            self.initial = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
            self.query = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
            layer = nn.TransformerDecoderLayer(hidden_dim, heads, hidden_dim * 4,
                                                dropout, batch_first=True, norm_first=True)
            self.reader = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(hidden_dim))
            self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU())
            self.state = nn.GRUCell(hidden_dim, hidden_dim)
            self.translation = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                             nn.GELU(), nn.Linear(hidden_dim, 2))
            self.rotation = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                          nn.GELU(), nn.Linear(hidden_dim, 1))
            return
        # Exact legacy parameter names and computation for existing checkpoints.
        self.step = nn.Embedding(future_action, hidden_dim)
        self.action = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.state = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, memory: Tensor, valid: Tensor, previous: Tensor,
                hidden: Tensor | None = None, step: int = 0) -> tuple[Tensor, Tensor]:
        if self.architecture_version == 2:
            # Mask before normalization: invalid cells (including NaNs) are never evidence.
            visual = self.visual_norm(memory.masked_fill(~valid[..., None], 0))
            visual = visual.masked_fill(~valid[..., None], 0)
            safe = valid.clone()
            safe[:, 0] |= ~safe.any(-1)
            pooled = visual.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            previous = torch.cat((previous[..., :2],
                                  torch.atan2(previous[..., 2:].sin(), previous[..., 2:].cos())), -1)
            action = self.action(previous)
            if hidden is None:
                hidden = self.initial(torch.cat((pooled, action), -1))
            query = self.query(torch.cat((hidden, action), -1))
            context = self.reader(query[:, None], visual, memory_key_padding_mask=~safe)[:, 0]
            fused = self.fusion(torch.cat((context, action, pooled), -1))
            hidden = self.state(fused, hidden)
            # One continuous local SE(2) increment per call; no kinematic shortcut.
            return torch.cat((self.translation(hidden), self.rotation(hidden)), -1), hidden
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(-1)
        if hidden is None:
            hidden = (memory * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        query = self.action(previous) + self.step.weight[step % self.future_action]
        context, _ = self.cross(query[:, None], memory, memory, key_padding_mask=~safe)
        hidden = self.state(context[:, 0], hidden)
        return self.output(hidden), hidden

