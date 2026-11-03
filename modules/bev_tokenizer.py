from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BEVTokenizer(nn.Module):
    """Keep spatial tokens so a BEV region can be replaced."""

    def __init__(self, feature_dim: int, hidden_dim: int, grid: tuple[int, int], heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.grid = tuple(grid)
        self.tokens = self.grid[0] * self.grid[1]
        self.projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.position = nn.Parameter(torch.randn(1, self.tokens, hidden_dim) * .02)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer, layers)
        self.age = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, feature: Tensor, age_s: Tensor, valid: Tensor, read_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        feature = torch.nan_to_num(feature.float())
        if read_mask is not None:
            source_mask = F.interpolate(read_mask.reshape(read_mask.shape[0], 1, *self.grid).float(), size=feature.shape[-2:], mode="nearest")
            feature = feature * source_mask
        tokens = F.adaptive_avg_pool2d(self.projection(feature), self.grid).flatten(2).transpose(1, 2) + self.position
        valid = F.adaptive_max_pool2d(valid.float().unsqueeze(1), self.grid).flatten(1).bool()
        ratio = valid.float().mean(-1)
        tokens = tokens + self.age(torch.stack((torch.log1p(age_s.clamp_min(0)), ratio), -1))[:, None]
        if read_mask is not None:
            valid = valid & read_mask
            tokens = tokens * read_mask[..., None]
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(-1)
        tokens = self.spatial(tokens, src_key_padding_mask=~safe)
        return tokens * valid[..., None], valid

