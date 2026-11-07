"""BEV timing, historical state initialization and coordinate alignment."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class TemporalBEVContext:
    """Manage temporal geometry without owning trainable network parameters."""

    def __init__(self, history_step: int, hidden_dim: int, grid: tuple[int, int]):
        self.history_step = history_step
        self.hidden_dim = hidden_dim
        self.grid = tuple(grid)

    def ages(self, features: Tensor, timestamps_us: Tensor | None, anchor_timestamp_us: Tensor | None, bev_age_s: Tensor | None) -> Tensor:
        batch, steps = features.shape[:2]
        if bev_age_s is not None:
            ages = bev_age_s.float().clamp_min(0)
            return ages[:, :steps]
        if timestamps_us is not None:
            times = timestamps_us.to(device=features.device, dtype=torch.float64)
            if times.ndim == 3:
                times = times.amax(-1)
            anchor = times[:, self.history_step] if anchor_timestamp_us is None else torch.as_tensor(anchor_timestamp_us, device=features.device, dtype=torch.float64)
            return ((anchor[:, None] - times[:, :steps]) / 1e6).float()
        slots = torch.arange(steps, device=features.device).float()
        return (self.history_step - slots)[None].expand(batch, -1) * .1

    def encode_history(self, features: Tensor, valid: Tensor, ages: Tensor, tokenizer, memory) -> dict[str, Tensor]:
        batch, steps = features.shape[:2]
        points = self.grid[0] * self.grid[1]
        hidden = features.new_zeros(batch, points, self.hidden_dim, dtype=torch.float32)
        cache_valid = torch.zeros(batch, points, dtype=torch.bool, device=features.device)
        cache_age = torch.full((batch, points), 1e6, dtype=torch.float32, device=features.device)
        for step in range(steps):
            tokens, frame_valid = tokenizer(features[:, step], ages[:, step], valid[:, step])
            hidden = memory(hidden, tokens, frame_valid)
            cache_valid |= frame_valid
            cache_age = torch.where(frame_valid, ages[:, step, None], cache_age)
        return {"memory": hidden, "valid": cache_valid, "age_s": cache_age}

    def align_memory(self, state: dict[str, Tensor], source_pose: Tensor,
                     destination_pose: Tensor, bev_range: Tensor) -> dict[str, Tensor]:
        """Warp old spatial memory into the arrived frame using current localization."""
        batch = source_pose.shape[0]
        bounds = torch.as_tensor(bev_range, device=source_pose.device, dtype=torch.float32)
        if bounds.ndim == 1:
            bounds = bounds[None].expand(batch, -1)
        rows = (torch.arange(self.grid[0], device=source_pose.device) + .5) / self.grid[0]
        cols = (torch.arange(self.grid[1], device=source_pose.device) + .5) / self.grid[1]
        y, x = torch.meshgrid(rows, cols, indexing="ij")
        points = torch.ones(batch, self.grid[0] * self.grid[1], 4, device=source_pose.device)
        points[..., 0] = bounds[:, 0, None] + x.flatten() * (bounds[:, 3] - bounds[:, 0])[:, None]
        points[..., 1] = bounds[:, 1, None] + y.flatten() * (bounds[:, 4] - bounds[:, 1])[:, None]
        points[..., 2] = 0
        transform = torch.linalg.solve(source_pose.double(), destination_pose.double()).float()
        source = points @ transform.transpose(-1, -2)
        grid = torch.stack((2 * (source[..., 0] - bounds[:, 0, None]) / (bounds[:, 3] - bounds[:, 0])[:, None] - 1,
                            2 * (source[..., 1] - bounds[:, 1, None]) / (bounds[:, 4] - bounds[:, 1])[:, None] - 1), -1)
        grid = grid.reshape(batch, *self.grid, 2)
        valid = F.grid_sample(state["valid"].reshape(batch, 1, *self.grid).float(), grid,
                              mode="nearest", align_corners=False).flatten(1).bool()
        inside = (grid.abs() <= 1).all(-1).flatten(1)
        valid &= inside
        # Invalid cells must not leak into valid neighbors through interpolation.
        memory = (state["memory"] * state["valid"][..., None]).transpose(1, 2).reshape(batch, -1, *self.grid)
        memory = F.grid_sample(memory, grid, align_corners=False).flatten(2).transpose(1, 2)
        age = F.grid_sample(state["age_s"].reshape(batch, 1, *self.grid), grid,
                            mode="nearest", align_corners=False).flatten(1)
        return {"memory": memory * valid[..., None], "valid": valid,
                "age_s": torch.where(valid, age, torch.full_like(age, 1e6))}
