from __future__ import annotations

import torch
from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F


class FreshModule(nn.Module):
    """Predict signed region utility through an evidence-gap bottleneck."""

    def __init__(self, hidden_dim: int, regions: int) -> None:
        super().__init__()
        self.plan = nn.Linear(hidden_dim, hidden_dim)
        self.gap = nn.Sequential(nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.gap_score = nn.Linear(hidden_dim, 1)
        self.utility = nn.Sequential(nn.Linear(hidden_dim * 3 + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.regions = regions

    def forward(self, memory: Tensor, valid: Tensor, age_s: Tensor, queries: Tensor, costs: Tensor, budget: Tensor, region_ids: Tensor) -> dict[str, Tensor]:
        plan = self.plan(queries.mean(1))[:, None]
        pooled, region_valid, region_age = [], [], []
        for region in range(self.regions):
            mask = region_ids == region
            weight = (valid & mask).float()
            pooled.append((memory * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1))
            region_valid.append(weight.sum(1) > 0)
            region_age.append(torch.where(valid & mask, age_s, torch.zeros_like(age_s)).amax(1))
        region_memory, region_valid, region_age = torch.stack(pooled, 1), torch.stack(region_valid, 1), torch.stack(region_age, 1)
        metadata = torch.stack((torch.log1p(region_age.clamp_min(0)), region_valid.float(), costs, budget[:, None].expand_as(costs)), -1)
        expanded_plan = plan.expand(-1, self.regions, -1)
        gap = self.gap(torch.cat((region_memory, expanded_plan, metadata), -1))
        utility = self.utility(torch.cat((region_memory, expanded_plan, gap, metadata), -1)).squeeze(-1)
        return {"evidence_gap": gap, "gap_score": self.gap_score(gap).squeeze(-1), "refresh_utility": utility, "region_valid": region_valid}

    @staticmethod
    def select(utility: Tensor, available: Tensor, costs: Tensor, budget: Tensor, cost_weight: float) -> Tensor:
        feasible = available & (costs <= budget[:, None])
        score = utility - cost_weight * costs
        score = score.masked_fill(~feasible, float("-inf"))
        best, index = score.max(-1)
        selected = torch.zeros_like(utility, dtype=torch.bool)
        selected.scatter_(1, index[:, None], True)
        return selected & torch.isfinite(best)[:, None] & (best[:, None] > 0)

    def region_available(self, valid: Tensor, region_ids: Tensor, grid: tuple[int, int]) -> Tensor:
        """Map a raw valid mask to the nine candidate regions without reading features."""
        points = F.adaptive_max_pool2d(valid.float().unsqueeze(1), grid).flatten(1).bool()
        return torch.stack([(points & (region_ids == region)).any(-1) for region in range(self.regions)], 1)

    def update_anchor(self, state: dict[str, Tensor], anchor: Tensor, anchor_valid: Tensor, anchor_age: Tensor, selected: Tensor, region_ids: Tensor, tokenizer, memory) -> dict[str, Tensor]:
        read_mask = selected[:, region_ids]
        tokens, frame_valid = tokenizer(anchor.float(), anchor_age, anchor_valid, read_mask)
        update_valid = frame_valid & selected[:, region_ids]
        memory = memory(state["memory"], tokens, update_valid)
        return {"memory": memory, "valid": state["valid"] | update_valid, "age_s": torch.where(update_valid, anchor_age[:, None], state["age_s"])}

    @torch.no_grad()
    def fresh_targets(self, state, feature, valid, target, target_mask, previous,
                      proposal, horizon, tokenizer, memory, decoder, region_ids):
        """Five-step ADE improvement from acquiring one real BEV region."""
        from modules.rolling_planner import RollingPlanner
        from modules.action_loss import ActionLoss

        baseline = ActionLoss.trajectory_error(ActionLoss.trajectory(proposal), target, target_mask)
        utilities = []
        for region in range(self.regions):
            selected = torch.zeros(feature.shape[0], self.regions, dtype=torch.bool, device=feature.device)
            selected[:, region] = True
            updated = self.update_anchor(state, feature, valid, feature.new_zeros(feature.shape[0], dtype=torch.float32),
                                         selected, region_ids, tokenizer, memory)
            actions, _ = RollingPlanner.decode(updated, previous, decoder, horizon)
            error = ActionLoss.trajectory_error(ActionLoss.trajectory(actions), target, target_mask)
            utilities.append(baseline - error)
        return torch.stack(utilities, -1)
