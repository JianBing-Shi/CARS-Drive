"""Binary trajectory replacement using arrived BEV and the retained plan."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from modules.fresh_module import FreshModule


class RefreshGate(nn.Module):
    """0 keeps the retained plan; 1 requests a full valid-BEV update."""
    region_available = FreshModule.region_available
    update_anchor = FreshModule.update_anchor

    def __init__(self, hidden_dim: int, regions: int, horizon: int):
        super().__init__()
        self.regions = regions
        self.horizon = horizon
        self.visual = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.plan = nn.Sequential(nn.Linear(horizon * 4, hidden_dim), nn.GELU())
        self.context = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 3 + 5, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(self, state, tokens, token_valid, retained_plan, plan_mask, previous_action, budget):
        def pool(values, mask):
            return values.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        old = pool(state['memory'], state['valid'])
        current = pool(tokens, token_valid)
        visual = self.visual(torch.cat((old, current, current - old), -1))
        plan = retained_plan.masked_fill(~plan_mask[..., None], 0)
        plan = self.plan(torch.cat((plan, plan_mask[..., None].float()), -1).flatten(1))
        age = state['age_s'].masked_fill(~state['valid'], 0).sum(1) / state['valid'].sum(1).clamp_min(1)
        metadata = torch.stack((plan_mask.float().mean(1), torch.log1p(age.clamp_min(0)),
                                state['valid'].float().mean(1), token_valid.float().mean(1), budget), -1)
        return self.classifier(torch.cat((visual, plan, self.context(previous_action), metadata), -1))

    @staticmethod
    def replacement_labels(keep, candidate, target, common_mask, feasible, epsilon, cost_weight, cost):
        """GT only scores equal-time endpoints; it never enters the classifier."""
        from modules.action_loss import ActionLoss
        with torch.no_grad():
            common = common_mask.long().cumprod(-1).bool()
            keep_error = ActionLoss.trajectory_error(keep, target, common)
            new_error = ActionLoss.trajectory_error(candidate, target, common)
            gain = keep_error - new_error
            valid = common.any(-1) & feasible
            labels = (gain > epsilon + cost_weight * cost).long()
        return labels, valid, gain
