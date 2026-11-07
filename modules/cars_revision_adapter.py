"""Plan-conditioned latent revision and candidate acceptance from causal BEV.

This module owns only the new refresh parameters. The existing tokenizer,
memory update and one-step action decoder remain separate, frozen components.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from torch.nn import functional as F
from modules.loss import TrajectoryL1Loss


class CARSRevisionAdapter(nn.Module):
    """Revise a planning latent, then score keep/new on common future times.

    ``context`` receives arrived observations only. ``adapt`` is differentiable
    with respect to these new parameters, including through a frozen decoder.
    ``compare`` returns two logits ordered as ``[keep, replace]``; ground truth
    never enters any of these three operations.
    """

    def region_available(self, valid: Tensor, region_ids: Tensor, grid: tuple[int, int]) -> Tensor:
        points = F.adaptive_max_pool2d(valid.float().unsqueeze(1), grid).flatten(1).bool()
        return torch.stack([(points & (region_ids == region)).any(-1)
                            for region in range(self.regions)], 1)

    def update_anchor(self, state, anchor, anchor_valid, anchor_age, selected,
                      region_ids, tokenizer, memory):
        read_mask = selected[:, region_ids]
        tokens, frame_valid = tokenizer(anchor.float(), anchor_age, anchor_valid, read_mask)
        update_valid = frame_valid & selected[:, region_ids]
        updated_memory = memory(state['memory'], tokens, update_valid)
        return {'memory': updated_memory, 'valid': state['valid'] | update_valid,
                'age_s': torch.where(update_valid, anchor_age[:, None], state['age_s'])}

    @staticmethod
    def replacement_labels(keep, candidate, target, common_mask, feasible,
                            epsilon, cost_weight, cost):
        with torch.no_grad():
            common = common_mask.long().cumprod(-1).bool()
            keep_error = TrajectoryL1Loss.trajectory_error(keep, target, common)
            new_error = TrajectoryL1Loss.trajectory_error(candidate, target, common)
            gain = keep_error - new_error
            valid = common.any(-1) & feasible
            labels = (gain > epsilon + cost_weight * cost).long()
        return labels, valid, gain

    def __init__(self, hidden_dim: int, regions: int, horizon: int,
                 heads: int = 4, dropout: float = 0., residual_scale: float = .1):
        super().__init__()
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if regions < 1 or horizon < 1:
            raise ValueError("regions and horizon must be positive")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.hidden_dim = hidden_dim
        self.regions = regions
        self.horizon = horizon
        self.residual_scale = residual_scale

        # Memory latents and tokenizer observations are different representations;
        # learn separate projections rather than treating their raw difference as
        # physical scene motion. Spatial tokens remain intact until plan reading.
        self.observation = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.old_memory = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.lag = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.evidence_type = nn.Embedding(2, hidden_dim)
        self.trajectory = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.global_query = nn.Sequential(nn.Linear(hidden_dim * 2 + 9, hidden_dim), nn.GELU())
        self.reader = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.read_ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                     nn.Linear(hidden_dim * 2, hidden_dim))
        self.context_fusion = nn.Sequential(nn.Linear(hidden_dim * 4 + 9, hidden_dim),
                                             nn.LayerNorm(hidden_dim), nn.GELU())
        self.revision = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
                                      nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        # Start close to the existing frozen planning representation while keeping
        # a gradient into the whole adapter from the first optimization step.
        nn.init.normal_(self.revision[-1].weight, std=1e-3)
        nn.init.zeros_(self.revision[-1].bias)
        self.score = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @staticmethod
    def _pool(values: Tensor, mask: Tensor) -> Tensor:
        values = torch.nan_to_num(values).masked_fill(~mask[..., None], 0)
        return values.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _encode_plan(self, plan: Tensor, mask: Tensor, dt: Tensor | float = 1.) -> Tensor:
        mask = mask.bool()
        plan = torch.nan_to_num(plan).masked_fill(~mask[..., None], 0)
        batch, steps = plan.shape[:2]
        interval = torch.as_tensor(dt, device=plan.device, dtype=plan.dtype).expand(batch)
        time = (torch.arange(steps, device=plan.device, dtype=plan.dtype) + 1)[None] * interval[:, None]
        yaw = plan[..., 2]
        features = torch.cat((plan[..., :2], yaw.sin()[..., None], yaw.cos()[..., None],
                              time[..., None], mask[..., None].to(plan.dtype)), -1)
        return self.trajectory(features.masked_fill(~mask[..., None], 0)).masked_fill(~mask[..., None], 0)

    def context(self, state, tokens: Tensor, token_valid: Tensor,
                monitor_tokens: Tensor, monitor_valid: Tensor, monitor_lags: Tensor,
                retained_plan: Tensor, plan_mask: Tensor, previous_action: Tensor,
                budget: Tensor, dt: Tensor | float = .1) -> Tensor:
        """Read aligned ``[B,K,N,D]`` monitoring tokens using the retained plan.

        Lags are nonnegative seconds relative to the current decision. Alignment
        and timestamp filtering happen in the rolling planner, before this call.
        A global query also sees all valid cells so approaching actors outside
        the immediate retained trajectory are not excluded by a spatial crop.
        """
        batch, frames, points, dim = monitor_tokens.shape
        if dim != self.hidden_dim or frames < 1 or points < 1:
            raise ValueError("monitor_tokens must contain nonempty [B,K,N,hidden_dim] observations")
        if monitor_valid.shape != monitor_tokens.shape[:-1] or monitor_lags.shape != (batch, frames):
            raise ValueError("monitor masks and lags do not match the observation history")
        if not torch.isfinite(monitor_lags).all() or (monitor_lags < 0).any():
            raise ValueError("monitor observations must have finite nonnegative causal lags")
        monitor_valid = monitor_valid.bool()
        token_valid = token_valid.bool()
        memory_valid = state['valid'].bool()
        plan_mask = plan_mask.bool()
        clean = torch.nan_to_num(monitor_tokens).masked_fill(~monitor_valid[..., None], 0)
        lag = monitor_lags.to(clean.dtype)
        time_encoding = self.lag(torch.stack((lag, torch.log1p(lag)), -1))
        observations = self.observation(clean) + time_encoding[:, :, None] + self.evidence_type.weight[0]
        observations = observations.masked_fill(~monitor_valid[..., None], 0).reshape(batch, frames * points, dim)
        old = torch.nan_to_num(state['memory']).masked_fill(~memory_valid[..., None], 0)
        memories = self.old_memory(old) + self.evidence_type.weight[1]
        memories = memories.masked_fill(~memory_valid[..., None], 0)
        source = torch.cat((observations, memories), 1)
        source_valid = torch.cat((monitor_valid.flatten(1), memory_valid), 1)
        safe = source_valid.clone()
        safe[:, 0] |= ~safe.any(-1)

        old_summary = self._pool(old, memory_valid)
        current_summary = self._pool(tokens, token_valid)
        age = torch.nan_to_num(state['age_s'], nan=0., posinf=0., neginf=0.).clamp_min(0)
        age = age.masked_fill(~memory_valid, 0).sum(1) / memory_valid.sum(1).clamp_min(1)
        interval = torch.as_tensor(dt, device=tokens.device, dtype=tokens.dtype).expand(batch)
        budgets = torch.as_tensor(budget, device=tokens.device, dtype=tokens.dtype).expand(batch)
        metadata = torch.cat((torch.nan_to_num(previous_action),
                              torch.stack((torch.log1p(age), memory_valid.float().mean(1),
                                           token_valid.float().mean(1), plan_mask.float().mean(1),
                                           budgets, interval), -1)), -1)
        plan_query = self._encode_plan(retained_plan, plan_mask, interval)
        global_query = self.global_query(torch.cat((old_summary, current_summary, metadata), -1))[:, None]
        query = torch.cat((plan_query, global_query), 1)
        attended, _ = self.reader(query, source, source, key_padding_mask=~safe, need_weights=False)
        read = self.read_norm(query + attended)
        read = read + self.read_ff(read)
        plan_summary = self._pool(read[:, :-1], plan_mask)
        return self.context_fusion(torch.cat((plan_summary, read[:, -1], old_summary, current_summary, metadata), -1))

    def adapt(self, base_state, context: Tensor):
        """Apply a bounded, per-token residual without mutating base_state."""
        valid = base_state['valid'].bool()
        base = torch.nan_to_num(base_state['memory']).masked_fill(~valid[..., None], 0)
        condition = context[:, None].expand(-1, base.shape[1], -1)
        residual = self.residual_scale * self.revision(torch.cat((base, condition), -1)).tanh()
        revised = (base + residual).masked_fill(~valid[..., None], 0)
        return {**base_state, 'memory': revised}

    def compare(self, context: Tensor, keep: Tensor, candidate: Tensor, common_mask: Tensor,
                dt: Tensor | float = .1) -> Tensor:
        """Score both plans on one shared time mask, returning [keep,new].

        The caller may detach candidate coordinates for classification, retaining
        the separate differentiable candidate for expert L1 supervision.
        Identical plans have identical scores by construction.
        """
        common = common_mask.bool().long().cumprod(-1).bool()
        scores = []
        for plan in (keep, candidate):
            encoded = self._encode_plan(plan, common, dt)
            pooled = self._pool(encoded, common)
            last = common.sum(-1).sub(1).clamp_min(0)
            endpoint = encoded.gather(1, last[:, None, None].expand(-1, 1, self.hidden_dim)).squeeze(1)
            endpoint = endpoint.masked_fill(~common.any(-1, keepdim=True), 0)
            scores.append(self.score(torch.cat((context, pooled, endpoint), -1)).squeeze(-1))
        return torch.stack(scores, -1)
