"""Event-centred rolling plans with causal visual acquisition."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from modules.action_loss import ActionLoss


class RollingPlanner(nn.Module):
    def __init__(self, horizon: int = 5):
        super().__init__()
        self.horizon = horizon

    @staticmethod
    def decode(state, previous, decoder, horizon):
        """Keep visual memory fixed throughout a candidate rollout."""
        hidden, actions, queries = None, [], []
        for step in range(horizon):
            previous, hidden = decoder(state["memory"], state["valid"], previous, hidden, step)
            actions.append(previous)
            queries.append(hidden)
        return torch.stack(actions, 1), torch.stack(queries, 1)

    @staticmethod
    def shift_plan(plan, source_pose, destination_pose):
        """Discard the executed endpoint and express remaining endpoints at now."""
        remaining = plan[:, 1:]
        if source_pose is None:
            # Synthetic tests without localization use the planned first motion.
            origin, angle = plan[:, :1, :2], plan[:, :1, 2]
            xy = remaining[..., :2] - origin
            yaw = remaining[..., 2] - angle
            x = angle.cos() * xy[..., 0] + angle.sin() * xy[..., 1]
            y = -angle.sin() * xy[..., 0] + angle.cos() * xy[..., 1]
        else:
            transform = torch.linalg.solve(destination_pose.double(), source_pose.double()).float()
            points = torch.cat((remaining[..., :2], torch.zeros_like(remaining[..., :1]),
                                torch.ones_like(remaining[..., :1])), -1)
            xy = (points @ transform.transpose(-1, -2))[..., :2]
            x, y = xy.unbind(-1)
            yaw = remaining[..., 2] + torch.atan2(transform[:, 1, 0], transform[:, 0, 0])[:, None]
        return torch.stack((x, y, torch.atan2(yaw.sin(), yaw.cos())), -1)

    def forward(self, features, valid, times, poses, bev_range, *, start, steps,
                tokenizer, memory, decoder, refresh, temporal, region_ids,
                budget, cost_weight, targets=None, target_mask=None):
        batch = features.shape[0]
        grid = tokenizer.grid
        # All historical ages are measured at the initial decision time.
        ages = ((times[:, start, None] - times) / 1e6).float()
        state = temporal.encode_history(features[:, :1], valid[:, :1], ages[:, :1], tokenizer, memory)
        for frame in range(1, start + 1):
            if poses is not None:
                state = temporal.align_memory(state, poses[:, frame - 1], poses[:, frame], bev_range)
            tokens, mask = tokenizer(features[:, frame], ages[:, frame], valid[:, frame])
            state = {"memory": memory(state["memory"], tokens, mask),
                     "valid": state["valid"] | mask,
                     "age_s": torch.where(mask, ages[:, frame, None], state["age_s"])}
        previous = features.new_zeros(batch, 3, dtype=torch.float32)
        initial_actions, _ = self.decode(state, previous, decoder, self.horizon)
        plan = ActionLoss.trajectory(initial_actions)
        remaining = torch.full((batch,), self.horizon, device=features.device, dtype=torch.long)
        costs = features.new_ones(batch, refresh.regions, dtype=torch.float32)
        budgets = features.new_full((batch,), budget, dtype=torch.float32)
        outputs = []
        for index in range(steps):
            frame = start + 1 + index
            old_pose = None if poses is None else poses[:, frame - 1]
            pose = None if poses is None else poses[:, frame]
            if poses is not None:
                state = temporal.align_memory(state, old_pose, pose, bev_range)
            elapsed = ((times[:, frame] - times[:, frame - 1]) / 1e6).float()
            state["age_s"] = state["age_s"] + elapsed[:, None]
            previous = plan[:, 0].clone()
            shifted = self.shift_plan(plan, old_pose, pose)
            plan = torch.cat((shifted, torch.zeros_like(plan[:, :1])), 1)
            remaining = (remaining - 1).clamp_min(0)
            proposal, queries = self.decode(state, previous, decoder, self.horizon)
            decision = refresh(state["memory"], state["valid"], state["age_s"], queries,
                               costs, budgets, region_ids)
            available = refresh.region_available(valid[:, frame], region_ids, grid)
            selected = refresh.select(decision["refresh_utility"], available, costs, budgets, cost_weight)
            forced = remaining == 0
            # Deadline resets planning, but never overrides the visual budget.
            updated = refresh.update_anchor(state, features[:, frame], valid[:, frame],
                                            features.new_zeros(batch, dtype=torch.float32), selected, region_ids, tokenizer, memory)
            candidate, _ = self.decode(updated, previous, decoder, self.horizon)
            replan = selected.any(-1) | forced
            plan = torch.where(replan[:, None, None], ActionLoss.trajectory(candidate), plan)
            remaining = torch.where(replan, torch.full_like(remaining, self.horizon), remaining)
            plan_mask = torch.arange(self.horizon, device=features.device)[None] < remaining[:, None]
            output = {"planned_trajectory": plan, "plan_mask": plan_mask,
                      "proposal_trajectory": ActionLoss.trajectory(proposal),
                      "refresh_utility": decision["refresh_utility"], "refresh_mask": selected,
                      "replan_mask": replan, "forced_replan": forced}
            if targets is not None:
                output["fresh_targets"] = refresh.fresh_targets(
                    state, features[:, frame], valid[:, frame], targets[:, index], target_mask[:, index],
                    previous, proposal, self.horizon, tokenizer, memory, decoder, region_ids)
            state = updated
            outputs.append(output)
        result = {key: torch.stack([item[key] for item in outputs], 1) for key in outputs[0]}
        result["initial_trajectory"] = ActionLoss.trajectory(initial_actions)
        return result
