"""Causal keep/replace planner; counterfactual labels are offline work only."""
from __future__ import annotations

import torch
from torch import nn
from modules.action_loss import ActionLoss
from modules.rolling_planner import RollingPlanner


class BinaryRollingPlanner(RollingPlanner):
    def __init__(self, horizon=5, probability_threshold=.5, label_epsilon=0.,
                 update_cost=1., fixed_interval=3, random_probability=.3):
        super().__init__(horizon)
        self.probability_threshold = probability_threshold
        self.label_epsilon = label_epsilon
        self.update_cost = update_cost
        self.fixed_interval = fixed_interval
        self.random_probability = random_probability

    @staticmethod
    def subset(state, mask):
        return {key: value[mask] for key, value in state.items()}

    def forward(self, features, valid, times, poses, bev_range, *, start, steps,
                tokenizer, memory, decoder, refresh, temporal, region_ids,
                budget, cost_weight, targets=None, target_mask=None, refresh_policy='learned',
                diagnostics=False, random_seed=1024):
        if refresh_policy not in {'learned', 'no_refresh', 'replan_only', 'fixed', 'random', 'all_fresh', 'oracle'}:
            raise ValueError(f'Unknown binary refresh policy: {refresh_policy}')
        if refresh_policy == 'oracle' and targets is None:
            raise ValueError('oracle requires offline expert labels')
        if (targets is None) != (target_mask is None):
            raise ValueError('targets and target_mask must be supplied together')
        if start < 0 or start + steps >= features.shape[1]:
            raise ValueError('refresh schedule exceeds cached observations')
        batch = features.shape[0]
        ages = ((times[:, start, None] - times) / 1e6).float()
        with torch.no_grad():
            state = temporal.encode_history(features[:, :1], valid[:, :1], ages[:, :1], tokenizer, memory)
            for frame in range(1, start + 1):
                if poses is not None:
                    state = temporal.align_memory(state, poses[:, frame - 1], poses[:, frame], bev_range)
                tokens, mask = tokenizer(features[:, frame], ages[:, frame], valid[:, frame])
                state = dict(memory=memory(state['memory'], tokens, mask), valid=state['valid'] | mask,
                             age_s=torch.where(mask, ages[:, frame, None], state['age_s']))
            previous = features.new_zeros(batch, 3, dtype=torch.float32)
            actions, _ = self.decode(state, previous, decoder, self.horizon)
            plan = ActionLoss.trajectory(actions)
            initial = plan.clone()
        remaining = torch.full((batch,), self.horizon, device=features.device, dtype=torch.long)
        budgets = torch.as_tensor(budget, device=features.device, dtype=torch.float32).expand(batch)
        cost = budgets.new_full((batch,), self.update_cost)
        slots = torch.arange(self.horizon, device=features.device)[None]
        generator = torch.Generator(device=features.device).manual_seed(random_seed)
        outputs = []
        for index in range(steps):
            frame = start + 1 + index
            with torch.no_grad():
                old_pose = None if poses is None else poses[:, frame - 1]
                pose = None if poses is None else poses[:, frame]
                if poses is not None:
                    state = temporal.align_memory(state, old_pose, pose, bev_range)
                state['age_s'] = state['age_s'] + ((times[:, frame] - times[:, frame - 1]) / 1e6).float()[:, None]
                # Consume stored local actions, never re-aligned cumulative endpoints.
                previous = actions[:, 0].clone()
                actions = torch.cat((actions[:, 1:], torch.zeros_like(actions[:, :1])), 1)
                plan = torch.cat((self.shift_plan(plan, old_pose, pose), torch.zeros_like(plan[:, :1])), 1)
                remaining = (remaining - 1).clamp_min(0)
                keep = plan.clone()
                keep_mask = slots < remaining[:, None]
                deadline = remaining == 0
                tokens, token_valid = tokenizer(features[:, frame], features.new_zeros(batch), valid[:, frame])
                feasible = token_valid.any(-1) & (budgets >= cost)
            logits = refresh(state, tokens, token_valid, keep, keep_mask, previous, budgets)
            probability = logits.softmax(-1)[:, 1]
            request = probability > self.probability_threshold
            if refresh_policy in {'no_refresh', 'replan_only'}:
                request = torch.zeros_like(request)
            elif refresh_policy == 'all_fresh':
                request = torch.ones_like(request)
            elif refresh_policy == 'fixed':
                request = torch.full_like(request, (index + 1) % self.fixed_interval == 0)
            elif refresh_policy == 'random':
                request = torch.rand(batch, device=features.device, generator=generator) < self.random_probability
            with torch.no_grad():
                updated = state
                needs_update = feasible & (request | deadline)
                if targets is not None or diagnostics or refresh_policy == 'oracle':
                    needs_update = torch.ones_like(feasible)
                if needs_update.any():
                    updated = {key: value.clone() for key, value in state.items()}
                    updated['memory'][needs_update] = memory(state['memory'][needs_update], tokens[needs_update], token_valid[needs_update])
                    updated['valid'][needs_update] |= token_valid[needs_update]
                    updated['age_s'][needs_update] = torch.where(token_valid[needs_update],
                        torch.zeros_like(state['age_s'][needs_update]), state['age_s'][needs_update])
                label_candidate = label_actions = None
                output = {}
                if targets is not None or diagnostics:
                    # Explicit offline branch: never required by a keep decision online.
                    label_actions, _ = self.decode(updated, previous, decoder, self.horizon)
                    label_candidate = ActionLoss.trajectory(label_actions)
                    output['candidate_trajectory'] = label_candidate
                    if targets is not None:
                        common = keep_mask & target_mask[:, index].bool()
                        labels, decision_valid, gain = refresh.replacement_labels(
                            keep, label_candidate, targets[:, index], common, feasible,
                            self.label_epsilon, cost_weight, cost)
                        output.update(refresh_labels=labels, decision_valid=decision_valid,
                                      replacement_gain=gain, common_mask=common.long().cumprod(-1).bool())
                        if refresh_policy == 'oracle':
                            request = labels.bool() & decision_valid
                    if diagnostics:
                        old_actions, _ = self.decode(state, previous, decoder, self.horizon)
                        output['proposal_trajectory'] = ActionLoss.trajectory(old_actions)
                        if targets is not None:
                            prefix = target_mask[:, index].long().cumprod(-1).bool()
                            output['visual_gain'] = (ActionLoss.trajectory_error(output['proposal_trajectory'], targets[:, index], prefix)
                                                     - ActionLoss.trajectory_error(label_candidate, targets[:, index], prefix))
                event = request & feasible & ~deadline
                # Deadline has no keep counterfactual: update using the same feasible
                # full-BEV rule, except the explicitly stale-memory controls.
                acquire = event | (deadline & feasible)
                if refresh_policy == 'replan_only':
                    acquire = torch.zeros_like(acquire)
                replan = event | deadline
                if refresh_policy == 'replan_only':
                    replan = torch.ones_like(replan)
                # Only the adopted subset decodes in the online (no-label) path.
                chosen_state = {key: torch.where(acquire[:, None, None] if value.ndim == 3 else acquire[:, None],
                                                 updated[key], value) for key, value in state.items()}
                adopted = torch.zeros_like(actions)
                reuse = replan & acquire if label_actions is not None else torch.zeros_like(replan)
                if reuse.any():
                    adopted[reuse] = label_actions[reuse]
                decode_needed = replan & ~reuse
                if decode_needed.any():
                    adopted[decode_needed], _ = self.decode(self.subset(chosen_state, decode_needed), previous[decode_needed], decoder, self.horizon)
                plan = torch.where(replan[:, None, None], ActionLoss.trajectory(adopted), plan)
                actions = torch.where(replan[:, None, None], adopted, actions)
                remaining = torch.where(replan, torch.full_like(remaining, self.horizon), remaining)
                state = chosen_state
                regions = refresh.region_available(valid[:, frame], region_ids, tokenizer.grid)
                output.update(planned_trajectory=plan, plan_mask=slots < remaining[:, None],
                              retained_trajectory=keep, retained_mask=keep_mask,
                              previous_action=previous, remaining_steps=keep_mask.sum(-1),
                              refresh_probability=probability.detach(), refresh_request=request & ~deadline,
                              refresh_feasible=feasible, event_refresh=event, evidence_acquired=acquire,
                              refresh_mask=regions & acquire[:, None], replan_mask=replan,
                              forced_replan=deadline, deadline_replan=deadline,
                              monitoring_reads=torch.ones_like(deadline), online_decode_calls=replan.long() * self.horizon)
            output['refresh_logits'] = logits
            outputs.append(output)
        result = {key: torch.stack([out[key] for out in outputs], 1) for key in outputs[0]}
        result['initial_trajectory'] = initial
        return result
