"""Causal latent revision and candidate acceptance with a frozen action decoder."""
from __future__ import annotations

import torch

from modules.action_loss import ActionLoss
from modules.binary_rolling_planner import BinaryRollingPlanner


class TrajectoryRevisionPlanner(BinaryRollingPlanner):
    """Generate before accepting; rejected candidates still incur decoding cost."""

    def __init__(self, horizon=5, probability_threshold=.5, label_epsilon=0.,
                 update_cost=1., fixed_interval=3, random_probability=.3, monitor_frames=3):
        super().__init__(horizon, probability_threshold, label_epsilon, update_cost,
                         fixed_interval, random_probability)
        if monitor_frames < 1:
            raise ValueError('monitor_frames must be positive')
        self.monitor_frames = monitor_frames

    @staticmethod
    def detach_state(state):
        return {key: value.detach() for key, value in state.items()}

    def forward(self, features, valid, times, poses, bev_range, *, start, steps,
                tokenizer, memory, decoder, refresh, temporal, region_ids,
                budget, cost_weight, targets=None, target_mask=None, refresh_policy='learned',
                diagnostics=False, random_seed=1024, adapter_enabled=True):
        if refresh_policy not in {'learned', 'no_refresh', 'replan_only', 'fixed', 'random', 'all_fresh', 'oracle'}:
            raise ValueError(f'Unknown revision policy: {refresh_policy}')
        if (targets is None) != (target_mask is None):
            raise ValueError('targets and target_mask must be supplied together')
        if refresh_policy == 'oracle' and targets is None:
            raise ValueError('oracle requires offline expert labels')
        if start < 0 or steps < 1 or start + steps >= features.shape[1]:
            raise ValueError('revision schedule exceeds cached observations')
        if times.ndim == 3:
            times = times.amax(-1)
        if poses is not None and bev_range is None:
            raise ValueError('BEV range is required for causal coordinate alignment')
        batch = features.shape[0]
        slots = torch.arange(self.horizon, device=features.device)[None]
        budgets = torch.as_tensor(budget, device=features.device, dtype=torch.float32).expand(batch)
        cost = budgets.new_full((batch,), self.update_cost)
        ages = ((times[:, start, None] - times) / 1e6).float()
        monitor = []
        with torch.no_grad():
            state = temporal.encode_history(features[:, :1], valid[:, :1], ages[:, :1], tokenizer, memory)
            for frame in range(1, start + 1):
                if poses is not None:
                    state = temporal.align_memory(state, poses[:, frame - 1], poses[:, frame], bev_range)
                tokens, mask = tokenizer(features[:, frame], ages[:, frame], valid[:, frame])
                state = dict(memory=memory(state['memory'], tokens, mask), valid=state['valid'] | mask,
                             age_s=torch.where(mask, ages[:, frame, None], state['age_s']))
            # Monitoring has its own arrived observations, independent of commits.
            for frame in range(max(0, start - self.monitor_frames + 2), start + 1):
                tokens, mask = tokenizer(features[:, frame], ages.new_zeros(batch), valid[:, frame])
                monitor.append((frame, tokens, mask))
            previous = features.new_zeros(batch, 3, dtype=torch.float32)
            actions, _ = self.decode(state, previous, decoder, self.horizon)
            plan = ActionLoss.trajectory(actions)
            initial = plan.clone()
        remaining = torch.full((batch,), self.horizon, device=features.device, dtype=torch.long)
        # All entries are detached independent candidates, not alternate world states.
        history = [(start, plan.detach(), slots.expand(batch, -1) < remaining[:, None])]
        generator = torch.Generator(device=features.device).manual_seed(random_seed)
        outputs = []
        for index in range(steps):
            frame = start + 1 + index
            with torch.no_grad():
                old_pose = None if poses is None else poses[:, frame - 1]
                pose = None if poses is None else poses[:, frame]
                if poses is not None:
                    state = temporal.align_memory(state, old_pose, pose, bev_range)
                dt = ((times[:, frame] - times[:, frame - 1]) / 1e6).float()
                state['age_s'] = state['age_s'] + dt[:, None]
                previous = actions[:, 0].clone()
                actions = torch.cat((actions[:, 1:], torch.zeros_like(actions[:, :1])), 1)
                plan = torch.cat((self.shift_plan(plan, old_pose, pose), torch.zeros_like(plan[:, :1])), 1)
                remaining = (remaining - 1).clamp_min(0)
                keep = plan.clone()
                keep_mask = slots < remaining[:, None]
                deadline = remaining == 0
                tokens, token_valid = tokenizer(features[:, frame], ages.new_zeros(batch), valid[:, frame])
                feasible = token_valid.any(-1) & (budgets >= cost)
                monitor.append((frame, tokens, token_valid))
                monitor = monitor[-self.monitor_frames:]
                aligned, monitor_masks, lags = [], [], []
                for source_frame, source_tokens, source_mask in monitor:
                    observation = dict(memory=source_tokens, valid=source_mask,
                                       age_s=ages.new_zeros(source_mask.shape))
                    if poses is not None and source_frame != frame:
                        observation = temporal.align_memory(observation, poses[:, source_frame], pose, bev_range)
                    aligned.append(observation['memory'])
                    monitor_masks.append(observation['valid'])
                    lags.append(((times[:, frame] - times[:, source_frame]) / 1e6).float())
                monitor_tokens = torch.stack(aligned, 1)
                monitor_valid = torch.stack(monitor_masks, 1)
                monitor_lags = torch.stack(lags, 1)
                # Shift every old candidate to the arrived frame and discard elapsed points.
                shifted_history = []
                for source_frame, old_plan, old_mask in history:
                    shifted = torch.cat((self.shift_plan(old_plan, old_pose, pose),
                                         torch.zeros_like(old_plan[:, :1])), 1)
                    shifted_mask = torch.cat((old_mask[:, 1:], torch.zeros_like(old_mask[:, :1])), 1)
                    if frame - source_frame < self.horizon:
                        shifted_history.append((source_frame, shifted, shifted_mask))
                history = shifted_history[-max(0, self.horizon - 1):] if self.horizon > 1 else []
                # Frozen evidence update. No speculative candidate is committed here.
                base = {key: value.clone() for key, value in state.items()}
                if feasible.any():
                    base['memory'][feasible] = memory(state['memory'][feasible], tokens[feasible], token_valid[feasible])
                    base['valid'][feasible] |= token_valid[feasible]
                    base['age_s'][feasible] = torch.where(token_valid[feasible],
                        torch.zeros_like(state['age_s'][feasible]), state['age_s'][feasible])

            context = refresh.context(state, tokens, token_valid, monitor_tokens, monitor_valid,
                                      monitor_lags, keep, keep_mask, previous, budgets, dt)
            revised = refresh.adapt(base, context) if adapter_enabled else base
            # Do not use no_grad here: decoder weights are frozen, its input Jacobian is required.
            candidate_actions = features.new_zeros(batch, self.horizon, 3, dtype=torch.float32)
            if feasible.any():
                decoded, _ = self.decode(self.subset(revised, feasible), previous[feasible], decoder, self.horizon)
                candidate_actions[feasible] = decoded
            candidate = ActionLoss.trajectory(candidate_actions)
            # Classification cannot optimize its labels by moving the candidate geometry.
            logits = refresh.compare(context, keep, candidate.detach(), keep_mask, dt=dt)
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
            output = dict(candidate_trajectory=candidate, candidate_valid=feasible, refresh_logits=logits)
            if targets is not None:
                labels, decision_valid, gain = refresh.replacement_labels(
                    keep, candidate.detach(), targets[:, index], keep_mask & target_mask[:, index].bool(),
                    feasible, self.label_epsilon, cost_weight, cost)
                output.update(refresh_labels=labels, decision_valid=decision_valid, replacement_gain=gain,
                              common_mask=(keep_mask & target_mask[:, index].bool()).long().cumprod(-1).bool())
                if refresh_policy == 'oracle':
                    request = labels.bool() & decision_valid
                history_logits = [logits * 0 for _ in range(self.horizon - 1)]
                history_labels = [torch.zeros_like(remaining) for _ in range(self.horizon - 1)]
                history_valid = [torch.zeros_like(feasible) for _ in range(self.horizon - 1)]
                history_gain = [budgets * 0 for _ in range(self.horizon - 1)]
                history_masks = [torch.zeros_like(keep_mask) for _ in range(self.horizon - 1)]
                for source_frame, old_plan, old_mask in history:
                    lag_index = frame - source_frame - 1
                    common = old_mask & target_mask[:, index].bool()
                    old_labels, old_valid, old_gain = refresh.replacement_labels(
                        old_plan, candidate.detach(), targets[:, index], common, feasible,
                        self.label_epsilon, cost_weight, cost)
                    # Input mask is based only on elapsed plan time, never on future GT availability.
                    history_logits[lag_index] = refresh.compare(context, old_plan, candidate.detach(), old_mask, dt=dt)
                    history_labels[lag_index] = old_labels
                    history_valid[lag_index] = old_valid
                    history_gain[lag_index] = old_gain
                    history_masks[lag_index] = common.long().cumprod(-1).bool()
                if self.horizon > 1:
                    output.update(history_logits=torch.stack(history_logits, 1),
                                  history_labels=torch.stack(history_labels, 1),
                                  history_valid=torch.stack(history_valid, 1),
                                  history_gain=torch.stack(history_gain, 1),
                                  history_common_mask=torch.stack(history_masks, 1))
                else:
                    output.update(history_logits=logits[:, None, :][:, :0],
                                  history_labels=remaining[:, None][:, :0],
                                  history_valid=feasible[:, None][:, :0], history_gain=budgets[:, None][:, :0],
                                  history_common_mask=keep_mask[:, None][:, :0])
            with torch.no_grad():
                diagnostic_calls = torch.zeros_like(remaining)
                if diagnostics:
                    old_actions, _ = self.decode(state, previous, decoder, self.horizon)
                    output['proposal_trajectory'] = ActionLoss.trajectory(old_actions)
                    diagnostic_calls += self.horizon
                    if targets is not None:
                        prefix = target_mask[:, index].long().cumprod(-1).bool()
                        # Includes latent adaptation; this is not isolated visual acquisition gain.
                        output['visual_gain'] = (ActionLoss.trajectory_error(output['proposal_trajectory'], targets[:, index], prefix)
                                                 - ActionLoss.trajectory_error(candidate, targets[:, index], prefix))
                event = request & feasible & ~deadline
                accept = event | (deadline & feasible)
                if refresh_policy == 'replan_only':
                    accept = torch.zeros_like(accept)
                replan = event | deadline
                if refresh_policy == 'replan_only':
                    replan = torch.ones_like(replan)
                fallback = replan & ~accept
                adopted = candidate_actions.detach().clone()
                if fallback.any():
                    adopted[fallback], _ = self.decode(self.subset(state, fallback), previous[fallback], decoder, self.horizon)
                chosen_state = {key: torch.where(accept[:, None, None] if value.ndim == 3 else accept[:, None],
                                                 revised[key].detach(), value) for key, value in state.items()}
                plan = torch.where(replan[:, None, None], ActionLoss.trajectory(adopted), keep)
                actions = torch.where(replan[:, None, None], adopted, actions)
                remaining = torch.where(replan, torch.full_like(remaining, self.horizon), remaining)
                state = self.detach_state(chosen_state)
                # Rejected candidates only enter the auxiliary training bank, not active state.
                history.append((frame, candidate.detach(), slots < (feasible.long() * self.horizon)[:, None]))
                history = history[-self.horizon:]
                regions = refresh.region_available(valid[:, frame], region_ids, tokenizer.grid)
                output.update(planned_trajectory=plan.detach(), plan_mask=slots < remaining[:, None],
                              retained_trajectory=keep, retained_mask=keep_mask,
                              previous_action=previous, remaining_steps=keep_mask.sum(-1),
                              refresh_probability=probability.detach(), refresh_request=request & ~deadline,
                              refresh_feasible=feasible, event_refresh=event, evidence_acquired=accept,
                              refresh_mask=regions & accept[:, None], replan_mask=replan,
                              forced_replan=deadline, deadline_replan=deadline,
                              monitoring_reads=torch.ones_like(deadline), candidate_evidence_reads=feasible,
                              candidate_generated=feasible, candidate_accepted=accept,
                              candidate_rejected=feasible & ~accept,
                              online_decode_calls=(feasible.long() + fallback.long()) * self.horizon,
                              diagnostic_decode_calls=diagnostic_calls)
            outputs.append(output)
        result = {key: torch.stack([out[key] for out in outputs], 1) for key in outputs[0]}
        result.update(initial_trajectory=initial,
                      initial_decode_calls=torch.full_like(remaining, self.horizon))
        return result
