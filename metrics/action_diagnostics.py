"""Paired checkpoint diagnostics; interventions are evaluation-only, not retraining."""
import csv
from collections import defaultdict

import torch

from datasets.dataset_utils import local_actions
from metrics.action import sample_metrics
from metrics.evaluator import Evaluator
from metrics.statistics import aggregate, paired_comparison
from modules import RollingPlanner


def history_action(batch, history):
    poses = torch.cat((batch['trajectory']['past_trajectory'].float(),
                       batch['trajectory']['anchor_trajectory'].float()[:, None]), 1)
    action = local_actions(poses)[:, -1]
    valid = batch['trajectory']['trajectory_mask'][:, history-1:history+1].all(-1)
    return action, valid


def replay(model, inputs, *, lag=0, vision='fresh', teacher_actions=None,
           initial_action=None, fixed_horizon=None, dropout_steps=0,
           history_skip=0, zero_age=False):
    """Delay actual observations, align their tokens, never use future images.

    Delayed history has no observations before the stored window. Missing-current
    retains causal history. No-vision removes all feature evidence (an inference
    ablation, not a trained ego-state baseline). Fixed rollout resets decoder
    hidden/step exactly as RollingPlanner.decode does.
    """
    features = inputs['bev_features']
    masks = inputs['bev_valid_mask']
    times = inputs['timestamps_us'].double()
    poses = inputs.get('reference_ego_to_global')
    batch = features.shape[0]
    h, future = model.history_features, model.future_action
    memory = features.new_zeros(batch, model.grid[0]*model.grid[1], model.hidden_dim, dtype=torch.float32)
    state = dict(memory=memory, valid=torch.zeros(memory.shape[:2], dtype=torch.bool, device=features.device),
                 age_s=torch.full(memory.shape[:2], 1e6, device=features.device))
    previous = features.new_zeros(batch, 3, dtype=torch.float32) if initial_action is None else initial_action
    hidden, outputs = None, []
    for frame in range(h + (1 if fixed_horizon else future)):
        if frame and poses is not None:
            state = model.temporal.align_memory(state, poses[:, frame-1], poses[:, frame], inputs['bev_range'])
        if frame:
            state['age_s'] = state['age_s'] + ((times[:, frame]-times[:, frame-1])/1e6).float()[:, None]
        source = frame-lag
        read = source >= 0 and vision != 'none'
        read &= frame >= history_skip
        read &= not (vision == 'history_only' and frame >= h)
        read &= not (h <= frame < h+dropout_steps)
        if read:
            # Historical ages match the normal checkpoint forward; delayed arrivals
            # carry their actual nominal delay rather than being relabeled fresh.
            age = ((times[:, max(h, frame)]-times[:, source])/1e6).float()
            if zero_age:
                age = torch.zeros_like(age)
            tokens, valid = model.tokenizer(features[:, source], age, masks[:, source])
            observed = dict(memory=tokens, valid=valid, age_s=age[:, None].expand_as(valid))
            if poses is not None and source != frame:
                observed = model.temporal.align_memory(observed, poses[:, source], poses[:, frame], inputs['bev_range'])
            state = dict(memory=model.memory(state['memory'], observed['memory'], observed['valid']),
                         valid=state['valid'] | observed['valid'],
                         age_s=torch.where(observed['valid'], observed['age_s'], state['age_s']))
        if frame < h:
            continue
        if fixed_horizon:
            actions, _ = RollingPlanner.decode(state, previous, model.decoder, fixed_horizon)
            return actions
        action, hidden = model.decoder(state['memory'], state['valid'], previous, hidden, frame-h)
        outputs.append(action)
        previous = action if teacher_actions is None else teacher_actions[:, frame-h]
    return torch.stack(outputs, 1)


class ActionDiagnosticsEvaluator(Evaluator):
    """Export paired replay, stale/recovery, and fixed-five-step experiments."""

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.model.mode != 'action':
            raise ValueError('Diagnostics require an action checkpoint')
        if self.model.future_action < 5 or self.model.history_features < max(self.options['stale_steps']):
            raise ValueError('Diagnostics require five future steps and sufficient delay history')
        self.rows = []
        self.completed_batches = 0

    def validation_step(self, batch, batch_idx):
        inputs = {k: batch[k] for k in ('bev_features', 'bev_valid_mask', 'timestamps_us',
                  'reference_ego_to_global', 'bev_range') if k in batch}
        model = self.model
        h, f = model.history_features, model.future_action
        if not (inputs['timestamps_us'].diff(dim=1)==100_000).all():
            raise ValueError('Diagnostic delay steps require a uniformly sampled 10 Hz cache')
        target, valid = batch['future_trajectory'].float(), batch['future_action_mask'].bool()
        historical, historical_valid = history_action(batch, h)
        teacher = local_actions(target)
        predictions = {'fresh': replay(model, inputs),
                       'teacher_forced_diagnostic': replay(model, inputs, teacher_actions=teacher),
                       'history_motion_initialization': replay(model, inputs, initial_action=historical),
                       'no_vision': replay(model, inputs, vision='none'),
                       'history_only': replay(model, inputs, vision='history_only'),
                       'drop_3_then_recover': replay(model, inputs, dropout_steps=3)}
        for lag in self.options['stale_steps']:
            if lag:
                predictions[f'delayed_{lag*100}ms'] = replay(model, inputs, lag=lag)
        largest_lag = max(self.options['stale_steps'])
        predictions['fresh_short_history_control'] = replay(model, inputs, history_skip=largest_lag)
        predictions['delayed_zero_age_diagnostic'] = replay(model, inputs, lag=largest_lag, zero_age=True)
        # Verify numerical parity against the production forward on every batch.
        production = model(**inputs, teacher_forcing=False)
        torch.testing.assert_close(predictions['fresh'], production, atol=2e-5, rtol=2e-4)
        predictions['constant_action'] = historical[:, None].expand(-1, f, -1)
        # Fixed-context and arrived-observation forecasts share anchor, horizon and
        # labels; decoder resets follow the deployed refresh candidate path.
        for lag in (0, *[x for x in self.options['stale_steps'] if x]):
            name = 'fixed5_fresh' if lag == 0 else f'fixed5_delayed_{lag*100}ms'
            predictions[name] = replay(model, inputs, lag=lag, fixed_horizon=5)
        predictions['fixed5_no_vision'] = replay(model, inputs, vision='none', fixed_horizon=5)
        predictions['fixed5_arrived_observations'] = predictions['fresh'][:, :5]
        predictions['fixed5_constant_action'] = historical[:, None].expand(-1, 5, -1)
        offset = batch_idx * self.trainer.val_dataloaders.batch_size
        for scenario, actions in predictions.items():
            horizon = actions.shape[1]
            mask = valid[:, :horizon]
            if scenario in ('constant_action', 'fixed5_constant_action', 'history_motion_initialization'):
                mask = mask & historical_valid[:, None]
            scores = sample_metrics(actions, target[:, :horizon], mask, model.action_loss, .1)
            truth = teacher[:, :horizon]
            # Signed local errors separate speed bias from lateral/heading drift.
            for dim, name in enumerate(('dx_bias_m', 'dy_bias_m', 'dyaw_bias_rad')):
                error = actions[..., dim]-truth[..., dim]
                if dim == 2:
                    error = torch.atan2(error.sin(), error.cos())
                scores[name] = error.masked_fill(~mask, 0).sum(-1)/mask.sum(-1).clamp_min(1)
            scores['predicted_step_distance_m'] = actions[..., :2].norm(dim=-1).mean(-1)
            scores['expert_step_distance_m'] = truth[..., :2].norm(dim=-1).mean(-1)
            if not scenario.startswith('fixed5_'):
                xy_error = (actions[..., :2]-truth[..., :2]).norm(dim=-1)
                for start, end, label in ((0,3,'first3'),(3,f,'after3')):
                    m = mask[:, start:end]
                    scores[f'{label}_local_error_m'] = xy_error[:, start:end].masked_fill(~m,0).sum(-1)/m.sum(-1).clamp_min(1)
            scores = {k:v.detach().cpu().tolist() for k,v in scores.items()}
            for i in range(len(target)):
                row = dict(scenario=scenario, sample=str(self.dataset.samples[offset+i].relative_to(self.dataset.event_dir)),
                           recording=batch['recording'][i], event_type=batch['event_type'][i],
                           anchor_timestamp_us=int(batch['anchor_timestamp_us'][i]), horizon_s=horizon*.1,
                           historical_speed_mps=float(historical[i,:2].norm()/.1),
                           historical_yaw_rate_radps=float(historical[i,2]/.1))
                row.update({k:v[i] if torch.isfinite(torch.tensor(v[i])) else None for k,v in scores.items()})
                self.rows.append(row)
        self.completed_batches += 1
        print(f'Diagnostics: {self.completed_batches}/{len(self.trainer.val_dataloaders)} batches', flush=True)

    def on_validation_epoch_end(self):
        groups = defaultdict(dict)
        for row in self.rows:
            groups[row['scenario']].setdefault((row['recording'],row['anchor_timestamp_us']), row)
        results = {}
        for scenario, unique in groups.items():
            rows = list(unique.values())
            keys = [k for k,v in rows[0].items() if k not in ('scenario','sample','recording','event_type',
                    'anchor_timestamp_us','horizon_s','historical_speed_mps','historical_yaw_rate_radps')]
            results[scenario] = aggregate(rows, keys)
            results[scenario]['by_event_type'] = {kind:aggregate([r for r in rows if r['event_type']==kind],keys)
                                                     for kind in sorted({r['event_type'] for r in rows})}
            results[scenario]['by_speed'] = {name: aggregate([r for r in rows if low <= r['historical_speed_mps'] < high], keys)
                for name,low,high in [('stationary',0,.5),('slow',.5,2),('medium',2,5),('fast',5,float('inf'))]}
            results[scenario]['by_turn'] = {name: aggregate([r for r in rows if (abs(r['historical_yaw_rate_radps']) >= .1)==turn], keys)
                for name,turn in [('near_straight',False),('turning',True)]}
            reference = 'fixed5_fresh' if scenario.startswith('fixed5_') else 'fresh'
            paired = []
            for identity, row in unique.items():
                ref = groups[reference][identity]
                paired.append(dict(recording=row['recording'],
                    model_ade=row['ade_full_horizon_m'],constant_action_ade=ref['ade_full_horizon_m'],
                    model_fde=row['fde_m'],constant_action_fde=ref['fde_m']))
            results[scenario]['reference_minus_scenario'] = paired_comparison(paired, ('ade','fde'), bootstrap_samples=0)
        report = dict(scenarios=results, samples=len(groups['fresh']), manifest_rows=len(self.rows)//len(groups),
                      checkpoint=self.config['trainer']['ckpt_path'], seed=self.seed, config=self.config,
                      protocol=dict(task='paired_checkpoint_diagnostics',closed_loop=False,
                          selection='seeded event-stratified unique anchors; all scenarios share samples; first sampled event label used for multi-label anchors',
                          no_vision='all BEV evidence removed; inference ablation, not a trained nonvisual model',
                          delay='real earlier cached BEVs; age refers to BEV query timestamps, not exact camera exposure age; tokens aligned using causal localization',
                          delay_boundary='pre-window delayed observations unavailable, not padded with future evidence',
                          fixed5='anchor only; fixed visual memory; reset hidden, zero previous action and step=0 as refresh candidate',
                          teacher_forced='future expert prefix is diagnostic only, not deployable performance',
                          motion_initialized='extra historical ego-motion input; inference diagnostic, not a retrained architecture',
                          ci='not computed; no authoritative log map supplied',
                          adaptation='robustness evaluation only; checkpoint unchanged, no stale-input finetuning'))
        report['protocol']['delay_controls'] = {
            'fresh_short_history_control': 'fresh observations but omit the first max(stale_steps) history frames',
            'delayed_zero_age_diagnostic': 'largest delay with token age input set to zero; knowingly incorrect metadata, diagnostic only'}
        report['device'] = str(self.device)
        report['torch_version'] = str(torch.__version__)
        output = self.export(report)
        keys = list(dict.fromkeys(k for row in self.rows for k in row))
        with (output/'per_sample.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(self.rows)
