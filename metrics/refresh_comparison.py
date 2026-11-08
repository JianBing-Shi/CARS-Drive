"""Paired planning interventions using each mode's production rolling planner."""
import csv
import math
from pathlib import Path

import torch

from metrics.evaluator import Evaluator
from metrics.statistics import aggregate


REVISION_POLICIES = ('learned', 'no_refresh', 'replan_only', 'fixed', 'random',
                     'random_2048', 'random_4096', 'all_fresh', 'oracle',
                     'all_fresh_base', 'no_refresh_base')


def rollout(model, batch, policy):
    extra = dict(targets=batch.get('planning_targets'), target_mask=batch.get('planning_mask'), diagnostics=True)
    if policy.endswith('_base'):
        extra['adapter_enabled'] = False
        policy = policy[:-5]
    if policy.startswith('random_'):
        extra['random_seed'] = int(policy.split('_')[1])
        policy = 'random'
    features = batch['bev_features']
    if features.ndim != 5 or features.shape[1] != 41:
        raise ValueError('Planning evaluation requires the expanded 41-frame cache')
    if 'anchor_index' in batch and not (batch['anchor_index'] == model.cache_anchor_step).all():
        raise ValueError('Cache anchor_index disagrees with checkpoint cache_anchor_step')
    if batch.get('action_history_sequence') is None:
        raise ValueError('Planning evaluation requires causal action_history_sequence')
    valid = batch.get('bev_valid_mask', torch.ones_like(features[:, :, 0], dtype=torch.bool)).bool()
    times = batch.get('timestamps_us')
    if times is None:
        times = torch.arange(features.shape[1], device=features.device)[None].expand(len(features), -1) * 100000
    return model.cars_planner(
        features, valid, times.double(), batch.get('reference_ego_to_global'), batch.get('bev_range'),
        start=model.cache_anchor_step + model.cars_adapter_offset, steps=model.cars_adapter_steps,
        tokenizer=model.tokenizer, memory=model.memory, decoder=model.decoder,
        refresh=model.cars_adapter, temporal=model.temporal, region_ids=model.region_ids,
        budget=model.cars_adapter_budget, cost_weight=model.cost_weight,
        action_history_sequence=batch.get('action_history_sequence'),
        action_encoder=model.action_encoder, history_step=model.history_step,
        refresh_policy=policy, **extra)


def window_errors(prediction, target, mask):
    """One mean per window; FDE uses the last common valid endpoint per decision."""
    difference = prediction[..., :2].masked_fill(~mask[..., None], 0) - target[..., :2].masked_fill(~mask[..., None], 0)
    error = difference.norm(dim=-1)
    count = mask.sum((1, 2))
    ade = error.sum((1, 2)) / count.clamp_min(1)
    last = mask.long().sum(-1) - 1
    valid = last >= 0
    fde = error.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
    fde = fde.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
    return {'ade_m': ade.masked_fill(count == 0, float('nan')),
            'fde_m': fde.masked_fill(~valid.any(1), float('nan'))}


class RefreshComparisonEvaluator(Evaluator):
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        model = self.model
        if model.mode not in {'plan1', 'plan2'}:
            raise ValueError('Comparison requires a plan1 or plan2 checkpoint')
        if (model.plan_horizon, model.cars_adapter_offset, model.cars_adapter_steps) != (10, -5, 10):
            raise ValueError('Cached planning labels require horizon=10, offset=-5, steps=10')
        self.policies = REVISION_POLICIES
        self.rows = []

    def validation_step(self, batch, batch_idx):
        if not (batch['timestamps_us'].diff(dim=1) == 100_000).all():
            raise ValueError('Expected uniformly sampled 10 Hz windows')
        outputs = {policy: rollout(self.model, batch, policy) for policy in self.policies}
        target = batch['planning_targets'].float()
        valid = batch['planning_mask'].bool().long().cumprod(-1).bool()
        common = valid.clone()
        for output in outputs.values():
            common &= output['plan_mask']
        offset = len(self.rows) // len(self.policies)
        for policy, output in outputs.items():
            scores = {}
            for horizon in (5, 10):
                complete = valid[..., :horizon].all(-1) & output['candidate_valid']
                candidate_mask = complete[..., None].expand(*complete.shape, horizon)
                scores.update({f'fixed{horizon}_{key}': value for key, value in window_errors(
                    output['candidate_trajectory'][..., :horizon, :], target[..., :horizon, :], candidate_mask).items()})
                scores[f'fixed{horizon}_decisions'] = complete.sum(1)
            for name, prediction, reference, mask in (
                ('retained_common', output['planned_trajectory'], target, common),
                ('next_step', output['planned_trajectory'][..., :1, :], target[..., :1, :],
                 valid[..., :1] & output['plan_mask'][..., :1])):
                scores.update({f'{name}_{key}': value for key, value in window_errors(prediction, reference, mask).items()})
            scores.update(
                refreshed_regions_per_decision=output['refresh_mask'].float().sum(-1).mean(1),
                refresh_fraction=output['refresh_mask'].any(-1).float().mean(1),
                replan_fraction=output['replan_mask'].float().mean(1),
                forced_replan_fraction=output['forced_replan'].float().mean(1),
                event_refresh_fraction=output['event_refresh'].float().mean(1),
                common_endpoints=common.sum((1, 2)))
            for key in ('monitoring_reads', 'online_decode_calls', 'candidate_generated', 'candidate_accepted',
                        'candidate_rejected', 'candidate_evidence_reads', 'diagnostic_decode_calls'):
                scores[key] = output[key].sum(1).float()
            scores['history_valid_pairs'] = output['history_valid'].sum((1, 2)).float()
            scores['initial_decode_calls'] = output['initial_decode_calls'].float()
            scores['total_online_decode_calls'] = scores['initial_decode_calls'] + scores['online_decode_calls']
            values = {key: value.cpu().tolist() for key, value in scores.items()}
            for index in range(target.shape[0]):
                row = dict(policy=policy, sample=self.dataset._processed_file_names[offset + index],
                           recording=batch['recording'][index], event_type=batch['event_type'][index],
                           anchor_timestamp_us=int(batch['anchor_timestamp_us'][index]))
                row.update({key: v[index] if math.isfinite(v[index]) else None for key, v in values.items()})
                self.rows.append(row)
        print(f'Compared {offset + target.shape[0]}/{self.count} anchors', flush=True)

    def on_validation_epoch_end(self):
        keys = [key for key in self.rows[0] if key not in
                ('policy', 'sample', 'recording', 'event_type', 'anchor_timestamp_us')]
        groups = {policy: [row for row in self.rows if row['policy'] == policy] for policy in self.policies}
        comparisons = {}
        for baseline in self.policies[1:]:
            paired = {}
            for key in ('fixed10_ade_m', 'fixed10_fde_m', 'fixed5_ade_m', 'fixed5_fde_m',
                        'retained_common_ade_m', 'retained_common_fde_m', 'next_step_ade_m'):
                pairs = [(a[key], b[key]) for a, b in zip(groups['learned'], groups[baseline])
                         if a[key] is not None and b[key] is not None]
                delta = [b - a for a, b in pairs]
                mean = sum(delta) / len(delta) if delta else None
                base_mean = sum(b for a, b in pairs) / len(pairs) if pairs else None
                paired[key] = dict(paired_anchors=len(delta), mean_improvement_m=mean,
                                   improvement_percent=100 * mean / base_mean if base_mean else None,
                                   learned_win_fraction=sum(x > 1e-9 for x in delta) / len(delta) if delta else None)
            comparisons[baseline] = paired
        report = dict(
            policies={policy: {**aggregate(rows, keys),
                'by_event_type': {event: aggregate([r for r in rows if r['event_type'] == event], keys)
                    for event in sorted({r['event_type'] for r in rows})}} for policy, rows in groups.items()},
            learned_vs_baseline=comparisons,
            protocol=dict(closed_loop=False, task='expert log replay of causal CARS revision and selection',
                refresh_protocol=self.model.cars_adapter_protocol(),
                fixed10='Full 1.0s candidate ADE/FDE at complete decisions; candidate may be rejected.',
                fixed5='First 0.5s candidate prefix diagnostic; the main candidate contains ten steps.',
                retained_common='Actual retained plans scored on the common future intersection across policies; FDE horizon varies.',
                next_step='First endpoint of the actual plan at each 0.1s decision using logged localization.',
                policies={
                    'learned': ('generate candidate, then categorical keep/replace' if self.model.mode == 'plan1'
                                else 'generate candidate, then replace if predicted old-minus-new error exceeds epsilon'),
                    'no_refresh': 'retain until deadline; every feasible candidate is generated and counted',
                    'replan_only': 'generate and reject fresh candidates, then replan from stale memory; both decodes counted',
                    'fixed': 'adopt at configured interval plus deadlines',
                    'random': 'adopt with configured seeded random probability plus deadlines',
                    'random_2048': 'random policy with seed 2048', 'random_4096': 'random policy with seed 4096',
                    'all_fresh': 'generate and adopt adapted candidate every feasible decision',
                    'oracle': 'offline same-state GT L1-gain decision, nondeployable and not a global optimum',
                    'all_fresh_base': 'disable latent adapter and adopt frozen-base candidate every feasible decision',
                    'no_refresh_base': 'disable latent adapter and adopt only at deadlines; candidates still counted'},
                contribution='Compare all_fresh with all_fresh_base for adaptation, and learned with all_fresh for selection.',
                cost_matching='Fixed/random triggers are not automatically cost matched.',
                timing='Decode counts include generated rejected candidates and initial plan; offline diagnostic references excluded.',
                evidence_acquired='refresh_mask counts memory commits; candidate_evidence_reads counts fresh evidence used by every candidate.',
                inputs='Each policy owns causal state and retained plan; shared frozen predictor and completed logged action history; GT only scores offline labels/oracle.',
                aggregation='Equal-weight anchor window means; positive paired improvement means learned is better.',
                limitations='Offline validation; no closed-loop safety, independent-test significance, or saved candidate decode claim.'),
            run=dict(checkpoint=str(Path(self.config['trainer']['ckpt_path']).resolve()),
                config=self.config, seed=self.seed, device=str(self.device), torch_version=str(torch.__version__),
                unique_anchors=self.count, manifest=self.options.get('source_manifest', self.dataset.processed_file_names_path),
                model_hyperparameters=dict(self.model.hparams)))
        output = self.export(report)
        with (output / 'per_sample.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        print('Policy         fixed10 ADE/FDE (m)     retained ADE     next-step', flush=True)
        for policy, summary in report['policies'].items():
            metrics = summary['metrics']
            print(f"{policy:14} {metrics['fixed10_ade_m']} / {metrics['fixed10_fde_m']}    "
                  f"{metrics['retained_common_ade_m']}    {metrics['next_step_ade_m']}")
