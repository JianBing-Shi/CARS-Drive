"""Offline CARS audit for categorical preference and continuous quality learning."""
import csv

import torch
from torch.nn import functional as F

from datasets.dataset_utils import local_actions
from metrics.evaluator import Evaluator
from metrics.refresh_comparison import rollout
from metrics.refresh_gate import binary_statistics, selection_statistics
from metrics.statistics import aggregate
from metrics.trajectory_revision import revision_metrics


class BinaryRefreshEvaluator(Evaluator):
    """Keep the CLI class name while reporting each planning objective accurately."""

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.model.mode not in {'plan1', 'plan2'}:
            raise ValueError('CARS audit requires a plan1 or plan2 checkpoint')
        self.rows, self.history_rows = [], []

    def validation_step(self, batch, batch_idx):
        model = self.model
        output = rollout(model, batch, 'learned')
        target, mask = batch['planning_targets'], batch['planning_mask'].bool()
        metrics = revision_metrics(
            output, target, mask, model.hparams.refresh_probability_threshold,
            trajectory_weight=model.hparams.refresh_trajectory_weight,
            classification_weight=model.hparams.refresh_classification_weight,
            history_weight=model.hparams.refresh_history_weight,
            heading_weight=model.hparams.refresh_heading_weight,
            quality_weight=model.hparams.get('quality_loss_weight', 1.),
            gain_weight=model.hparams.get('gain_loss_weight', 1.))
        self.log_dict({f'val/{key}': value for key, value in metrics.items() if torch.isfinite(value)},
                      batch_size=len(target))
        expert_actions = local_actions(torch.nan_to_num(target))
        offset = len(self.rows) // model.cars_adapter_steps
        for i in range(len(target)):
            for t in range(model.cars_adapter_steps):
                common = (mask[i, t] & output['plan_mask'][i, t]).long().cumprod(0).bool()
                n = int(common.sum())
                distance = (output['planned_trajectory'][i, t, :, :2] - target[i, t, :, :2]).norm(dim=-1)
                generated = bool(output['candidate_valid'][i, t])
                prefix = mask[i, t].long().cumprod(0).bool() & generated
                candidate_difference = (output['candidate_trajectory'][i, t].masked_fill(~prefix[:, None], 0)
                                        - target[i, t].masked_fill(~prefix[:, None], 0))
                candidate_error = candidate_difference[:, :2].norm(dim=-1)
                count = int(prefix.sum())
                xy = float(candidate_difference[:, :2].abs().mean(-1).sum() / count) if count else None
                heading = (float(torch.atan2(candidate_difference[:, 2].sin(), candidate_difference[:, 2].cos()).abs().sum() / count)
                           if count else None)
                eligible = bool(output['decision_valid'][i, t])
                adopted = bool(output['replace_decision'][i, t])
                gain = float(output['replacement_gain'][i, t]) if eligible else None
                history_valid = output['history_valid'][i, t].bool()
                row = dict(sample=str(self.dataset.samples[offset + i].relative_to(self.dataset.event_dir)),
                           recording=batch['recording'][i], event_type=batch['event_type'][i],
                           anchor_timestamp_us=int(batch['anchor_timestamp_us'][i]), decision_index=t,
                           relative_event_s=(model.cars_adapter_offset + 1 + t) * .1,
                           remaining_steps=int(output['remaining_steps'][i, t]),
                           valid_decision=eligible, label=int(output['refresh_labels'][i, t]),
                           replace_decision=adopted, requested=bool(output['refresh_request'][i, t]),
                           feasible=bool(output['refresh_feasible'][i, t]),
                           event_refresh=bool(output['event_refresh'][i, t]),
                           deadline_replan=bool(output['deadline_replan'][i, t]),
                           replan=bool(output['replan_mask'][i, t]),
                           evidence_acquired=bool(output['evidence_acquired'][i, t]),
                           monitoring_reads=int(output['monitoring_reads'][i, t]),
                           online_decode_calls=int(output['online_decode_calls'][i, t]),
                           common_endpoints=n, actual_horizon_s=n * .1,
                           ade_m=float(distance[common].mean()) if n else None,
                           fde_m=float(distance[n - 1]) if n else None,
                           next_step_error_m=float(distance[0]) if n else None,
                           replacement_gain=gain, candidate_valid=generated,
                           trajectory_xy_l1=xy, trajectory_heading_l1=heading,
                           trajectory_l1=xy + model.hparams.refresh_heading_weight * heading if count else None,
                           history_valid_pairs=int(history_valid.sum()),
                           realized_gain=gain if adopted else (0. if gain is not None else None),
                           decision_regret=(max(-gain, 0.) if adopted else max(gain, 0.)) if gain is not None else None,
                           harmful_replacement=bool(adopted and gain < 0) if gain is not None else None,
                           total_candidate_gain_m=(float(output['visual_gain'][i, t])
                               if 'visual_gain' in output and prefix.any() else None))
                for horizon in (5, 10):
                    full = len(prefix) >= horizon and bool(prefix[:horizon].all())
                    row[f'fixed{horizon}_candidate_ade_m'] = float(candidate_error[:horizon].mean()) if full else None
                    row[f'fixed{horizon}_candidate_fde_m'] = float(candidate_error[horizon - 1]) if full else None
                for key in ('candidate_generated', 'candidate_accepted', 'candidate_rejected', 'candidate_evidence_reads'):
                    row[key] = bool(output[key][i, t])
                row['diagnostic_decode_calls'] = int(output['diagnostic_decode_calls'][i, t])
                row['initial_decode_calls'] = int(output['initial_decode_calls'][i]) if t == 0 else 0
                row['total_online_decode_calls'] = row['online_decode_calls'] + row['initial_decode_calls']
                if model.mode == 'plan1':
                    row['probability'] = float(output['refresh_probability'][i, t])
                    row['replacement_ce'] = (float(F.cross_entropy(output['refresh_logits'][i, t][None],
                        output['refresh_labels'][i, t][None])) if eligible else None)
                    row['history_ce'] = (float(F.cross_entropy(output['history_logits'][i, t][history_valid],
                        output['history_labels'][i, t][history_valid])) if history_valid.any() else None)
                else:
                    errors, targets = output['quality_errors'][i, t], output['quality_targets'][i, t]
                    row.update(predicted_old_error=float(errors[0]), predicted_new_error=float(errors[1]),
                               true_old_error=float(targets[0]) if eligible else None,
                               true_new_error=float(targets[1]) if eligible else None,
                               predicted_gain=float(output['predicted_gain'][i, t]),
                               quality_l1=float((errors - targets).abs().mean()) if eligible else None,
                               gain_l1=abs(float(output['predicted_gain'][i, t]) - gain) if eligible else None)
                    errors = output['history_quality_errors'][i, t][history_valid]
                    targets = output['history_quality_targets'][i, t][history_valid]
                    history_gain = output['history_gain'][i, t][history_valid]
                    row['history_quality_l1'] = float((errors - targets).abs().mean()) if len(errors) else None
                    row['history_gain_l1'] = float(((errors[:, 0] - errors[:, 1]) - history_gain).abs().mean()) if len(errors) else None
                for lag in history_valid.nonzero().flatten().tolist():
                    history_row = dict(recording=row['recording'], event_type=row['event_type'],
                        anchor_timestamp_us=row['anchor_timestamp_us'], decision_index=t,
                        history_lag_frames=lag + 1, label=int(output['history_labels'][i, t, lag]))
                    if model.mode == 'plan1':
                        history_row['probability'] = float(output['history_logits'][i, t, lag].softmax(-1)[1])
                    else:
                        errors, targets = output['history_quality_errors'][i, t, lag], output['history_quality_targets'][i, t, lag]
                        predicted_gain = float(errors[0] - errors[1])
                        history_row.update(predicted_gain=predicted_gain,
                            replacement_gain=float(output['history_gain'][i, t, lag]),
                            replace_decision=predicted_gain > model.hparams.refresh_label_epsilon,
                            quality_l1=float((errors - targets).abs().mean()),
                            gain_l1=abs(predicted_gain - float(output['history_gain'][i, t, lag])))
                    self.history_rows.append(history_row)
                # Expert-derived motion groups are report labels, never model inputs.
                speed = float(expert_actions[i, t, 0, :2].norm() / .1)
                turn = abs(float(expert_actions[i, t, 0, 2] / .1))
                row['motion_group'] = 'stationary' if speed < .5 else ('turning' if turn >= .1 else 'moving_straight')
                self.rows.append(row)

    def _selection_summary(self, rows, history=False):
        labels = torch.tensor([row['label'] for row in rows])
        valid = torch.tensor([True if history else row['valid_decision'] for row in rows], dtype=torch.bool)
        if self.model.mode == 'plan1':
            stats = binary_statistics(torch.tensor([row['probability'] for row in rows]), labels,
                                      valid, self.model.hparams.refresh_probability_threshold)
        else:
            stats = selection_statistics(torch.tensor([row['replace_decision'] for row in rows], dtype=torch.bool), labels, valid)
        return {**{key: float(value) if torch.isfinite(value) else None for key, value in stats.items()},
                'valid_pairs' if history else 'valid_decisions': int(valid.sum())}

    def on_validation_epoch_end(self):
        unique, events = {}, {}
        for row in self.rows:
            identity = (row['recording'], row['anchor_timestamp_us'], row['decision_index'])
            unique.setdefault(identity, row)
            events.setdefault(row['event_type'], {}).setdefault(identity, row)
        rows = list(unique.values())
        metadata = {'sample', 'recording', 'event_type', 'anchor_timestamp_us', 'decision_index',
                    'motion_group', 'probability', 'label', 'valid_decision'}
        keys = [key for key in rows[0] if key not in metadata] if rows else []
        def summarize(group):
            stats = aggregate([{**row, **{key: float(row[key]) if row[key] is not None else None for key in keys}}
                               for row in group], keys)
            stats['selection'] = self._selection_summary(group)
            weights = [('trajectory_l1', self.model.hparams.refresh_trajectory_weight)]
            if self.model.mode == 'plan1':
                weights += [('replacement_ce', self.model.hparams.refresh_classification_weight),
                            ('history_ce', self.model.hparams.refresh_history_weight)]
            else:
                quality_weight = self.model.hparams.get('quality_loss_weight', 1.)
                gain_weight = self.model.hparams.get('gain_loss_weight', 1.)
                history_weight = self.model.hparams.refresh_history_weight
                weights += [('quality_l1', quality_weight), ('gain_l1', gain_weight),
                            ('history_quality_l1', history_weight * quality_weight),
                            ('history_gain_l1', history_weight * gain_weight)]
            stats['loss'] = sum((stats['metrics'].get(key) or 0.) * weight for key, weight in weights)
            stats['loss_reduction'] = 'weighted supported-component means; history averaged within each decision'
            return stats
        history = list({(row['recording'], row['anchor_timestamp_us'], row['decision_index'], row['history_lag_frames']): row
                        for row in self.history_rows}.values())
        history_summary = self._selection_summary(history, history=True)
        history_summary['by_lag_frames'] = {str(lag): self._selection_summary(
            [row for row in history if row['history_lag_frames'] == lag], history=True)
            for lag in sorted({row['history_lag_frames'] for row in history})}
        if self.model.mode == 'plan2':
            history_summary['error_metrics'] = aggregate(history, ['quality_l1', 'gain_l1'])
        report = dict(global_metrics=summarize(rows),
            by_event_type={key: summarize(list(value.values())) for key, value in events.items()},
            by_recording={key: summarize([row for row in rows if row['recording'] == key]) for key in sorted({row['recording'] for row in rows})},
            by_remaining_steps={str(key): summarize([row for row in rows if row['remaining_steps'] == key]) for key in sorted({row['remaining_steps'] for row in rows})},
            by_motion={key: summarize([row for row in rows if row['motion_group'] == key]) for key in sorted({row['motion_group'] for row in rows})},
            history_preferences=history_summary,
            decode_counts=dict(initial=sum(row['initial_decode_calls'] for row in rows),
                               decisions=sum(row['online_decode_calls'] for row in rows),
                               total_online=sum(row['total_online_decode_calls'] for row in rows),
                               diagnostic=sum(row['diagnostic_decode_calls'] for row in rows)),
            protocol=dict(self.model.cars_adapter_protocol(),
                labels='old-minus-new common-future coordinate position L1 plus weighted wrapped heading L1',
                supervision=('candidate trajectory L1, actual-plan and history preference CE' if self.model.mode == 'plan1'
                             else 'candidate trajectory L1, shared nonnegative quality L1 and signed gain L1; historical auxiliary L1'),
                history='valid old candidates only; actual retained source excluded; no requirement that the latest candidate wins',
                monitoring='causal current and past BEV; recent completed logged actions; no encoder savings claimed',
                cost='every feasible candidate, accepted or rejected, incurs ten decoder calls; initial and diagnostic calls reported separately',
                timing='offline diagnostic evaluation, not deployment latency',
                event_time='relative to scenario label, not verified event onset',
                total_candidate_gain='diagnostic stale-memory replan ADE minus fresh adapted candidate ADE; combines evidence and adaptation',
                regret='relative to raw common-future weighted L1 gain; label epsilon excluded',
                aggregation='unique recording/anchor/decision; history averaged per decision for training objective', closed_loop=False),
            checkpoint=self.config['trainer']['ckpt_path'], config=self.config,
            model_hyperparameters=dict(self.model.hparams))
        output = self.export(report)
        for name, exported_rows in (('per_decision.csv', self.rows), ('history_preferences.csv', self.history_rows)):
            if exported_rows:
                with (output / name).open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(exported_rows[0]))
                    writer.writeheader()
                    writer.writerows(exported_rows)
