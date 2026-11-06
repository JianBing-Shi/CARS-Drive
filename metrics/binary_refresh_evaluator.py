"""Offline binary-gate audit; counterfactuals are excluded from online timing."""
import csv
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from datasets.dataset_utils import local_actions
from metrics.evaluator import Evaluator
from metrics.refresh_gate import gate_metrics, binary_statistics
from metrics.statistics import aggregate
from metrics.trajectory_revision import revision_metrics


class BinaryRefreshEvaluator(Evaluator):
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.rows = []
        self.history_rows = []

    def validation_step(self, batch, batch_idx):
        model = self.model
        features = batch['bev_features']
        valid = batch.get('bev_valid_mask', torch.ones_like(features[:,:,0], dtype=torch.bool)).bool()
        times = batch.get('timestamps_us')
        if times is None:
            times = torch.arange(features.shape[1], device=features.device)[None].expand(len(features),-1) * 100000
        output = model.planner(features, valid, times.double(), batch.get('reference_ego_to_global'), batch.get('bev_range'),
            start=model.history_features+model.refresh_offset, steps=model.refresh_steps,
            tokenizer=model.tokenizer, memory=model.memory, decoder=model.decoder,
            refresh=model.refresh, temporal=model.temporal, region_ids=model.region_ids,
            budget=model.refresh_budget, cost_weight=model.cost_weight,
            targets=batch['planning_targets'], target_mask=batch['planning_mask'], diagnostics=True)
        target, mask = batch['planning_targets'], batch['planning_mask'].bool()
        if model.refresh_version == 3:
            metrics = revision_metrics(
                output, target, mask, model.hparams.refresh_probability_threshold,
                trajectory_weight=model.hparams.refresh_trajectory_weight,
                classification_weight=model.hparams.refresh_classification_weight,
                history_weight=model.hparams.refresh_history_weight,
                heading_weight=model.hparams.refresh_heading_weight)
        else:
            metrics = gate_metrics(output, target, mask, model.hparams.refresh_probability_threshold)
        # Scalar training validation remains separate from exact deduplicated reports below.
        self.log_dict({f'val/{key}':value for key,value in metrics.items() if torch.isfinite(value)},
                      batch_size=len(features))
        expert_actions = local_actions(torch.nan_to_num(target))
        offset = len(self.rows) // model.refresh_steps
        for i in range(len(features)):
            for t in range(model.refresh_steps):
                common = (mask[i,t] & output['plan_mask'][i,t]).long().cumprod(0).bool()
                n = int(common.sum())
                distance = (output['planned_trajectory'][i,t,:,:2]-target[i,t,:,:2]).norm(dim=-1)
                candidate_valid = bool(output['candidate_valid'][i,t]) if model.refresh_version == 3 else True
                full = bool(mask[i,t].all()) and candidate_valid
                candidate_error = (output['candidate_trajectory'][i,t,:,:2]-target[i,t,:,:2]).norm(dim=-1)
                row = dict(sample=str(self.dataset.samples[offset+i].relative_to(self.dataset.event_dir)),
                           recording=batch['recording'][i], event_type=batch['event_type'][i],
                           anchor_timestamp_us=int(batch['anchor_timestamp_us'][i]), decision_index=t,
                           relative_event_s=(model.refresh_offset+1+t)*.1,
                           remaining_steps=int(output['remaining_steps'][i,t]),
                           valid_decision=bool(output['decision_valid'][i,t]),
                           label=int(output['refresh_labels'][i,t]),
                           probability=float(output['refresh_probability'][i,t]),
                           requested=bool(output['refresh_request'][i,t]), feasible=bool(output['refresh_feasible'][i,t]),
                           event_refresh=bool(output['event_refresh'][i,t]),
                           deadline_replan=bool(output['deadline_replan'][i,t]),
                           replan=bool(output['replan_mask'][i,t]),
                           evidence_acquired=bool(output['evidence_acquired'][i,t]),
                           monitoring_reads=1, online_decode_calls=int(output['online_decode_calls'][i,t]),
                           common_endpoints=n, actual_horizon_s=n*.1,
                           ade_m=float(distance[common].mean()) if n else None,
                           fde_m=float(distance[n-1]) if n else None,
                           next_step_error_m=float(distance[0]) if n else None,
                           fixed5_candidate_ade_m=float(candidate_error.mean()) if full else None,
                           fixed5_candidate_fde_m=float(candidate_error[-1]) if full else None,
                           replacement_gain_m=float(output['replacement_gain'][i,t]) if output['decision_valid'][i,t] else None,
                           visual_gain_m=(float(output['visual_gain'][i,t])
                                          if 'visual_gain' in output and mask[i,t].any() else None))
                if model.refresh_version == 3:
                    row['total_candidate_gain_m'] = row['visual_gain_m']
                    row['visual_gain_m'] = None
                    prefix = mask[i,t].long().cumprod(0).bool()
                    endpoint_count = int(prefix.sum()) if candidate_valid else 0
                    difference = (output['candidate_trajectory'][i,t] - target[i,t]).masked_fill(~prefix[:,None], 0)
                    xy = float(difference[:,:2].abs().mean(-1).sum()/max(endpoint_count,1)) if endpoint_count else None
                    heading = (float(torch.atan2(difference[:,2].sin(), difference[:,2].cos()).abs().sum()/endpoint_count)
                               if endpoint_count else None)
                    history_valid = output['history_valid'][i,t].bool()
                    history_ce = (float(F.cross_entropy(output['history_logits'][i,t][history_valid],
                                                        output['history_labels'][i,t][history_valid]))
                                  if history_valid.any() else None)
                    row.update(candidate_valid=candidate_valid,
                               candidate_generated=bool(output['candidate_generated'][i,t]),
                               candidate_accepted=bool(output['candidate_accepted'][i,t]),
                               candidate_rejected=bool(output['candidate_rejected'][i,t]),
                               candidate_evidence_reads=bool(output['candidate_evidence_reads'][i,t]),
                               diagnostic_decode_calls=int(output['diagnostic_decode_calls'][i,t]),
                               initial_decode_calls=int(output['initial_decode_calls'][i]) if t == 0 else 0,
                               total_online_decode_calls=(int(output['online_decode_calls'][i,t])
                                                          + (int(output['initial_decode_calls'][i]) if t == 0 else 0)),
                               trajectory_xy_l1=xy, trajectory_heading_l1=heading,
                               trajectory_l1=xy+model.hparams.refresh_heading_weight*heading if xy is not None else None,
                               replacement_ce=(float(F.cross_entropy(output['refresh_logits'][i,t][None],
                                                                     output['refresh_labels'][i,t][None]))
                                               if output['decision_valid'][i,t] else None),
                               history_ce=history_ce, history_valid_pairs=int(history_valid.sum()))
                    gain = row['replacement_gain_m']
                    adopted = row['event_refresh']
                    row['realized_gain_m'] = gain if adopted else (0. if gain is not None else None)
                    row['decision_regret_m'] = (max(-gain,0.) if adopted else max(gain,0.)) if gain is not None else None
                    row['harmful_replacement'] = bool(adopted and gain < 0) if gain is not None else None
                    for lag in history_valid.nonzero().flatten().tolist():
                        self.history_rows.append(dict(
                            recording=row['recording'], event_type=row['event_type'],
                            anchor_timestamp_us=row['anchor_timestamp_us'], decision_index=t,
                            history_lag_frames=lag+1,
                            label=int(output['history_labels'][i,t,lag]),
                            probability=float(output['history_logits'][i,t,lag].softmax(-1)[1])))
                # Motion groups are offline expert-label strata, never policy inputs.
                speed = float(expert_actions[i,t,0,:2].norm()/.1)
                turn = abs(float(expert_actions[i,t,0,2]/.1))
                row['motion_group'] = 'stationary' if speed < .5 else ('turning' if turn >= .1 else 'moving_straight')
                self.rows.append(row)

    def on_validation_epoch_end(self):
        metric_keys = ['ade_m','fde_m','next_step_error_m','fixed5_candidate_ade_m',
                       'fixed5_candidate_fde_m','replacement_gain_m','visual_gain_m',
                       'event_refresh','deadline_replan','replan','evidence_acquired',
                       'monitoring_reads','online_decode_calls','common_endpoints']
        if self.model.refresh_version == 3:
            metric_keys += ['candidate_generated','candidate_accepted','candidate_rejected',
                            'trajectory_xy_l1','trajectory_heading_l1','trajectory_l1',
                            'replacement_ce','history_ce','history_valid_pairs',
                            'realized_gain_m','decision_regret_m','harmful_replacement',
                            'total_candidate_gain_m','candidate_evidence_reads','diagnostic_decode_calls',
                            'initial_decode_calls','total_online_decode_calls']
        unique = {}
        events = {}
        for row in self.rows:
            identity = (row['recording'], row['anchor_timestamp_us'], row['decision_index'])
            unique.setdefault(identity, row)
            events.setdefault(row['event_type'], {}).setdefault(identity, row)
        rows = list(unique.values())
        def summarize(group):
            numeric = [{**row, **{key: float(row[key]) if row[key] is not None else None
                                  for key in metric_keys}} for row in group]
            stats = aggregate(numeric, metric_keys)
            probability = torch.tensor([r['probability'] for r in group])
            labels = torch.tensor([r['label'] for r in group])
            valid = torch.tensor([r['valid_decision'] for r in group], dtype=torch.bool)
            classification = binary_statistics(probability, labels, valid,
                                                self.model.hparams.refresh_probability_threshold)
            stats['classification'] = {k:float(v) if torch.isfinite(v) else None for k,v in classification.items()}
            stats['classification']['valid_decisions'] = int(valid.sum())
            if self.model.refresh_version == 3:
                components = stats['metrics']
                weights = (('trajectory_l1', self.model.hparams.refresh_trajectory_weight),
                           ('replacement_ce', self.model.hparams.refresh_classification_weight),
                           ('history_ce', self.model.hparams.refresh_history_weight))
                stats['loss'] = sum((components[key] or 0.)*weight for key,weight in weights)
                stats['loss_reduction'] = 'sum of weighted supported-component means; history averaged per decision'
            return stats
        report = dict(global_metrics=summarize(rows),
                      by_event_type={k:summarize(list(v.values())) for k,v in events.items()},
                      by_recording={k:summarize([r for r in rows if r['recording']==k]) for k in sorted({r['recording'] for r in rows})},
                      by_remaining_steps={str(k):summarize([r for r in rows if r['remaining_steps']==k]) for k in sorted({r['remaining_steps'] for r in rows})},
                      by_motion={k:summarize([r for r in rows if r['motion_group']==k]) for k in sorted({r['motion_group'] for r in rows})},
                      protocol=dict(self.model.refresh_protocol(), labels='on-policy keep-minus-new common-prefix ADE gain',
                                    monitoring='current full cached BEV every decision; encoder savings not claimed',
                                    cost='full valid-BEV memory updates; monitoring and deadline counts reported separately',
                                    timing='offline counterfactual scoring; not online latency',
                                    motion_groups='expert-label speed/yaw strata; not policy input; acceleration strata not inferred',
                                    aggregation='unique recording/anchor/decision globally; multi-label event groups preserved',
                                    event_time='relative to scenario label, not verified event onset', closed_loop=False),
                      checkpoint=self.config['trainer']['ckpt_path'], config=self.config,
                      model_hyperparameters=dict(self.model.hparams))
        if self.model.refresh_version == 3:
            unique_history = {}
            for row in self.history_rows:
                identity = (row['recording'],row['anchor_timestamp_us'],row['decision_index'],row['history_lag_frames'])
                unique_history.setdefault(identity,row)
            history = list(unique_history.values())
            history_scores = binary_statistics(
                torch.tensor([row['probability'] for row in history]),
                torch.tensor([row['label'] for row in history]),
                torch.ones(len(history),dtype=torch.bool), self.model.hparams.refresh_probability_threshold)
            report['history_preferences'] = {
                **{key:float(value) if torch.isfinite(value) else None for key,value in history_scores.items()},
                'valid_pairs':len(history)}
            by_lag = {}
            for lag in sorted({row['history_lag_frames'] for row in history}):
                group = [row for row in history if row['history_lag_frames']==lag]
                scores = binary_statistics(
                    torch.tensor([row['probability'] for row in group]),
                    torch.tensor([row['label'] for row in group]),
                    torch.ones(len(group),dtype=torch.bool), self.model.hparams.refresh_probability_threshold)
                by_lag[str(lag)] = {**{key:float(value) if torch.isfinite(value) else None
                                      for key,value in scores.items()},'valid_pairs':len(group)}
            report['history_preferences']['by_lag_frames'] = by_lag
            anchors = {(row['recording'],row['anchor_timestamp_us']) for row in rows}
            report['decode_counts'] = dict(
                initial=len(anchors)*self.model.plan_horizon,
                decisions=sum(row['online_decode_calls'] for row in rows),
                total_online=len(anchors)*self.model.plan_horizon+sum(row['online_decode_calls'] for row in rows),
                diagnostic=sum(row['diagnostic_decode_calls'] for row in rows))
            report['protocol'].update(
                task='causal trajectory revision and replacement using a frozen action predictor',
                supervision='fixed coordinate/periodic trajectory L1; actual-plan and valid-history preference CE',
                history='only old plans with common future endpoints; auxiliary pairs do not require the new plan to win',
                total_candidate_gain='old-memory unadapted replan ADE minus current-evidence adapted candidate ADE; combines evidence and adaptation',
                regret='relative to raw common-prefix ADE preference; label epsilon and computation cost are not included',
                cost='every feasible decision generates a candidate before acceptance; rejected candidates count toward decode calls',
                evidence_acquired='planning-memory commits; candidate_evidence_reads separately counts fresh evidence used even by rejected candidates',
                decode_counts='online_decode_calls per row excludes the shared initial plan; decode_counts includes it once per unique anchor',
                timing='offline diagnostic/reference branches excluded; candidate generation is part of the deployment path')
        output = self.export(report)
        with (output/'per_decision.csv').open('w', newline='', encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
        if self.model.refresh_version == 3 and self.history_rows:
            with (output/'history_preferences.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(self.history_rows[0]));writer.writeheader();writer.writerows(self.history_rows)
