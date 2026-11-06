"""Paired refresh interventions using the production rolling planner."""
import csv
import math
from pathlib import Path

import torch

from metrics.evaluator import Evaluator
from metrics.statistics import aggregate


POLICIES = ("learned", "no_refresh", "replan_only", "cyclic", "all_fresh")
BINARY_POLICIES = ("learned", "no_refresh", "replan_only", "fixed", "random", "random_2048", "random_4096", "all_fresh", "oracle")
REVISION_POLICIES = BINARY_POLICIES + ('all_fresh_base', 'no_refresh_base')


def rollout(model, batch, policy):
    extra = {}
    if model.refresh_version == 3 and policy.endswith('_base'):
        extra['adapter_enabled'] = False
        policy = policy[:-5]
    if model.refresh_version >= 2:
        extra.update(targets=batch.get('planning_targets'), target_mask=batch.get('planning_mask'), diagnostics=True)
        if policy.startswith('random_'):
            extra['random_seed'] = int(policy.split('_')[1])
            policy = 'random'
    return model.planner(
        batch["bev_features"], batch["bev_valid_mask"].bool(),
        batch["timestamps_us"].double(), batch.get("reference_ego_to_global"), batch.get("bev_range"),
        start=model.history_features + model.refresh_offset, steps=model.refresh_steps,
        tokenizer=model.tokenizer, memory=model.memory, decoder=model.decoder,
        refresh=model.refresh, temporal=model.temporal, region_ids=model.region_ids,
        budget=model.refresh_budget, cost_weight=model.cost_weight, refresh_policy=policy, **extra)


def window_errors(prediction, target, mask):
    """One mean per window; FDE uses the last common valid endpoint per decision."""
    error = (prediction[..., :2] - target[..., :2]).norm(dim=-1)
    count = mask.sum((1, 2))
    ade = error.masked_fill(~mask, 0).sum((1, 2)) / count.clamp_min(1)
    last = mask.long().sum(-1) - 1  # masks must be prefixes
    valid = last >= 0
    fde = error.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
    fde = fde.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
    return {"ade_m": ade.masked_fill(count == 0, float("nan")),
            "fde_m": fde.masked_fill(~valid.any(1), float("nan"))}


class RefreshComparisonEvaluator(Evaluator):
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        model = self.model
        if model.mode != "refresh":
            raise ValueError("Comparison requires a refresh checkpoint")
        self.policies = (REVISION_POLICIES if model.refresh_version == 3 else
                         BINARY_POLICIES if model.refresh_version == 2 else POLICIES)
        # Cached labels currently encode precisely this event-centred schedule.
        if (model.plan_horizon, model.refresh_offset, model.refresh_steps) != (5, -5, 10):
            raise ValueError("Cached planning labels require horizon=5, offset=-5, steps=10")
        self.rows = []

    def validation_step(self, batch, batch_idx):
        if not (batch["timestamps_us"].diff(dim=1) == 100_000).all():
            raise ValueError("Expected uniformly sampled 10 Hz windows")
        outputs = {policy: rollout(self.model, batch, policy) for policy in self.policies}
        target, valid = batch["planning_targets"].float(), batch["planning_mask"].bool()
        complete = valid.all(-1)[..., None].expand_as(valid)
        common = valid.clone()
        for output in outputs.values():
            common &= output["plan_mask"]
        offset = len(self.rows) // len(self.policies)
        for policy, output in outputs.items():
            scores = {}
            candidate_complete = complete
            if self.model.refresh_version == 3:
                candidate_complete = complete & output['candidate_valid'][..., None]
            for name, prediction, mask in (
                ("fixed5", output["candidate_trajectory"], candidate_complete),
                ("retained_common", output["planned_trajectory"], common),
                ("next_step", output["planned_trajectory"][..., :1, :], valid[..., :1])):
                scores.update({f"{name}_{key}": value for key, value in window_errors(prediction, target[..., :prediction.shape[-2], :], mask).items()})
            scores.update(
                refreshed_regions_per_decision=output["refresh_mask"].float().sum(-1).mean(1),
                refresh_fraction=output["refresh_mask"].any(-1).float().mean(1),
                replan_fraction=output["replan_mask"].float().mean(1),
                forced_replan_fraction=output["forced_replan"].float().mean(1),
                fixed5_decisions=candidate_complete[..., 0].sum(1),
                common_endpoints=common.sum((1, 2)))
            if self.model.refresh_version >= 2:
                scores.update(event_refresh_fraction=output['event_refresh'].float().mean(1),
                              monitoring_reads=output['monitoring_reads'].sum(1).float(),
                              online_decode_calls=output['online_decode_calls'].sum(1).float())
            if self.model.refresh_version == 3:
                scores.update(candidate_generated=output['candidate_generated'].sum(1).float(),
                              candidate_accepted=output['candidate_accepted'].sum(1).float(),
                              candidate_rejected=output['candidate_rejected'].sum(1).float(),
                              history_valid_pairs=output['history_valid'].sum((1,2)).float(),
                              candidate_evidence_reads=output['candidate_evidence_reads'].sum(1).float(),
                              diagnostic_decode_calls=output['diagnostic_decode_calls'].sum(1).float(),
                              initial_decode_calls=output['initial_decode_calls'].float(),
                              total_online_decode_calls=output['initial_decode_calls'].float()
                                                        +output['online_decode_calls'].sum(1).float())
            values = {key: value.cpu().tolist() for key, value in scores.items()}
            for index in range(target.shape[0]):
                row = dict(policy=policy, sample=self.dataset._processed_file_names[offset + index],
                           recording=batch["recording"][index], event_type=batch["event_type"][index],
                           anchor_timestamp_us=int(batch["anchor_timestamp_us"][index]))
                row.update({key: v[index] if math.isfinite(v[index]) else None for key, v in values.items()})
                self.rows.append(row)
        print(f"Compared {offset + target.shape[0]}/{self.count} anchors", flush=True)

    def on_validation_epoch_end(self):
        keys = [key for key in self.rows[0] if key not in
                ("policy", "sample", "recording", "event_type", "anchor_timestamp_us")]
        groups = {policy: [row for row in self.rows if row["policy"] == policy] for policy in self.policies}
        comparisons = {}
        for baseline in self.policies[1:]:
            paired = {}
            for key in ("fixed5_ade_m", "fixed5_fde_m", "retained_common_ade_m", "retained_common_fde_m", "next_step_ade_m"):
                pairs = [(a[key], b[key]) for a, b in zip(groups["learned"], groups[baseline])
                         if a[key] is not None and b[key] is not None]
                delta = [b - a for a, b in pairs]
                mean = sum(delta) / len(delta) if delta else None
                base_mean = sum(b for a, b in pairs) / len(pairs) if pairs else None
                paired[key] = dict(paired_anchors=len(delta), mean_improvement_m=mean,
                                   improvement_percent=100 * mean / base_mean if base_mean else None,
                                   learned_win_fraction=sum(x > 1e-9 for x in delta) / len(delta) if delta else None)
            comparisons[baseline] = paired
        report = {
            "policies": {policy: {
                **aggregate(rows, keys),
                "by_event_type": {event: aggregate([r for r in rows if r['event_type'] == event], keys)
                                  for event in sorted({r['event_type'] for r in rows})}}
                for policy, rows in groups.items()},
            "learned_vs_baseline": comparisons,
            "protocol": {
                "closed_loop": False, "task": "expert log replay of rolling planner",
                "fixed5": "Post-acquisition candidate; 0.5s ADE/FDE over identical complete decisions. This candidate may not be executed by keep-plan policies.",
                "retained_common": "Actual retained plans scored on intersection of all policy plan masks and GT prefix masks. FDE horizon varies; it is not fixed5 FDE.",
                "next_step": "First endpoint of the actual plan at each 0.1s decision, using logged localization.",
                "policies": {
                    "learned": "Deployed learned region selection and deadline replanning",
                    "no_refresh": "No visual acquisition; replan only on five-step deadline",
                    "replan_only": "No visual acquisition; replan each decision",
                    "cyclic": "Acquire one region per decision in cyclic index order, subject to budget; same decoder",
                    "all_fresh": "Acquire all available regions every decision; higher-cost reference, exceeds learned budget"},
                "aggregation": "Unique recording/timestamp anchors; equal-weight window means; positive improvement means learned is better.",
                "inputs": "Each policy owns its continuous state and plan; shared initial history, frozen predictor and logged poses; GT is used only for scoring.",
                "limitations": "Validation-set comparison; no independent-test or statistical-significance claim. Cached-BEV region counts measure acquisition proxy, not encoder latency or FLOPs."},
            "run": {"checkpoint": str(Path(self.config["trainer"]["ckpt_path"]).resolve()),
                    "config": self.config, "seed": self.seed, "device": str(self.device),
                    "torch_version": str(torch.__version__), "unique_anchors": self.count,
                    "manifest": self.options.get('source_manifest', self.dataset.processed_file_names_path),
                    "model_hyperparameters": dict(self.model.hparams)}}
        if self.model.refresh_version >= 2:
            report['protocol'].update(
                refresh_protocol=self.model.refresh_protocol(),
                policies={'learned':'binary softmax keep/replace',
                          'no_refresh':'deadline-only refresh using the shared full-BEV maintenance rule',
                          'replan_only':'stale-memory replan every decision, no new planning evidence',
                          'fixed':'full-BEV refresh at configured fixed interval plus deadlines',
                          'random':'seeded random trigger with configured probability plus deadlines',
                          'random_2048':'random policy, seed 2048', 'random_4096':'random policy, seed 4096',
                          'all_fresh':'full-BEV refresh and replan every decision',
                          'oracle':'offline same-state GT W decision only; nondeployable and not a global optimum'},
                cost_matching='Fixed interval/random probability require development calibration; not automatically cost matched.',
                timing='Offline candidates are diagnostics only; online decode-call count excludes counterfactuals.',
                inputs='Current full cached BEV is monitored by all policies; full evidence acquired under shared feasible deadline rule except stale-memory replan control.')
        if self.model.refresh_version == 3:
            report['protocol'].update(
                task='trajectory revision with shared frozen action predictor and shared trained revision adapter',
                policies={**report['protocol']['policies'],
                          'learned':'generate adapted candidate, then classify keep/replace',
                          'all_fresh':'generate and adopt adapted candidate each feasible decision',
                          'no_refresh':'retain until deadline; feasible adapted candidates are generated and counted each decision',
                          'replan_only':'generate and reject fresh candidates, then replan from stale memory each decision; both decodes counted',
                          'all_fresh_base':'disable latent adapter; adopt the frozen base predictor candidate each feasible decision',
                          'no_refresh_base':'disable latent adapter; adopt base candidates only at deadlines; all candidates are still counted'},
                timing='Decode-call counts include all candidates generated by the deployed policy, including rejected candidates; offline reference candidates excluded.',
                evidence_acquired='refresh_mask counts memory commits; candidate_evidence_reads counts fresh evidence used by all generated candidates',
                contribution='Unsuffixed policies share the trained adapter. The explicit _base controls disable it and share the frozen action predictor; compare all_fresh with all_fresh_base to isolate adaptation.')
        output = self.export(report)
        with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        print("Policy         fixed5 ADE/FDE (m)     retained ADE     next-step     regions/decision", flush=True)
        for policy, summary in report["policies"].items():
            m = summary["metrics"]
            print(f"{policy:14} {m['fixed5_ade_m']} / {m['fixed5_fde_m']}    "
                  f"{m['retained_common_ade_m']}    {m['next_step_ade_m']}    "
                  f"{m['refreshed_regions_per_decision']:.3f}")
        improvement = comparisons["no_refresh"]["fixed5_ade_m"]["mean_improvement_m"]
        print(f"Learned vs no_refresh fixed5 ADE improvement: {improvement} m (positive = better)")
