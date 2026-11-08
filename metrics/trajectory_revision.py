"""Expert L1 trajectory revision with categorical or continuous plan comparison."""
import torch
from torch.nn import functional as F

from metrics.refresh_gate import gate_metrics


def masked_l1(prediction, target, mask):
    """One loss per supported item; padding may contain NaN and never contributes."""
    expanded = mask
    while expanded.ndim < prediction.ndim:
        expanded = expanded[..., None]
    difference = (prediction.masked_fill(~expanded, 0)
                  - target.detach().masked_fill(~expanded, 0)).abs()
    while difference.ndim > mask.ndim:
        difference = difference.mean(-1)
    return difference


def history_mean(values, valid):
    """First average pairs per decision, then average supported decisions."""
    counts = valid.sum(-1)
    supported = counts > 0
    per_decision = values.masked_fill(~valid, 0).sum(-1) / counts.clamp_min(1)
    return per_decision.masked_fill(~supported, 0).sum() / supported.sum().clamp_min(1)


def revision_metrics(output, target, valid, threshold=.5, trajectory_weight=1.,
                     classification_weight=1., history_weight=.25, heading_weight=.1,
                     quality_weight=1., gain_weight=1.):
    """Full candidate L1, actual-plan comparison, and auxiliary historical pairs.

    plan1 uses CE for relative preference. plan2 regresses both plan errors and
    their signed difference with L1. All geometry targets are detached; history
    is averaged within each decision to avoid overweighting longer histories.
    """
    metrics = gate_metrics(output, target, valid, threshold)
    candidate = output['candidate_trajectory']
    prefix = valid.bool().long().cumprod(-1).bool()
    mask = prefix & output['candidate_valid'].bool()[..., None]
    difference = candidate.masked_fill(~mask[..., None], 0) - target.detach().masked_fill(~mask[..., None], 0)
    endpoint_count = mask.sum(-1)
    supported = endpoint_count > 0
    denominator = endpoint_count.clamp_min(1)
    xy = difference[..., :2].abs().mean(-1).sum(-1) / denominator
    angle = torch.atan2(difference[..., 2].sin(), difference[..., 2].cos()).abs()
    heading = angle.sum(-1) / denominator
    support_count = supported.sum().clamp_min(1)
    xy_loss = xy.masked_fill(~supported, 0).sum() / support_count
    heading_loss = heading.masked_fill(~supported, 0).sum() / support_count
    trajectory = xy_loss + heading_weight * heading_loss
    history_valid = output['history_valid'].bool()
    eligible = output['decision_valid'].bool()
    quality_mode = output.get('planning_mode') == 'plan2' or 'quality_errors' in output
    if quality_mode:
        quality = masked_l1(output['quality_errors'], output['quality_targets'], eligible).sum() / eligible.sum().clamp_min(1)
        gain = masked_l1(output['predicted_gain'], output['replacement_gain'], eligible).sum() / eligible.sum().clamp_min(1)
        historical_quality = history_mean(masked_l1(
            output['history_quality_errors'], output['history_quality_targets'], history_valid), history_valid)
        predicted_history_gain = output['history_quality_errors'][..., 0] - output['history_quality_errors'][..., 1]
        historical_gain = history_mean(masked_l1(predicted_history_gain, output['history_gain'], history_valid), history_valid)
        history = quality_weight * historical_quality + gain_weight * historical_gain
        replacement = quality_weight * quality + gain_weight * gain
        metrics.update(quality_l1=quality, gain_l1=gain, history_quality_l1=historical_quality,
                       history_gain_l1=historical_gain, history_l1=history,
                       predicted_gain_mae=gain)
    else:
        replacement = classification_weight * metrics['loss']
        history_logits = output['history_logits']
        if history_valid.any():
            historical_errors = F.cross_entropy(history_logits[history_valid],
                output['history_labels'][history_valid].long(), reduction='none')
            errors = history_logits[..., 0].masked_fill(~history_valid, 0) * 0
            history = history_mean(errors.masked_scatter(history_valid, historical_errors), history_valid)
        else:
            history = history_logits.masked_fill(~history_valid[..., None], 0).sum() * 0
        metrics.update(replacement_ce=metrics['loss'], history_ce=history)
    metrics.update(loss=trajectory_weight * trajectory + replacement + history_weight * history,
                   trajectory_l1=trajectory, trajectory_xy_l1=xy_loss,
                   trajectory_heading_l1=heading_loss,
                   weighted_trajectory_loss=trajectory_weight * trajectory,
                   weighted_replacement_loss=replacement, weighted_history_loss=history_weight * history,
                   candidate_valid_decisions=supported.sum().float(),
                   history_valid_pairs=history_valid.sum().float())
    distance = difference[..., :2].norm(dim=-1)
    ade = distance.sum(-1) / denominator
    metrics['candidate_ade'] = ade.masked_fill(~supported, 0).sum() / support_count
    for horizon in sorted({min(5, candidate.shape[-2]), candidate.shape[-2]}):
        complete = mask[..., :horizon].all(-1)
        metrics[f'candidate_fixed{horizon}_ade'] = (distance[..., :horizon].mean(-1).masked_fill(~complete, 0).sum()
                                                       / complete.sum().clamp_min(1))
        metrics[f'candidate_fixed{horizon}_fde'] = (distance[..., horizon - 1].masked_fill(~complete, 0).sum()
                                                       / complete.sum().clamp_min(1))
    gain = output['replacement_gain'].detach().masked_fill(~eligible, 0)
    accepted = output.get('replace_decision', output['event_refresh']).bool() & eligible
    realized = gain.masked_fill(~accepted, 0)
    regret = torch.where(accepted, (-gain).clamp_min(0), gain.clamp_min(0))
    metrics['realized_gain'] = realized.sum() / eligible.sum().clamp_min(1)
    metrics['decision_regret'] = regret.masked_fill(~eligible, 0).sum() / eligible.sum().clamp_min(1)
    metrics['harmful_replacements'] = (accepted & (gain < 0)).sum().float()
    for key in ('candidate_generated', 'candidate_accepted', 'candidate_rejected',
                'candidate_evidence_reads', 'diagnostic_decode_calls', 'initial_decode_calls'):
        if key in output:
            metrics[key] = output[key].float().sum()
    if 'initial_decode_calls' in output:
        metrics['total_online_decode_calls'] = metrics['online_decode_calls'] + metrics['initial_decode_calls']
    return metrics
