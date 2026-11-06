"""Supervise latent trajectory revision without updating the action predictor."""
import torch
from torch.nn import functional as F

from metrics.refresh_gate import gate_metrics


def revision_metrics(output, target, valid, threshold=.5, trajectory_weight=1.,
                     classification_weight=1., history_weight=.25, heading_weight=.1):
    """Fixed coordinate/periodic L1 plus actual-plan and historical preference CE.

    Historical comparisons are averaged within each decision first. A decision
    with more surviving old candidates therefore does not receive more weight.
    Candidate construction remains differentiable; classification labels and
    historical plans are detached by the planner.
    """
    metrics = gate_metrics(output, target, valid, threshold)
    replacement = metrics['loss']
    candidate = output['candidate_trajectory']
    prefix = valid.bool().long().cumprod(-1).bool()
    generated = output['candidate_valid'].bool()
    mask = prefix & generated[..., None]
    safe_candidate = candidate.masked_fill(~mask[..., None], 0)
    safe_target = target.masked_fill(~mask[..., None], 0)
    difference = safe_candidate - safe_target
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

    history_logits = output['history_logits']
    history_valid = output['history_valid'].bool()
    history_labels = output['history_labels'].long()
    if history_valid.any():
        historical_errors = F.cross_entropy(
            history_logits[history_valid], history_labels[history_valid], reduction='none')
        errors = history_logits[..., 0] * 0
        errors = errors.masked_scatter(history_valid, historical_errors)
        counts = history_valid.sum(-1)
        history_supported = counts > 0
        per_decision = errors.sum(-1) / counts.clamp_min(1)
        history = per_decision[history_supported].mean()
    else:
        history = history_logits.sum() * 0
    metrics.update(loss=trajectory_weight * trajectory + classification_weight * replacement
                   + history_weight * history,
                   trajectory_l1=trajectory, trajectory_xy_l1=xy_loss,
                   trajectory_heading_l1=heading_loss, replacement_ce=replacement,
                   history_ce=history,
                   weighted_trajectory_loss=trajectory_weight * trajectory,
                   weighted_replacement_loss=classification_weight * replacement,
                   weighted_history_loss=history_weight * history,
                   candidate_valid_decisions=supported.sum().float(),
                   history_valid_pairs=history_valid.sum().float())
    distance = difference[..., :2].norm(dim=-1)
    ade = distance.sum(-1) / denominator
    metrics['candidate_ade'] = ade.masked_fill(~supported, 0).sum() / support_count
    complete = mask.all(-1)
    metrics['candidate_fixed5_ade'] = (ade.masked_fill(~complete, 0).sum()
                                      / complete.sum().clamp_min(1))
    metrics['candidate_fixed5_fde'] = (distance[..., -1].masked_fill(~complete, 0).sum()
                                       / complete.sum().clamp_min(1))
    gain = output['replacement_gain'].detach()
    eligible = output['decision_valid'].bool()
    accepted = output['event_refresh'].bool() & eligible
    realized = gain.masked_fill(~accepted, 0)
    regret = torch.where(accepted, (-gain).clamp_min(0), gain.clamp_min(0))
    metrics['realized_gain'] = realized.sum() / eligible.sum().clamp_min(1)
    metrics['decision_regret'] = regret.masked_fill(~eligible, 0).sum() / eligible.sum().clamp_min(1)
    metrics['harmful_replacements'] = (accepted & (gain < 0)).sum().float()
    for key in ('candidate_generated', 'candidate_accepted', 'candidate_rejected'):
        if key in output:
            metrics[key] = output[key].float().sum()
    for key in ('candidate_evidence_reads', 'diagnostic_decode_calls', 'initial_decode_calls'):
        if key in output:
            metrics[key] = output[key].float().sum()
    if 'initial_decode_calls' in output:
        metrics['total_online_decode_calls'] = metrics['online_decode_calls'] + metrics['initial_decode_calls']
    return metrics
