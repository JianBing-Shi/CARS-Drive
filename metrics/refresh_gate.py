"""Binary replacement supervision and masked trajectory/decision diagnostics."""
import torch
from torch.nn import functional as F


def binary_statistics(probability, labels, valid, threshold=.5):
    probability, labels = probability[valid], labels[valid].bool()
    predicted = probability > threshold
    tp = (predicted & labels).sum().float()
    fp = (predicted & ~labels).sum().float()
    fn = (~predicted & labels).sum().float()
    tn = (~predicted & ~labels).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    # Average precision with tied probabilities grouped, no optional sklearn dependency.
    ap = probability.new_tensor(float('nan'))
    if labels.any():
        order = probability.argsort(descending=True)
        sorted_probability, sorted_labels = probability[order], labels[order].float()
        end = torch.cat((sorted_probability[1:] != sorted_probability[:-1],
                         torch.ones(1, device=probability.device, dtype=torch.bool)))
        cumulative_tp = sorted_labels.cumsum(0)[end]
        count = torch.arange(1, len(labels) + 1, device=probability.device)[end]
        recall_points = cumulative_tp / labels.sum()
        ap = ((recall_points - F.pad(recall_points[:-1], (1, 0))) * cumulative_tp / count).sum()
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, positive_labels=labels.sum().float(),
                negative_labels=(~labels).sum().float(), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall).clamp_min(1e-12),
                pr_auc_ap=ap, unnecessary_refresh_rate=fp / (tp + fp).clamp_min(1),
                missed_refresh_rate=fn / (tp + fn).clamp_min(1))


def gate_metrics(output, target, valid, threshold=.5):
    eligible = output['decision_valid']
    logits, labels = output['refresh_logits'], output['refresh_labels']
    loss = F.cross_entropy(logits[eligible], labels[eligible]) if eligible.any() else logits.sum() * 0
    mask = (valid.bool() & output['plan_mask']).long().cumprod(-1).bool()
    difference = (output['planned_trajectory'][..., :2] - target[..., :2]).masked_fill(~mask[..., None], 0)
    distance = difference.norm(dim=-1)
    count = mask.sum().clamp_min(1)
    last = mask.sum(-1) - 1
    supported = last >= 0
    fde = distance.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
    metrics = dict(loss=loss, ade=distance.sum() / count,
                   retained_fde=fde.masked_fill(~supported, 0).sum() / supported.sum().clamp_min(1),
                   next_step_error=distance[..., 0].sum() / mask[..., 0].sum().clamp_min(1),
                   refresh_rate=output['event_refresh'].float().mean(),
                   request_rate=output['refresh_request'].float().mean(),
                   evidence_acquisition_rate=output['evidence_acquired'].float().mean(),
                   deadline_replan_rate=output['deadline_replan'].float().mean(),
                   replan_rate=output['replan_mask'].float().mean(),
                   valid_decisions=eligible.sum().float(),
                   monitoring_reads=output['monitoring_reads'].float().sum(),
                   online_decode_calls=output['online_decode_calls'].float().sum())
    metrics.update(binary_statistics(output['refresh_probability'], labels, eligible, threshold))
    return metrics
