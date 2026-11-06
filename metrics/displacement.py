"""Masked scene-level displacement metrics, including Plan-R1 joint modes."""
import torch
from torchmetrics import Metric


class JointDisplacement(Metric):
    """Choose one shared mode per scene; average only evaluated scenes.

    ADE includes all masked agent/time pairs. FDE includes agents valid at the
    configured final timestamp; it never substitutes an earlier endpoint.
    Empty scenes do not contribute to the denominator.
    """
    endpoint = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, scored_predict_traj, scored_target_traj, scored_target_mask):
        if not (len(scored_predict_traj) == len(scored_target_traj) == len(scored_target_mask)):
            raise ValueError("Prediction, target and mask scene counts must agree")
        for prediction, target, mask in zip(scored_predict_traj, scored_target_traj, scored_target_mask):
            if (prediction.ndim != 4 or prediction.shape[-1] != 2
                    or target.shape != (prediction.shape[0], prediction.shape[2], 2)
                    or mask.shape != target.shape[:2] or prediction.shape[1] == 0):
                raise ValueError("Expected [agents,modes,time,2], [agents,time,2], [agents,time]")
            mask = mask.bool()
            if self.endpoint:
                if target.shape[1] == 0:
                    continue
                prediction, target, mask = prediction[:, :, -1:], target[:, -1:], mask[:, -1:]
            if not mask.any():
                continue
            predicted = prediction.permute(1, 0, 2, 3)[:, mask]
            targets = target[mask]
            if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
                raise ValueError("Non-finite coordinates at valid endpoints")
            error = (predicted - targets[None]).norm(dim=-1).mean(-1).min()
            self.sum += error
            self.count += 1

    def compute(self):
        return self.sum / self.count.clamp_min(1) if self.count > 0 else self.sum.new_tensor(float("nan"))


def displacement_metrics(trajectory, expert, prefix):
    """Single-mode position errors; full-horizon and partial-prefix ADE separate."""
    if trajectory.shape != expert.shape or prefix.shape != expert.shape[:2]:
        raise ValueError("Expected matching [B,T,D] trajectories and [B,T] mask")
    distance = (trajectory[..., :2] - expert[..., :2]).norm(dim=-1)
    count = prefix.sum(-1)
    ade = distance.masked_fill(~prefix, 0).sum(-1) / count.masked_fill(count == 0, 1)
    ade = ade.masked_fill(count == 0, float("nan"))
    complete = prefix.all(-1)
    return {"ade_m": ade, "ade_full_horizon_m": ade.masked_fill(~complete, float("nan")),
            "fde_m": distance[:, -1].masked_fill(~complete, float("nan"))}
