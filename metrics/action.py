"""Per-window action and trajectory errors for expert-log replay."""
import torch
from torch import Tensor
from datasets.dataset_utils import local_actions
from modules import ActionLoss
from metrics.displacement import displacement_metrics as position_metrics


def sample_metrics(actions: Tensor, expert: Tensor, valid: Tensor,
                   loss_fn: ActionLoss, interval_s: float) -> dict[str, Tensor]:
    """Per-window metrics; missing targets are NaN rather than artificial zeros.

    A trajectory endpoint needs all preceding actions to be valid. FDE requires
    the configured horizon, rather than a shorter last available endpoint.
    """
    if actions.shape != expert.shape or valid.shape != expert.shape[:2]:
        raise ValueError("Expected actions/expert [B,T,3] and validity [B,T]")
    if not torch.isfinite(actions).all():
        raise ValueError("Non-finite predicted actions")
    if (valid & ~torch.isfinite(expert).all(-1)).any():
        raise ValueError("A valid expert pose contains non-finite values")
    expert = torch.nan_to_num(expert)
    targets = local_actions(expert)
    trajectory = ActionLoss.trajectory(actions)
    prefix = valid.long().cumprod(-1).bool()
    distance = (trajectory[..., :2] - expert[..., :2]).norm(dim=-1)
    dyaw = trajectory[..., 2] - expert[..., 2]
    heading = torch.atan2(dyaw.sin(), dyaw.cos()).abs()
    error = actions - targets
    error[..., 2] = torch.atan2(error[..., 2].sin(), error[..., 2].cos())

    def masked_mean(value, mask):
        count = mask.sum(-1).float()
        return value.masked_fill(~mask, 0).sum(-1) / count.masked_fill(count == 0, float("nan"))

    xy, yaw = loss_fn.action_components(actions, targets)
    xy = masked_mean(xy, valid)
    yaw = masked_mean(yaw, valid)
    position_l1 = masked_mean((trajectory[..., :2] - expert[..., :2]).abs().mean(-1), prefix)
    heading_l1 = masked_mean(heading, prefix)
    weighted_xy = loss_fn.xy_weight * xy
    weighted_yaw = loss_fn.yaw_weight * yaw
    weighted_position = loss_fn.trajectory_weight * position_l1
    weighted_heading = loss_fn.heading_weight * heading_l1
    action = weighted_xy + weighted_yaw
    positions = position_metrics(trajectory, expert, prefix)
    metrics = {
        **positions,
        "heading_mae_rad": masked_mean(heading, prefix),
        "heading_final_rad": heading[:, -1].masked_fill(~prefix.all(-1), float("nan")),
        "action_dx_mae_m": masked_mean(error[..., 0].abs(), valid),
        "action_dy_mae_m": masked_mean(error[..., 1].abs(), valid),
        "action_dyaw_mae_rad": masked_mean(error[..., 2].abs(), valid),
        "action_loss": action,
        "xy_loss": xy, "yaw_loss": yaw,
        "trajectory_loss": position_l1, "heading_loss": heading_l1,
        "weighted_xy_loss": weighted_xy, "weighted_yaw_loss": weighted_yaw,
        "weighted_trajectory_loss": weighted_position, "weighted_heading_loss": weighted_heading,
        "loss": action + weighted_position + weighted_heading,
    }
    for step in range(actions.shape[1]):
        metrics[f"l2_{(step + 1) * interval_s:.1f}s_m"] = distance[:, step].masked_fill(
            ~prefix[:, step], float("nan"))
    return metrics


