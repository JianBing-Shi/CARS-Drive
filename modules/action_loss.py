"""L1 supervision for local actions and their integrated trajectories."""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ActionLoss(nn.Module):
    def __init__(self, action_loss: str = "l1", yaw_weight: float = 1., trajectory_weight: float = 0.,
                 heading_weight: float = 0., xy_weight: float = 1.):
        super().__init__()
        if action_loss != "l1":
            raise ValueError("action_loss is fixed to l1")
        weights = (xy_weight, yaw_weight, trajectory_weight, heading_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("L1 loss weights must be finite and nonnegative")
        self.action_loss = action_loss
        self.xy_weight = xy_weight
        self.yaw_weight = yaw_weight
        self.trajectory_weight = trajectory_weight
        self.heading_weight = heading_weight

    @staticmethod
    def trajectory(actions: Tensor) -> Tensor:
        yaw = actions[..., 2].cumsum(1)
        previous = F.pad(yaw[:, :-1], (1, 0))
        c, s = previous.cos(), previous.sin()
        local = torch.stack((c * actions[..., 0] - s * actions[..., 1], s * actions[..., 0] + c * actions[..., 1]), -1)
        return torch.cat((local.cumsum(1), yaw[..., None]), -1)

    @staticmethod
    def trajectory_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        error = (prediction[..., :2] - target[..., :2]).masked_fill(~valid[..., None], 0)
        distance = torch.linalg.vector_norm(error, dim=-1)
        return distance.sum(-1) / valid.sum(-1).clamp_min(1)

    @staticmethod
    def action_components(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        dyaw = torch.atan2(torch.sin(prediction[..., 2] - target[..., 2]), torch.cos(prediction[..., 2] - target[..., 2]))
        return (prediction[..., :2] - target[..., :2]).abs().mean(-1), dyaw.abs()

    def action_error(self, prediction: Tensor, target: Tensor) -> Tensor:
        xy, yaw = self.action_components(prediction, target)
        return self.xy_weight * xy + self.yaw_weight * yaw

    def forward(self, prediction: Tensor, target: Tensor, expert_trajectory: Tensor, valid: Tensor) -> dict[str, Tensor]:
        valid = valid.bool()
        # Do not evaluate NaN padding even in a subsequently masked loss branch.
        safe_prediction = prediction.masked_fill(~valid[..., None], 0)
        safe_target = target.masked_fill(~valid[..., None], 0)
        xy, yaw = self.action_components(safe_prediction, safe_target)
        count = valid.sum().clamp_min(1)
        xy = xy.masked_fill(~valid, 0).sum() / count
        yaw = yaw.masked_fill(~valid, 0).sum() / count
        prefix = valid.long().cumprod(-1).bool()
        trajectory = self.trajectory(safe_prediction)
        per_scene = self.trajectory_error(trajectory, expert_trajectory, prefix)
        available = prefix.any(-1)
        ade = per_scene.masked_fill(~available, 0).sum() / available.sum().clamp_min(1)
        # Position supervision is coordinate L1; Euclidean ADE stays an evaluation metric.
        position = (trajectory[..., :2] - expert_trajectory[..., :2]).masked_fill(~prefix[..., None], 0).abs().mean(-1)
        position = position.sum(-1) / prefix.sum(-1).clamp_min(1)
        position = position.masked_fill(~available, 0).sum() / available.sum().clamp_min(1)
        angle = (trajectory[..., 2] - expert_trajectory[..., 2]).masked_fill(~prefix, 0)
        heading = torch.atan2(angle.sin(), angle.cos()).abs().sum() / prefix.sum().clamp_min(1)
        weighted_xy = self.xy_weight * xy
        weighted_yaw = self.yaw_weight * yaw
        weighted_position = self.trajectory_weight * position
        weighted_heading = self.heading_weight * heading
        action = weighted_xy + weighted_yaw
        return {"loss": action + weighted_position + weighted_heading,
                "action_loss": action, "xy_loss": xy, "yaw_loss": yaw,
                "trajectory_loss": position, "heading_loss": heading, "ade": ade,
                "weighted_xy_loss": weighted_xy, "weighted_yaw_loss": weighted_yaw,
                "weighted_trajectory_loss": weighted_position, "weighted_heading_loss": weighted_heading}
