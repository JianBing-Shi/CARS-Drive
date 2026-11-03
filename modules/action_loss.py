"""Local action supervision and integrated trajectory metrics."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ActionLoss(nn.Module):
    def __init__(self, action_loss: str, yaw_weight: float, trajectory_weight: float):
        super().__init__()
        self.action_loss = action_loss
        self.yaw_weight = yaw_weight
        self.trajectory_weight = trajectory_weight

    @staticmethod
    def trajectory(actions: Tensor) -> Tensor:
        yaw = actions[..., 2].cumsum(1)
        previous = F.pad(yaw[:, :-1], (1, 0))
        c, s = previous.cos(), previous.sin()
        local = torch.stack((c * actions[..., 0] - s * actions[..., 1], s * actions[..., 0] + c * actions[..., 1]), -1)
        return torch.cat((local.cumsum(1), yaw[..., None]), -1)

    @staticmethod
    def trajectory_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        distance = torch.linalg.vector_norm(prediction[..., :2] - target[..., :2], dim=-1)
        return (distance * valid.float()).sum(-1) / valid.float().sum(-1).clamp_min(1)

    def action_error(self, prediction: Tensor, target: Tensor) -> Tensor:
        dyaw = torch.atan2(torch.sin(prediction[..., 2] - target[..., 2]), torch.cos(prediction[..., 2] - target[..., 2]))
        loss = F.l1_loss if self.action_loss == "l1" else F.smooth_l1_loss
        return loss(prediction[..., :2], target[..., :2], reduction="none").mean(-1) + self.yaw_weight * loss(dyaw, torch.zeros_like(dyaw), reduction="none")

    def forward(self, prediction: Tensor, target: Tensor, expert_trajectory: Tensor, valid: Tensor) -> dict[str, Tensor]:
        action = (self.action_error(prediction, target) * valid).sum() / valid.sum().clamp_min(1)
        ade = self.trajectory_error(self.trajectory(prediction), expert_trajectory, valid).mean()
        return {"loss": action + self.trajectory_weight * ade,
                "action_loss": action, "trajectory_loss": ade, "ade": ade}
