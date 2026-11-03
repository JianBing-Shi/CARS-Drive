"""Lightning entry point for causal action prediction and evidence refresh."""
from __future__ import annotations

import math
from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules import BEVTemporal
from modules import BEVTokenizer 
from modules import BEVMemory
from modules import ActionDecoder
from modules import RollingPlanner
from modules import FreshModule
from modules import ActionLoss

class FRESHDrive(pl.LightningModule):
    """Causal BEV model with action and refresh training modes."""

    def __init__(self, history_features: int = 20, future_action: int = 10, feature_dim: int = 256, hidden_dim: int = 256,
                 spatial_grid: tuple[int, int] = (9, 9), macro_grid: tuple[int, int] = (3, 3), heads: int = 8, layers: int = 2,
                 dropout: float = .1, mode: str = "action", refresh_budget: float = 1., max_refresh: int = 1, cost_weight: float = 0.,
                 utility_weight: float = 1., action_loss: str = "smooth_l1", yaw_weight: float = 1., trajectory_loss_weight: float = 0.,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4, warmup_epochs: int = 4, T_max: int = 32,
                 anchor_dropout: float = .2, cameras: int | None = None,
                 plan_horizon: int = 5, refresh_steps: int = 10, refresh_offset: int = -5) -> None:
        super().__init__()
        if mode not in {"action", "refresh"} or action_loss not in {"l1", "smooth_l1"}:
            raise ValueError("invalid mode or action_loss")
        if max_refresh != 1:
            raise ValueError("first architecture supports an empty set or one region")
        if not 0 <= anchor_dropout < 1:
            raise ValueError("anchor_dropout must be in [0, 1)")
        if not isinstance(warmup_epochs, int) or not isinstance(T_max, int) or warmup_epochs < 0 or T_max <= 0 or warmup_epochs >= T_max:
            raise ValueError("require 0 <= warmup_epochs < T_max")
        self.save_hyperparameters()

        # 数据窗口与网络配置
        self.mode = mode
        self.history_features = history_features
        self.future_action = future_action
        self.hidden_dim = hidden_dim
        self.grid = tuple(spatial_grid)
        self.regions = macro_grid[0] * macro_grid[1]

        # 优化器与损失配置
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.trajectory_loss_weight = trajectory_loss_weight
        self.utility_weight = utility_weight

        # 视觉更新配置
        self.refresh_budget = refresh_budget
        self.cost_weight = cost_weight
        self.anchor_dropout = anchor_dropout
        rows = torch.arange(self.grid[0]) * macro_grid[0] // self.grid[0]
        cols = torch.arange(self.grid[1]) * macro_grid[1] // self.grid[1]
        region_ids = rows[:, None] * macro_grid[1] + cols[None, :]
        self.register_buffer("region_ids", region_ids.flatten(), persistent=False)

        # 动作预测器
        self.tokenizer = BEVTokenizer(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            grid=self.grid,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        self.memory = BEVMemory(hidden_dim=hidden_dim)
        self.decoder = ActionDecoder(
            hidden_dim=hidden_dim,
            future_action=future_action,
            heads=heads,
        )

        # 时序状态与证据刷新
        self.temporal = BEVTemporal(
            history_features=history_features,
            hidden_dim=hidden_dim,
            grid=self.grid,
        )
        self.plan_horizon = plan_horizon
        self.refresh_steps = refresh_steps
        self.refresh_offset = refresh_offset
        self.planner = RollingPlanner(plan_horizon)
        self.refresh = FreshModule(hidden_dim=hidden_dim, regions=self.regions)

        # 动作损失与轨迹评价
        self.action_loss = ActionLoss(
            action_loss=action_loss,
            yaw_weight=yaw_weight,
            trajectory_weight=trajectory_loss_weight,
        )

        # 在优化器创建前确定可训练参数
        if self.mode == "refresh":
            self.freeze_pred_model()
        else:
            for p in self.refresh.parameters():
                p.requires_grad = False

    def forward(self, bev_features: Tensor, bev_valid_mask: Tensor | None = None, timestamps_us: Tensor | None = None,
                anchor_timestamp_us: Tensor | None = None, future_actions: Tensor | None = None, bev_age_s: Tensor | None = None,
                *, budget: Tensor | float | None = None, region_costs: Tensor | None = None, return_details: bool = False,
                teacher_forcing: bool = False, reference_ego_to_global: Tensor | None = None,
                bev_range: Tensor | None = None, planning_targets: Tensor | None = None,
                planning_mask: Tensor | None = None) -> Tensor | dict[str, Tensor]:
        if self.mode == "refresh":
            self.freeze_pred_model()
            valid = torch.ones_like(bev_features[:, :, 0], dtype=torch.bool) if bev_valid_mask is None else bev_valid_mask.bool()
            times = timestamps_us.double() if timestamps_us is not None else torch.arange(bev_features.shape[1], device=bev_features.device)[None].expand(bev_features.shape[0], -1).double() * 100_000
            if times.ndim == 3:
                times = times.amax(-1)
            result = self.planner(
                bev_features, valid, times, reference_ego_to_global, bev_range,
                start=self.history_features + self.refresh_offset, steps=self.refresh_steps,
                tokenizer=self.tokenizer, memory=self.memory, decoder=self.decoder,
                refresh=self.refresh, temporal=self.temporal, region_ids=self.region_ids,
                budget=self.refresh_budget, cost_weight=self.cost_weight,
                targets=planning_targets, target_mask=planning_mask)
            return result if return_details else result["planned_trajectory"]

        if bev_features.ndim != 5 or bev_features.shape[1] < self.history_features + self.future_action:
            raise ValueError("Sequential replay requires history_features + future_action BEV frames")
        batch, steps = bev_features.shape[:2]
        valid = torch.ones(batch, steps, bev_features.shape[-2], bev_features.shape[-1], dtype=torch.bool, device=bev_features.device) if bev_valid_mask is None else bev_valid_mask.bool()
        ages = self.temporal.ages(bev_features, timestamps_us, anchor_timestamp_us, bev_age_s)
        if reference_ego_to_global is None:
            state = self.temporal.encode_history(bev_features[:, :self.history_features], valid[:, :self.history_features], ages[:, :self.history_features], self.tokenizer, self.memory)
        else:
            state = self.temporal.encode_history(bev_features[:, :1], valid[:, :1], ages[:, :1], self.tokenizer, self.memory)
            for frame in range(1, self.history_features):
                state = self.temporal.align_memory(state, reference_ego_to_global[:, frame - 1], reference_ego_to_global[:, frame], bev_range)
                tokens, frame_valid = self.tokenizer(bev_features[:, frame], ages[:, frame], valid[:, frame])
                state = {"memory": self.memory(state["memory"], tokens, frame_valid),
                         "valid": state["valid"] | frame_valid,
                         "age_s": torch.where(frame_valid, ages[:, frame, None], state["age_s"])}
            state["reference_pose"] = reference_ego_to_global[:, self.history_features - 1]
        previous = bev_features.new_zeros(batch, 3, dtype=torch.float32)
        hidden, outputs = None, []
        for step in range(self.future_action):
            frame = self.history_features + step
            # Ages are measured relative to the moving decision time.
            elapsed = ages[:, frame - 1] - ages[:, frame] if step else ages.new_zeros(batch)
            output = self.observe_step(bev_features[:, frame], valid[:, frame], state,
                                       previous, hidden, step, elapsed, ages[:, frame].clamp_min(0),
                                       budget, region_costs,
                                       None if reference_ego_to_global is None else reference_ego_to_global[:, frame], bev_range)
            state, hidden = output["state"], output["hidden"]
            previous = future_actions[:, step] if teacher_forcing and future_actions is not None else output["actions"]
            outputs.append(output)
        keys = ("actions", "proposal_actions", "queries", "refresh_mask", "refresh_utility")
        result = {key: torch.stack([output[key] for output in outputs], 1) for key in keys}
        return result if return_details else result["actions"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        if self.mode == "action":
            # Expert poses use the anchor coordinate frame; actions use each step's ego frame.
            expert = batch["future_trajectory"].float()
            poses = F.pad(expert, (0, 0, 1, 0))
            delta = poses[:, 1:] - poses[:, :-1]
            yaw = poses[:, :-1, 2]
            target = torch.stack((yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                                  -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                                  torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())), -1)
            result = self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), target, return_details=True, teacher_forcing=True,
                          reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))
            valid = batch["future_action_mask"].bool()
            losses = self.action_loss(result["actions"], target, expert, valid)
            loss = losses["loss"]
            self.log("train/trajectory_loss", losses["trajectory_loss"], batch_size=target.shape[0])
            self.log("train/loss", loss, prog_bar=True, batch_size=target.shape[0])
            self.log("train/action_loss", losses["action_loss"], batch_size=target.shape[0])
            self.log("train/ade", losses["ade"], batch_size=target.shape[0])
            return loss

        elif self.mode == "refresh":
            result = self(
                batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"),
                reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"),
                planning_targets=batch["planning_targets"], planning_mask=batch["planning_mask"], return_details=True)
            valid = batch["planning_mask"].any(-1)
            errors = F.huber_loss(result["refresh_utility"], result["fresh_targets"], reduction="none").mean(-1)
            loss = self.utility_weight * (errors * valid).sum() / valid.sum().clamp_min(1)
            mask = batch["planning_mask"] & result["plan_mask"]
            distance = (result["planned_trajectory"][..., :2] - batch["planning_targets"][..., :2]).norm(dim=-1)
            ade = (distance * mask).sum() / mask.sum().clamp_min(1)
            self.log("train/loss", loss, prog_bar=True, batch_size=valid.shape[0])
            self.log("train/ade", ade, batch_size=valid.shape[0])
            self.log("train/refresh_rate", result["refresh_mask"].any(-1).float().mean(), batch_size=valid.shape[0])
            self.log("train/replan_rate", result["replan_mask"].float().mean(), batch_size=valid.shape[0])
            return loss

    def on_train_start(self) -> None:
        """Prepare the fixed action predictor before refresh training starts."""
        if self.mode == "refresh":
            self.freeze_pred_model()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        if self.mode == "action":
            # Expert poses use the anchor coordinate frame; actions use each step's ego frame.
            expert = batch["future_trajectory"].float()
            poses = F.pad(expert, (0, 0, 1, 0))
            delta = poses[:, 1:] - poses[:, :-1]
            yaw = poses[:, :-1, 2]
            target = torch.stack((yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                                  -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                                  torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())), -1)
            result = self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), target, return_details=True, teacher_forcing=False,
                          reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))
            valid = batch["future_action_mask"].bool()
            losses = self.action_loss(result["actions"], target, expert, valid)
            loss = losses["loss"]
            self.log("val/trajectory_loss", losses["trajectory_loss"], batch_size=target.shape[0])
            self.log("val/loss", loss, prog_bar=True, batch_size=target.shape[0])
            self.log("val/action_loss", losses["action_loss"], batch_size=target.shape[0])
            self.log("val/ade", losses["ade"], batch_size=target.shape[0])
            return loss

        elif self.mode == "refresh":
            result = self(
                batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"),
                reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"),
                planning_targets=batch["planning_targets"], planning_mask=batch["planning_mask"], return_details=True)
            valid = batch["planning_mask"].any(-1)
            errors = F.huber_loss(result["refresh_utility"], result["fresh_targets"], reduction="none").mean(-1)
            loss = self.utility_weight * (errors * valid).sum() / valid.sum().clamp_min(1)
            mask = batch["planning_mask"] & result["plan_mask"]
            distance = (result["planned_trajectory"][..., :2] - batch["planning_targets"][..., :2]).norm(dim=-1)
            ade = (distance * mask).sum() / mask.sum().clamp_min(1)
            self.log("val/loss", loss, prog_bar=True, batch_size=valid.shape[0])
            self.log("val/ade", ade, batch_size=valid.shape[0])
            self.log("val/refresh_rate", result["refresh_mask"].any(-1).float().mean(), batch_size=valid.shape[0])
            self.log("val/replan_rate", result["replan_mask"].float().mean(), batch_size=valid.shape[0])
            return loss

    def freeze_pred_model(self) -> None:
        self.tokenizer.eval()
        self.memory.eval()
        self.decoder.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.memory.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def observe_step(self, feature: Tensor, valid: Tensor, state: dict[str, Tensor],
                     previous: Tensor, hidden: Tensor | None = None, step: int = 0,
                     elapsed_s: Tensor | float = .1, age_s: Tensor | None = None,
                     budget: Tensor | float | None = None,
                     region_costs: Tensor | None = None,
                     reference_pose: Tensor | None = None, bev_range: Tensor | None = None) -> dict[str, Any]:
        """Consume one arrived observation; return one action and reusable state."""
        batch = feature.shape[0]
        if reference_pose is not None:
            state = self.temporal.align_memory(state, state["reference_pose"], reference_pose, bev_range)
        elapsed = torch.as_tensor(elapsed_s, device=feature.device).reshape(-1)
        state = {**state, "age_s": state["age_s"] + elapsed[:, None]}
        age = feature.new_zeros(batch, dtype=torch.float32) if age_s is None else age_s
        if self.mode == "action":
            frame_valid = valid.clone()
            if self.training and self.anchor_dropout:
                keep = torch.rand(batch, device=feature.device) >= self.anchor_dropout
                frame_valid &= keep[:, None, None]
            selected = torch.ones(batch, self.regions, dtype=torch.bool, device=feature.device)
            updated = self.refresh.update_anchor(state, feature, frame_valid, age, selected, self.region_ids, self.tokenizer, self.memory)
            action, query = self.decoder(updated["memory"], updated["valid"], previous, hidden, step)
            if reference_pose is not None:
                updated["reference_pose"] = reference_pose
            return {"actions": action, "proposal_actions": action, "queries": query,
                    "refresh_mask": torch.zeros_like(selected),
                    "refresh_utility": feature.new_zeros(batch, self.regions),
                    "state": updated, "hidden": query}
        raise ValueError("Refresh uses RollingPlanner to retain and update five-step plans")

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.MultiheadAttention, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)
        no_decay_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                if "bias" in name or "weight" not in name:
                    no_decay.add(full_name)
                elif isinstance(module, weight_modules):
                    decay.add(full_name)
                elif isinstance(module, no_decay_modules):
                    no_decay.add(full_name)
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad} 
        assert not decay & no_decay
        assert parameters.keys() == decay | no_decay
        
        optim_groups = [
            {"params": [parameters[name] for name in sorted(decay)],
             "weight_decay": self.weight_decay},
            {"params": [parameters[name] for name in sorted(no_decay)], 
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max
    
        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        return self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), return_details=True,
                    reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))

    