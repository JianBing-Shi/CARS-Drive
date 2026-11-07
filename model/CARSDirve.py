"""Lightning entry point for causal action prediction and evidence refresh."""
from __future__ import annotations

import math
import warnings
from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules import TemporalBEVContext
from modules import BEVTokenizer 
from modules import BEVMemory
from modules import ActionDecoder
from modules import ActionEncoder
from modules import RollingTrajectoryPlanner
from modules import TrajectoryL1Loss
from modules import CARSRevisionAdapter, CARSRevisionPlanner

class CARSDirve(pl.LightningModule):
    """Causal BEV model with pred and plan training modes."""

    def __init__(self, history_step: int = 20, future_step: int = 10, feature_dim: int = 256, hidden_dim: int = 256,
                 spatial_grid: tuple[int, int] = (9, 9), macro_grid: tuple[int, int] = (3, 3), heads: int = 8, layers: int = 2,
                 dropout: float = .1, mode: str = "pred", refresh_budget: float = 1., max_refresh: int = 1, cost_weight: float = 0.,
                 utility_weight: float = 1., action_loss: str = "l1", yaw_weight: float = 1., trajectory_loss_weight: float = 0.,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4, warmup_epochs: int = 4, T_max: int = 32,
                 lr_min_ratio: float = .05,
                 anchor_dropout: float = .2, cameras: int | None = None,
                 plan_horizon: int = 5, refresh_steps: int = 10, refresh_offset: int = -5,
                 architecture_version: int = 1, decoder_layers: int = 2,
                 teacher_forcing_ratio: float = 1., heading_loss_weight: float = 0.,
                 xy_loss_weight: float = 1., refresh_version: int = 3,
                 refresh_observation: str = "current_full_bev", refresh_update: str = "full_valid_bev",
                 refresh_probability_threshold: float = .5, refresh_label_epsilon: float = 0.,
                 refresh_update_cost: float = 1., refresh_fixed_interval: int = 3,
                 refresh_random_probability: float = .3,
                 refresh_monitor_frames: int = 3, refresh_residual_scale: float = .1,
                 refresh_trajectory_weight: float = 1., refresh_classification_weight: float = 1.,
                 refresh_history_weight: float = .25, refresh_heading_weight: float = .1) -> None:
        super().__init__()
        if mode not in {"pred", "plan"}:
            raise ValueError("invalid mode")
        if action_loss != "l1":
            raise ValueError("action_loss is fixed to l1")
        if refresh_version != 3:
            raise ValueError("CARS requires refresh_version=3")
        if refresh_version == 3:
            revision_weights = (refresh_trajectory_weight, refresh_classification_weight,
                                refresh_history_weight, refresh_heading_weight)
            if (not isinstance(refresh_monitor_frames, int) or refresh_monitor_frames < 1
                    or not math.isfinite(refresh_residual_scale) or refresh_residual_scale <= 0
                    or any(not math.isfinite(w) or w < 0 for w in revision_weights)
                    or refresh_trajectory_weight <= 0 or refresh_classification_weight <= 0):
                raise ValueError("Invalid trajectory revision supervision or monitoring settings")
        if refresh_observation != "current_full_bev" or refresh_update != "full_valid_bev":
            raise ValueError("Binary gate requires current_full_bev monitoring and full_valid_bev updates")
        if (not 0 <= refresh_probability_threshold <= 1 or not 0 <= refresh_random_probability <= 1
                or refresh_label_epsilon < 0 or refresh_update_cost <= 0 or refresh_fixed_interval < 1):
            raise ValueError("Invalid binary refresh settings")
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must lie in [0, 1]")
        if max_refresh != 1:
            raise ValueError("first architecture supports an empty set or one region")
        if not 0 <= anchor_dropout < 1:
            raise ValueError("anchor_dropout must be in [0, 1)")
        if not isinstance(warmup_epochs, int) or not isinstance(T_max, int) or warmup_epochs < 0 or T_max <= 0 or warmup_epochs >= T_max:
            raise ValueError("require 0 <= warmup_epochs < T_max")
        if not 0 < lr_min_ratio <= 1:
            raise ValueError("lr_min_ratio must lie in (0, 1]")
        self.save_hyperparameters()
        # Marks coordinate-L1 trajectory supervision in new checkpoints.
        self.hparams["action_loss_version"] = 2
        self.cars_adapter_version = refresh_version

        # 数据窗口与网络配置
        self.mode = mode
        self.history_step = history_step
        self.future_step = future_step
        # Keep the public window names used by evaluators and legacy
        # checkpoints.  The model constructor historically exposed these as
        # ``history_features``/``future_action``; diagnostics and evaluators
        # still consume that interface while the implementation uses the
        # clearer ``*_step`` names internally.
        self.history_features = history_step
        self.future_action = future_step
        self.hidden_dim = hidden_dim
        self.grid = tuple(spatial_grid)
        self.regions = macro_grid[0] * macro_grid[1]

        # 优化器与损失配置
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.lr_min_ratio = lr_min_ratio
        self.trajectory_loss_weight = trajectory_loss_weight
        self.utility_weight = utility_weight
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # 视觉更新配置
        self.cars_adapter_budget = refresh_budget
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
            architecture_version=architecture_version,
        )
        self.memory = BEVMemory(hidden_dim=hidden_dim)
        # Encode the causal action history before decoding the next action.
        self.action_encoder = ActionEncoder(hidden_dim)
        self.decoder = ActionDecoder(
            hidden_dim=hidden_dim,
            future_step=future_step,
            heads=heads,
            architecture_version=architecture_version,
            layers=decoder_layers,
            dropout=dropout,
        )
        # 时序状态与证据刷新
        self.temporal = TemporalBEVContext(
            history_step=history_step,
            hidden_dim=hidden_dim,
            grid=self.grid,
        )
        self.plan_horizon = plan_horizon
        self.cars_adapter_steps = refresh_steps
        self.cars_adapter_offset = refresh_offset
        self.cars_planner = RollingTrajectoryPlanner(plan_horizon)
        self.cars_adapter = nn.Identity()
        if mode == 'plan':
            self.cars_planner = CARSRevisionPlanner(plan_horizon, refresh_probability_threshold,
                refresh_label_epsilon, refresh_update_cost, refresh_fixed_interval,
                refresh_random_probability, refresh_monitor_frames)
            self.cars_adapter = CARSRevisionAdapter(hidden_dim, self.regions, plan_horizon,
                                                   heads=heads, dropout=0., residual_scale=refresh_residual_scale)

        # 动作损失与轨迹评价
        self.action_loss = TrajectoryL1Loss(
            action_loss=action_loss,
            yaw_weight=yaw_weight,
            trajectory_weight=trajectory_loss_weight,
            heading_weight=heading_loss_weight,
            xy_weight=xy_loss_weight,
        )

        # 在优化器创建前确定可训练参数
        if self.mode == "pred":
            for p in self.cars_adapter.parameters():
                p.requires_grad = False
        elif self.mode == "plan":
            self.freeze_pred_model()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Lightning applies load overrides to hyper_parameters before its load hook.
        # Keep the training objective separately so overrides cannot conceal it.
        keys = ("action_loss_version", "action_loss", "xy_loss_weight", "yaw_weight",
                "trajectory_loss_weight", "heading_loss_weight")
        checkpoint["action_loss_config"] = {key: self.hparams[key] for key in keys}
        checkpoint["refresh_protocol"] = self.cars_adapter_protocol()

    def cars_adapter_protocol(self) -> dict[str, Any]:
        keys = ("refresh_version", "refresh_observation", "refresh_update", "plan_horizon",
                "refresh_steps", "refresh_offset", "refresh_budget", "cost_weight",
                "refresh_probability_threshold", "refresh_label_epsilon", "refresh_update_cost",
                "refresh_fixed_interval", "refresh_random_probability")
        if self.cars_adapter_version == 3:
            keys += ("refresh_monitor_frames", "refresh_residual_scale", "refresh_trajectory_weight",
                     "refresh_classification_weight", "refresh_history_weight", "refresh_heading_weight")
        return {key: self.hparams[key] for key in keys}

    def revision_metrics(self, result, target, mask):
        from metrics.trajectory_revision import revision_metrics
        return revision_metrics(result, target, mask, self.hparams.refresh_probability_threshold,
            trajectory_weight=self.hparams.refresh_trajectory_weight,
            classification_weight=self.hparams.refresh_classification_weight,
            history_weight=self.hparams.refresh_history_weight,
            heading_weight=self.hparams.refresh_heading_weight)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.checkpoint_refresh_protocol = checkpoint.get("refresh_protocol")
        source_version = checkpoint.get("refresh_protocol", checkpoint.get("hyper_parameters", {})).get("refresh_version", 1)
        if source_version != self.cars_adapter_version:
            raise ValueError("Refresh architecture differs from checkpoint; initialize binary gate from action weights")
        if (self.cars_adapter_version == 3 and self.checkpoint_refresh_protocol is not None
                and self.checkpoint_refresh_protocol != self.cars_adapter_protocol()):
            raise ValueError('Trajectory revision protocol differs from checkpoint; preserve adapter and supervision settings')
        source = checkpoint.get("action_loss_config", checkpoint.get("hyper_parameters", {}))
        defaults = {"action_loss_version": 1, "action_loss": "smooth_l1",
                    "xy_loss_weight": 1., "yaw_weight": 1.,
                    "trajectory_loss_weight": 0., "heading_loss_weight": 0.}
        self.checkpoint_loss_config = {key: source.get(key, default) for key, default in defaults.items()}
        if any(value != self.hparams[key] for key, value in self.checkpoint_loss_config.items()):
            warnings.warn("Checkpoint loss settings differ: weights are loaded, but reported losses use "
                          "the current fixed L1 objective. Historical loss values are not directly comparable.",
                          UserWarning, stacklevel=2)

    def forward(self, bev_features: Tensor, bev_valid_mask: Tensor | None = None, timestamps_us: Tensor | None = None,
                anchor_timestamp_us: Tensor | None = None, future_actions: Tensor | None = None, bev_age_s: Tensor | None = None,
                history_actions: Tensor | None = None,
                *, budget: Tensor | float | None = None, region_costs: Tensor | None = None, return_details: bool = False,
                teacher_forcing: bool | float = False, reference_ego_to_global: Tensor | None = None,
                bev_range: Tensor | None = None, planning_targets: Tensor | None = None,
                planning_mask: Tensor | None = None) -> Tensor | dict[str, Tensor]:
        if self.mode not in {"pred", "plan"}:
            raise RuntimeError(f"Unsupported mode: {self.mode}")
        if self.mode == "plan":
            self.freeze_pred_model()
            batch = bev_features.shape[0]
            if history_actions is None:
                history_actions = bev_features.new_zeros(batch, self.history_step, 3)
            history_context = self.action_encoder(history_actions[:, -self.history_step:])
            valid = torch.ones_like(bev_features[:, :, 0], dtype=torch.bool) if bev_valid_mask is None else bev_valid_mask.bool()
            times = timestamps_us.double() if timestamps_us is not None else torch.arange(bev_features.shape[1], device=bev_features.device)[None].expand(bev_features.shape[0], -1).double() * 100_000
            if times.ndim == 3:
                times = times.amax(-1)
            result = self.cars_planner(
                bev_features, valid, times, reference_ego_to_global, bev_range,
                start=self.history_step + self.cars_adapter_offset, steps=self.cars_adapter_steps,
                tokenizer=self.tokenizer, memory=self.memory, decoder=self.decoder,
                refresh=self.cars_adapter, temporal=self.temporal, region_ids=self.region_ids,
                budget=self.cars_adapter_budget, cost_weight=self.cost_weight,
                targets=planning_targets, target_mask=planning_mask,
                history_context=history_context)
            return result if return_details else result["planned_trajectory"]

        if bev_features.ndim != 5 or bev_features.shape[1] < self.history_step + self.future_step:
            raise ValueError("Sequential replay requires history_step + future_step BEV frames")
        batch, steps = bev_features.shape[:2]
        if history_actions is None:
            # Inference and legacy callers may not have an ego-motion history
            # available.  Use a zero causal prefix so the visual-only path
            # remains well-defined; training still supplies annotated history
            # actions explicitly.
            history_actions = bev_features.new_zeros(batch, self.history_step, 3)
        history_context = self.action_encoder(history_actions[:, -self.history_step:])
        valid = torch.ones(batch, steps, bev_features.shape[-2], bev_features.shape[-1], dtype=torch.bool, device=bev_features.device) if bev_valid_mask is None else bev_valid_mask.bool()
        ages = self.temporal.ages(bev_features, timestamps_us, anchor_timestamp_us, bev_age_s)
        if reference_ego_to_global is None:
            state = self.temporal.encode_history(bev_features[:, :self.history_step], valid[:, :self.history_step], ages[:, :self.history_step], self.tokenizer, self.memory)
        else:
            state = self.temporal.encode_history(bev_features[:, :1], valid[:, :1], ages[:, :1], self.tokenizer, self.memory)
            for frame in range(1, self.history_step):
                state = self.temporal.align_memory(state, reference_ego_to_global[:, frame - 1], reference_ego_to_global[:, frame], bev_range)
                tokens, frame_valid = self.tokenizer(bev_features[:, frame], ages[:, frame], valid[:, frame])
                state = {"memory": self.memory(state["memory"], tokens, frame_valid),
                         "valid": state["valid"] | frame_valid,
                         "age_s": torch.where(frame_valid, ages[:, frame, None], state["age_s"])}
            state["reference_pose"] = reference_ego_to_global[:, self.history_step - 1]
        previous = bev_features.new_zeros(batch, 3, dtype=torch.float32)
        hidden, outputs = None, []
        fixed_actions = None
        for step in range(self.future_step):
            frame = self.history_step + step
            # Ages are measured relative to the moving decision time.
            elapsed = ages[:, frame - 1] - ages[:, frame] if step else ages.new_zeros(batch)
            output = self.observe_step(bev_features[:, frame], valid[:, frame], state,
                                       previous, hidden, step, elapsed, ages[:, frame].clamp_min(0),
                                       budget, region_costs,
                                       None if reference_ego_to_global is None else reference_ego_to_global[:, frame], bev_range,
                                       history_context)
            state, hidden = output["state"], output["hidden"]
            if step == 0 and return_details and not self.training:
                # Candidate acceptance check only: five calls to the same one-step head.
                # Use the same zero prefix / fresh hidden convention as RollingTrajectoryPlanner.
                fixed_actions, _ = self.cars_planner.decode(state, previous, self.decoder,
                                                       min(self.plan_horizon, self.future_step), history_context)
            previous = output["actions"]
            if teacher_forcing and future_actions is not None:
                ratio = float(teacher_forcing)
                if not 0 <= ratio <= 1:
                    raise ValueError("teacher_forcing must lie in [0, 1]")
                expert_previous = future_actions[:, step]
                usable = torch.isfinite(expert_previous).all(-1)
                choose = usable if ratio == 1 else usable & (torch.rand(batch, device=previous.device) < ratio)
                previous = torch.where(choose[:, None], torch.nan_to_num(expert_previous), previous)
            outputs.append(output)
        keys = ("actions", "proposal_actions", "queries", "refresh_mask", "refresh_utility")
        result = {key: torch.stack([output[key] for output in outputs], 1) for key in keys}
        if fixed_actions is not None:
            result["fixed_actions"] = fixed_actions
        return result if return_details else result["actions"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        if self.mode == "pred":
            # Expert poses use the anchor coordinate frame; actions use each step's ego frame.
            expert = batch["future_trajectory"].float()
            poses = F.pad(expert, (0, 0, 1, 0))
            delta = poses[:, 1:] - poses[:, :-1]
            yaw = poses[:, :-1, 2]
            target = torch.stack((yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                                  -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                                  torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())), -1)
            result = self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), target, return_details=True, teacher_forcing=self.teacher_forcing_ratio,
                          history_actions=batch["history_actions"],
                          reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))
            valid = batch["future_action_mask"].bool()
            losses = self.action_loss(result["actions"], target, expert, valid)
            loss = losses["loss"]
            self.log("train/trajectory_loss", losses["trajectory_loss"], batch_size=target.shape[0])
            self.log("train/loss", loss, prog_bar=True, batch_size=target.shape[0])
            self.log("train/action_loss", losses["action_loss"], batch_size=target.shape[0])
            self.log("train/heading_loss", losses["heading_loss"], batch_size=target.shape[0])
            self.log("train/ade", losses["ade"], batch_size=target.shape[0])
            for key in ("xy_loss", "yaw_loss", "weighted_xy_loss", "weighted_yaw_loss",
                        "weighted_trajectory_loss", "weighted_heading_loss"):
                self.log(f"train/{key}", losses[key], batch_size=target.shape[0])
            return loss

        elif self.mode == "plan":
            result = self(
                batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"),
                history_actions=batch.get("history_actions"),
                reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"),
                planning_targets=batch["planning_targets"], planning_mask=batch["planning_mask"], return_details=True)
            if self.cars_adapter_version == 3:
                metrics = self.revision_metrics(result, batch['planning_targets'], batch['planning_mask'])
                self.log_dict({f'train/{key}': value for key, value in metrics.items()},
                              batch_size=batch['planning_mask'].shape[0])
                return metrics['loss']
            raise RuntimeError("CARS v3 refresh metrics were not produced")

    def on_train_start(self) -> None:
        """Prepare the fixed action predictor before refresh training starts."""
        if self.mode == "pred":
            return
        elif self.mode == "plan":
            self.freeze_pred_model()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        if self.mode == "pred":
            # Expert poses use the anchor coordinate frame; actions use each step's ego frame.
            expert = batch["future_trajectory"].float()
            poses = F.pad(expert, (0, 0, 1, 0))
            delta = poses[:, 1:] - poses[:, :-1]
            yaw = poses[:, :-1, 2]
            target = torch.stack((yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                                  -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                                  torch.atan2(delta[..., 2].sin(), delta[..., 2].cos())), -1)
            result = self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), target, return_details=True, teacher_forcing=False,
                          history_actions=batch["history_actions"],
                          reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))
            valid = batch["future_action_mask"].bool()
            losses = self.action_loss(result["actions"], target, expert, valid)
            loss = losses["loss"]
            self.log("val/trajectory_loss", losses["trajectory_loss"], batch_size=target.shape[0])
            self.log("val/loss", loss, prog_bar=True, batch_size=target.shape[0])
            self.log("val/action_loss", losses["action_loss"], batch_size=target.shape[0])
            self.log("val/heading_loss", losses["heading_loss"], batch_size=target.shape[0])
            self.log("val/ade", losses["ade"], batch_size=target.shape[0])
            complete = valid.all(-1)
            if complete.any():
                final_error = (TrajectoryL1Loss.trajectory(result["actions"])[complete, -1, :2]
                               - expert[complete, -1, :2]).norm(dim=-1)
                self.log("val/fde", final_error.mean(), batch_size=int(complete.sum()))
            for key in ("xy_loss", "yaw_loss", "weighted_xy_loss", "weighted_yaw_loss",
                        "weighted_trajectory_loss", "weighted_heading_loss"):
                self.log(f"val/{key}", losses[key], batch_size=target.shape[0])
            first = self.action_loss(result["actions"][:, :1], target[:, :1], expert[:, :1], valid[:, :1])
            self.log("val/next_action_loss", first["action_loss"], batch_size=target.shape[0])
            if "fixed_actions" in result:
                actions = result["fixed_actions"]
                horizon = actions.shape[1]
                prefix = valid[:, :horizon].long().cumprod(-1).bool()
                distance = (TrajectoryL1Loss.trajectory(actions)[..., :2] - expert[:, :horizon, :2]).norm(dim=-1)
                available = prefix.any(-1)
                ade = distance.masked_fill(~prefix, 0).sum(-1) / prefix.sum(-1).clamp_min(1)
                if available.any():
                    self.log("val/fixed5_ade", ade[available].mean(), batch_size=int(available.sum()))
                complete = prefix.all(-1)
                if complete.any():
                    self.log("val/fixed5_fde", distance[complete, -1].mean(), batch_size=int(complete.sum()))
            return loss

        elif self.mode == "plan":
            metrics = self.cars_adapter_validation_metrics(batch)
            self.log_dict(metrics, on_step=False, on_epoch=True,
                          batch_size=batch["planning_mask"].shape[0])
            return metrics["val/loss"]

    def refresh_validation_metrics(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Shared refresh scores for training validation and offline evaluation."""
        result = self(
            batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"),
            history_actions=batch.get("history_actions"),
            reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"),
            planning_targets=batch["planning_targets"], planning_mask=batch["planning_mask"], return_details=True)
        if self.cars_adapter_version == 3:
            return {f'val/{key}': value for key, value in self.revision_metrics(
                result, batch['planning_targets'], batch['planning_mask']).items()}
        raise RuntimeError("CARS v3 refresh metrics were not produced")

    # Backward-compatible name used by Lightning validation and the offline
    # RefreshEvaluator.  Keep one implementation so both paths report the
    # same reductions.
    def cars_adapter_validation_metrics(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        return self.refresh_validation_metrics(batch)

    def freeze_pred_model(self) -> None:
        self.tokenizer.eval()
        self.memory.eval()
        self.action_encoder.eval()
        self.decoder.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        for p in self.memory.parameters():
            p.requires_grad = False
        for p in self.action_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def observe_step(self, feature: Tensor, valid: Tensor, state: dict[str, Tensor],
                     previous: Tensor, hidden: Tensor | None = None, step: int = 0,
                     elapsed_s: Tensor | float = .1, age_s: Tensor | None = None,
                     budget: Tensor | float | None = None,
                     region_costs: Tensor | None = None,
                     reference_pose: Tensor | None = None, bev_range: Tensor | None = None,
                     history_context: Tensor | None = None) -> dict[str, Any]:
        """Consume one arrived observation; return one action and reusable state."""
        batch = feature.shape[0]
        if reference_pose is not None:
            state = self.temporal.align_memory(state, state["reference_pose"], reference_pose, bev_range)
        elapsed = torch.as_tensor(elapsed_s, device=feature.device).reshape(-1)
        state = {**state, "age_s": state["age_s"] + elapsed[:, None]}
        age = feature.new_zeros(batch, dtype=torch.float32) if age_s is None else age_s
        if self.mode == "pred":
            frame_valid = valid.clone()
            if self.training and self.anchor_dropout:
                keep = torch.rand(batch, device=feature.device) >= self.anchor_dropout
                frame_valid &= keep[:, None, None]
            tokens, token_valid = self.tokenizer(feature.float(), age, frame_valid)
            updated_valid = token_valid
            updated = {"memory": self.memory(state["memory"], tokens, updated_valid),
                       "valid": state["valid"] | updated_valid,
                       "age_s": torch.where(updated_valid, age[:, None], state["age_s"])}
            action, query = self.decoder(updated["memory"], updated["valid"], previous, hidden, step, history_context)
            if reference_pose is not None:
                updated["reference_pose"] = reference_pose
            return {"actions": action, "proposal_actions": action, "queries": query,
                    "refresh_mask": torch.zeros(batch, self.regions, dtype=torch.bool, device=feature.device),
                    "refresh_utility": feature.new_zeros(batch, self.regions),
                    "state": updated, "hidden": query}
        elif self.mode == "plan":
            raise ValueError("Plan uses RollingTrajectoryPlanner to retain and update five-step plans")

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
            floor = self.lr_min_ratio
            if epoch >= T_max:
                return floor
            if warmup_epochs == 0:
                cosine = .5 * (1. + math.cos(math.pi * epoch / T_max))
                return floor + (1. - floor) * cosine
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))
            return floor + (1. - floor) * cosine

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        return self(batch["bev_features"], batch.get("bev_valid_mask"), batch.get("timestamps_us"), batch.get("anchor_timestamp_us"), return_details=True,
                    reference_ego_to_global=batch.get("reference_ego_to_global"), bev_range=batch.get("bev_range"))
