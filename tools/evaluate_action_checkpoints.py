"""Compare saved action checkpoints with one paired Lightning validation pass."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.nn import functional as F

from datamodules import NuplanDataModule
from datasets.dataset_utils import FRAME_INTERVAL_US, local_actions
from metrics.action import sample_metrics
from metrics.action_diagnostics import history_action
from metrics.statistics import aggregate
from model.FRESHDrive import FRESHDrive
from evaluate_action_diagnostics import select_samples, stage_inputs
from utils import load_config


def load_checkpoints(paths):
    """Run identical state dictionaries once, retaining all file identities."""
    models, metadata, fingerprints = [], [], {}
    for name in paths:
        checkpoint = torch.load(name, map_location="cpu", weights_only=False)
        digest = hashlib.sha256()
        for key, value in sorted(checkpoint["state_dict"].items()):
            value = value.detach().cpu().contiguous()
            digest.update(str((key, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        fingerprint = digest.hexdigest()
        if fingerprint not in fingerprints:
            model = FRESHDrive.load_from_checkpoint(name, map_location="cpu", strict=True, action_loss="l1")
            if model.mode != "action" or model.future_action < 5 or model.plan_horizon != 5:
                raise ValueError("Comparison requires action checkpoints with a five-step candidate")
            fingerprints[fingerprint] = len(models)
            models.append(model)
        metadata.append({"checkpoint": name, "epoch": checkpoint["epoch"],
                         "global_step": checkpoint["global_step"], "state_sha256": fingerprint,
                         "model_index": fingerprints[fingerprint],
                         "hyperparameters": checkpoint["hyper_parameters"],
                         "source_loss": checkpoint.get("action_loss_config", checkpoint["hyper_parameters"])})
    dimensions = {(m.history_features, m.future_action) for m in models}
    weights = {(m.action_loss.xy_weight, m.action_loss.yaw_weight,
                m.action_loss.trajectory_weight, m.action_loss.heading_weight) for m in models}
    if len(dimensions) != 1 or len(weights) != 1:
        raise ValueError("Paired comparison requires identical horizons and loss weights")
    return models, metadata


class CheckpointEvaluator(pl.LightningModule):
    def __init__(self, models, metadata, dataset, config, manifest):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.metadata, self.dataset, self.config, self.manifest = metadata, dataset, config, manifest
        self.rows = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        names = ("bev_features", "bev_valid_mask", "timestamps_us", "anchor_timestamp_us",
                 "reference_ego_to_global", "bev_range")
        self.inputs = move_data_to_device({key: batch[key] for key in names}, device)
        return batch

    def on_validation_epoch_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("Paired report requires one device")
        self.started = time.perf_counter()

    def append(self, scenario, actions, batch, valid, loss_fn, *, historical_loss=None):
        horizon = actions.shape[1]
        expert = batch["future_trajectory"][:, :horizon].float()
        valid = valid[:, :horizon]
        scores = sample_metrics(actions, expert, valid, loss_fn, FRAME_INTERVAL_US / 1e6)
        if historical_loss is not None:
            scores["historical_smooth_l1_action_loss"] = historical_loss
        target = local_actions(torch.nan_to_num(expert))
        difference = actions - target
        difference[..., 2] = torch.atan2(difference[..., 2].sin(), difference[..., 2].cos())
        for dimension, key in enumerate(("dx_bias_m", "dy_bias_m", "dyaw_bias_rad")):
            scores[key] = difference[..., dimension].masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
        scores = {key: value.tolist() for key, value in scores.items()}
        for i in range(len(actions)):
            identity = (batch["recording"][i], int(batch["anchor_timestamp_us"][i]))
            row = dict(scenario=scenario, recording=identity[0], anchor_timestamp_us=identity[1],
                       event_types="|".join(sorted(self.manifest[identity])),
                       manifest_multiplicity=sum(self.manifest[identity].values()),
                       valid_actions=int(valid[i].sum()), horizon_s=horizon * .1)
            row.update({key: values[i] if math.isfinite(values[i]) else None for key, values in scores.items()})
            self.rows.append(row)

    def validation_step(self, batch, batch_idx):
        h, future = self.models[0].history_features, self.models[0].future_action
        if self.inputs["bev_features"].shape[1] != h + future + 1:
            raise ValueError("Cache horizon differs from checkpoint")
        if not (batch["anchor_index"] == h).all() or not (batch["timestamps_us"].diff(dim=1) == FRAME_INTERVAL_US).all():
            raise ValueError("Expected matching anchors and uniformly sampled 10 Hz windows")
        valid = batch["future_action_mask"].bool()
        expert = batch["future_trajectory"].float()
        target = local_actions(torch.nan_to_num(expert))
        for index, model in enumerate(self.models):
            result = model(**self.inputs, teacher_forcing=False, return_details=True)
            actions = result["actions"].float().cpu()
            # Reconstruct the original beta=1 SmoothL1 monitor for legacy runs.
            source = next(item["source_loss"] for item in self.metadata if item["model_index"] == index)
            historical_loss = None
            if source.get("action_loss") == "smooth_l1":
                yaw = actions[..., 2] - target[..., 2]
                yaw = torch.atan2(yaw.sin(), yaw.cos())
                errors = F.smooth_l1_loss(actions[..., :2], target[..., :2], reduction="none").mean(-1)
                errors += source.get("yaw_weight", 1.) * F.smooth_l1_loss(yaw, torch.zeros_like(yaw), reduction="none")
                historical_loss = errors.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
            self.append(f"model_{index}/replay", actions, batch, valid, model.action_loss,
                        historical_loss=historical_loss)
            self.append(f"model_{index}/next", actions[:, :1], batch, valid, model.action_loss)
            self.append(f"model_{index}/fixed5", result["fixed_actions"].float().cpu(), batch, valid, model.action_loss)
        historical, available = history_action(batch, h)
        baseline = historical[:, None].expand(-1, future, -1)
        self.append("constant_action/replay", baseline, batch, valid & available[:, None], self.models[0].action_loss)
        self.append("constant_action/fixed5", baseline[:, :5], batch, valid & available[:, None], self.models[0].action_loss)
        if batch_idx % 10 == 0 or batch_idx + 1 == len(self.trainer.val_dataloaders):
            print(f"Checkpoint comparison: {batch_idx + 1}/{len(self.trainer.val_dataloaders)} batches; "
                  f"{time.perf_counter() - self.started:.1f} s", flush=True)

    def on_validation_epoch_end(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["scenario"]].append(row)
        summaries = {}
        identifiers = ("scenario", "recording", "anchor_timestamp_us", "event_types", "manifest_multiplicity",
                       "valid_actions", "horizon_s")
        for scenario, rows in groups.items():
            keys = [key for key in rows[0] if key not in identifiers]
            summary = aggregate(rows, keys)
            expanded = [row for row in rows for _ in range(row["manifest_multiplicity"])]
            summary["manifest_window_weighted"] = aggregate(expanded, keys)
            summary["by_event_type"] = {
                label: aggregate([row for row in rows if label in row["event_types"].split("|")], keys)
                for label in sorted({label for row in rows for label in row["event_types"].split("|")})}
            summaries[scenario] = summary
        report = dict(checkpoints=self.metadata, scenarios=summaries, config=self.config,
                      protocol=dict(closed_loop=False, teacher_forcing=False,
                                    replay="10 causal decisions, each sees newly arrived logged BEV; 1.0 s total",
                                    next="First action at the anchor; 0.1 s",
                                    fixed5="Anchor memory only, zero prefix, reset decoder hidden, five calls; 0.5 s",
                                    aggregation="Unique (recording,timestamp) anchors; all original event labels retained",
                                    manifest_window_weighted="Duplicated label windows counted as in training validation",
                                    historical_loss="Original beta=1 SmoothL1 action formula, separate from current L1",
                                    missing="Excluded, never zero; FDE requires complete horizon",
                                    constant_action="Repeat the last historical local action; no future state inputs",
                                    statistical_inference="Descriptive paired comparison, no independent-log CI"),
                      device=str(self.device), torch_version=str(torch.__version__),
                      elapsed_s=time.perf_counter()-self.started)
        output = Path(self.config["evaluation"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        keys = list(dict.fromkeys(key for row in self.rows for key in row))
        with (output / "per_sample.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        comparison = []
        for item in self.metadata:
            index = item["model_index"]
            replay = summaries[f"model_{index}/replay"]["metrics"]
            fixed = summaries[f"model_{index}/fixed5"]["metrics"]
            first = summaries[f"model_{index}/next"]["metrics"]
            comparison.append(dict(checkpoint=item["checkpoint"], epoch=item["epoch"], model_index=index,
                                   unique_anchors=summaries[f"model_{index}/replay"]["samples"],
                                   action_l1=replay["action_loss"],
                                   historical_smooth_l1=replay.get("historical_smooth_l1_action_loss"),
                                   ade_m=replay["ade_m"], fde_m=replay["fde_m"],
                                   fixed5_ade_m=fixed["ade_m"], fixed5_fde_m=fixed["fde_m"],
                                   next_action_l1=first["action_loss"], next_l2_m=first["ade_m"]))
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        print(f"Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024, workers=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/val/action_checkpoints.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    models, metadata = load_checkpoints(config["checkpoints"])
    data = NuplanDataModule(**config["datamodule"])
    data.setup("validate")
    dataset = data.val_dataset
    manifest = defaultdict(Counter)
    for path in dataset.samples:
        manifest[(path.parent.parent.name, int(path.stem))][path.parent.name] += 1
    indices = select_samples(dataset, None)
    indices = indices[:config["evaluation"]["max_samples"]]
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset._processed_file_names = [str(p.relative_to(dataset.event_dir)) for p in dataset.samples]
    print(f"Comparing {len(metadata)} files / {len(models)} unique states on {len(dataset)} anchors", flush=True)
    if config["evaluation"]["scratch_dir"]:
        stage_inputs(dataset, config["evaluation"]["scratch_dir"])
    evaluator = CheckpointEvaluator(models, metadata, dataset, config, manifest)
    pl.Trainer(**config["trainer"]).validate(evaluator, dataloaders=data.val_dataloader(), verbose=False)
