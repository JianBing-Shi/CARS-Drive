"""Lightning validation wrappers and report export for both model modes.

The CLI uses a single device so exact quantiles, anchor deduplication and
log-cluster bootstrap operate on the complete validation set.
"""
import csv
import json
import math
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device

from datasets.dataset_utils import FRAME_INTERVAL_US, local_actions
from modules import TrajectoryL1Loss
from metrics.action import sample_metrics
from metrics.collision import EgoFootprint, collision_metrics
from metrics.statistics import aggregate, paired_comparison


class Evaluator(pl.LightningModule):
    """Shared checkpoint context and JSON export for offline validation."""

    def __init__(self, model, dataset, config, config_path, count):
        super().__init__()
        self.model, self.dataset = model, dataset
        self.config, self.config_path, self.count = config, Path(config_path), count
        self.seed = torch.initial_seed()
        self.options = config["evaluation"]

    def on_validation_epoch_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("Report export requires single-device validation")

    def export(self, report):
        self.summary = report
        output = Path(self.options["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Results: {output.resolve()}")
        return output


class ActionEvaluator(Evaluator):
    """Detailed action metrics, baselines and per-window predictions."""

    def __init__(self, model, dataset, config, config_path, count):
        super().__init__(model, dataset, config, config_path, count)
        options = self.options = config["evaluation"]
        if min(options[k] for k in ("warmup_batches", "max_annotation_age_us", "bootstrap_samples")) < 0:
            raise ValueError("Warmup, annotation tolerance and bootstrap counts must be nonnegative")
        self.footprint = EgoFootprint(*options["ego_footprint"]) if options["ego_footprint"] else None
        self.log_map = json.loads(Path(options["log_map"]).read_text()) if options["log_map"] else None
        if self.log_map is not None and (not isinstance(self.log_map, dict) or not all(
                isinstance(k, str) and isinstance(v, str) and v for k, v in self.log_map.items())):
            raise ValueError("evaluation.log_map must map recording names to nonempty original log strings")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Labels and variable-size annotations stay on CPU; only inputs move.
        names = ("bev_features", "bev_valid_mask", "timestamps_us", "anchor_timestamp_us",
                 "reference_ego_to_global", "bev_range")
        self.inputs = move_data_to_device({k: batch[k] for k in names if k in batch}, device)
        return batch

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.rows, self.predictions, self.model_times = [], [], []
        self.warmed_up = False
        self.started = time.perf_counter()

    def validation_step(self, batch, batch_idx):
        model, options, dataset = self.model, self.options, self.dataset
        device, interval_s = self.device, FRAME_INTERVAL_US / 1e6
        footprint, log_map = self.footprint, self.log_map
        rows, predictions = self.rows, self.predictions
        expected_frames = model.history_features + model.future_action + 1
        if batch["bev_features"].shape[1] != expected_frames:
            raise ValueError(f"Expected {expected_frames} cached frames per window")
        if not (batch["anchor_index"] == model.history_features).all():
            raise ValueError("Cache anchor_index disagrees with checkpoint history_features")
        if not (batch["timestamps_us"].diff(dim=1) == FRAME_INTERVAL_US).all():
            raise ValueError("Expected a uniformly sampled 10 Hz window")
        expert = batch["future_trajectory"].float()
        valid = batch["future_action_mask"].bool()
        reconstructed = TrajectoryL1Loss.trajectory(local_actions(torch.nan_to_num(expert)))
        prefix = valid.long().cumprod(-1).bool()
        if not torch.allclose(reconstructed[..., :2][prefix], expert[..., :2][prefix], atol=1e-4, rtol=1e-4):
            raise ValueError("Local action labels fail SE(2) reconstruction")
        inputs = self.inputs
        if not self.warmed_up:
            for _ in range(options["warmup_batches"]):
                model(**inputs, teacher_forcing=False)
            self.warmed_up = True
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        actions = model(**inputs, teacher_forcing=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.model_times.append(time.perf_counter() - begin)
        actions = actions.float().cpu()
        if actions.shape != expert.shape:
            raise ValueError("Checkpoint output horizon/shape differs from expert targets")
        metrics = {"model_" + key: value for key, value in sample_metrics(
            actions, expert, valid, model.action_loss, interval_s).items()}
        # Baseline reads historical motion only, never future actions.
        history = batch["trajectory"]["past_trajectory"].float()
        anchor = batch["trajectory"]["anchor_trajectory"].float()
        last_action = local_actions(torch.cat((history, anchor[:, None]), dim=1))[:, -1:]
        baseline = last_action.expand(-1, model.future_action, -1)
        pose_mask = batch["trajectory"]["trajectory_mask"].bool()
        available = pose_mask[:, model.history_features - 1:model.history_features + 1].all(-1)
        baseline = torch.where(available[:, None, None], baseline, torch.zeros_like(baseline))
        metrics.update({"constant_action_" + key: value for key, value in sample_metrics(
            baseline, expert, valid & available[:, None], model.action_loss, interval_s).items()})
        horizon = model.future_action
        annotations = batch.get("tracked_objects")
        future_annotations = ([item[model.history_features + 1:model.history_features + 1 + horizon]
                               for item in annotations] if annotations is not None else None)
        # Targets are expressed in the trajectory annotation's anchor frame,
        # which need not exactly match the BEV query's reference pose.
        anchor_poses = batch["trajectory"].get("reference_ego_to_global")
        target_times = batch["timestamps_us"][:, model.history_features + 1:]
        collision_steps = {}
        for name, candidate, candidate_prefix in (
                ("model", TrajectoryL1Loss.trajectory(actions), prefix),
                ("constant_action", TrajectoryL1Loss.trajectory(baseline), prefix & available[:, None]),
                ("expert", expert, prefix)):
            scores, step_collisions = collision_metrics(
                candidate, candidate_prefix, future_annotations, anchor_poses, target_times,
                footprint=footprint, max_annotation_age_us=options["max_annotation_age_us"])
            metrics.update({name + "_" + key: value for key, value in scores.items()})
            collision_steps[name] = step_collisions
        self.metric_keys = list(metrics)
        metrics = {key: value.tolist() for key, value in metrics.items()}
        offset = len(rows)
        for index in range(expert.shape[0]):
            path = dataset.samples[offset + index]
            row = {"sample": str(path.relative_to(dataset.event_dir)),
                   "recording": str(batch["recording"][index]),
                   "event_type": str(batch["event_type"][index]),
                   "anchor_timestamp_us": int(batch["anchor_timestamp_us"][index]),
                   "valid_actions": int(valid[index].sum()),
                   "valid_trajectory_endpoints": int(prefix[index].sum())}
            if log_map is not None:
                if row["recording"] not in log_map:
                    raise ValueError(f"Missing original log ID for {row['recording']}")
                row["log_id"] = log_map[row["recording"]]
            row.update({key: values[index] if math.isfinite(values[index]) else None
                        for key, values in metrics.items()})
            rows.append(row)
        if options["save_predictions"]:
            predictions.append({"samples": [row["sample"] for row in rows[offset:]],
                                "actions": actions, "trajectory": TrajectoryL1Loss.trajectory(actions),
                                "expert_trajectory": expert, "action_mask": valid,
                                "trajectory_mask": prefix, "collision_by_step": collision_steps})

    def on_validation_epoch_end(self):
        options, model, dataset = self.options, self.model, self.dataset
        rows, predictions, model_times = self.rows, self.predictions, self.model_times
        footprint, log_map = self.footprint, self.log_map
        count, total = self.count, len(dataset)
        interval_s, device = FRAME_INTERVAL_US / 1e6, self.device
        root, dataset_dir = self.config["datamodule"]["root"], self.config["datamodule"].get("dir", "mini")
        batch_size = self.trainer.val_dataloaders.batch_size
        workers = self.trainer.val_dataloaders.num_workers
        # Exact duplicate (recording, timestamp, label) entries are not repeated.
        # A multi-label anchor contributes once globally, once to each supported label.
        unique = {}
        event_rows = {}
        for row in rows:
            identity = (row["recording"], row["anchor_timestamp_us"])
            unique.setdefault(identity, row)
            event_rows.setdefault(row["event_type"], {}).setdefault(identity, row)
        global_rows = list(unique.values())
        metric_keys = self.metric_keys
        summary = aggregate(global_rows, metric_keys)
        summary["manifest_windows_evaluated"] = len(rows)
        summary["unique_anchors_evaluated"] = len(global_rows)
        summary["by_event_type"] = {
            kind: aggregate(list(group.values()), metric_keys)
            for kind, group in sorted(event_rows.items())}
        summary["event_macro_average"] = {}
        summary["event_macro_support"] = {}
        for key in metric_keys:
            values = [group["metrics"][key] for group in summary["by_event_type"].values()
                      if group["metrics"][key] is not None]
            summary["event_macro_average"][key] = math.fsum(values)/len(values) if values else None
            summary["event_macro_support"][key] = len(values)
        summary["paired_baseline_comparison"] = paired_comparison(
            global_rows, ("ade_m", "ade_full_horizon_m", "fde_m", "heading_mae_rad", "collision_proxy_any"),
            cluster_key="log_id" if log_map else "recording",
            bootstrap_samples=options["bootstrap_samples"] if log_map else 0, seed=self.seed)
        summary["statistical_protocol"] = {
            "cluster_unit": "original_log_from_user_mapping" if log_map else "recording_or_DB_identity_unverified",
            "ci_enabled": log_map is not None and options["bootstrap_samples"] > 0,
            "bootstrap_samples": options["bootstrap_samples"] if log_map else 0,
            "confidence": 0.95, "resampling": "paired log-cluster percentile bootstrap; window-weighted mean",
            "global_deduplication": "recording + anchor_timestamp_us; first occurrence",
            "event_aggregation": "unique anchor within each label; unweighted mean across supported labels",
            "note": "No log-level CI without an authoritative evaluation.log_map; no inference of log IDs from DB names.",
        }
        summary["collision_protocol"] = {
            "enabled": footprint is not None,
            "disabled_reason": None if footprint else "Set evaluation.ego_footprint: [front, rear, left, right] in metres",
            "footprint_m": vars(footprint) if footprint else None,
            "definition": "fraction of complete windows with >=1 oriented-footprint overlap against logged objects",
            "detection": "discrete target timestamps; boundary contact counts; no swept detection",
            "objects": "all annotated categories; no ground-truth-collision exclusions",
            "max_annotation_age_us": options["max_annotation_age_us"],
            "annotation_time_matching": "absolute timestamp difference <= tolerance; offline annotations may precede or follow targets, never policy inputs",
            "missing_annotations": "excluded, not zero; coverage exported",
            "closed_loop": False, "at_fault": False,
        }
        summary["protocol"] = {
            "task": "autoregressive_action_expert_log_replay",
            "observations": "History, then arrived BEV at each 0.1 s decision; no future action inputs",
            "teacher_forcing": False, "closed_loop": False, "fixed_anchor_forecast": False,
            "history_frames": model.history_features, "future_actions": model.future_action,
            "interval_s": interval_s, "horizon_s": model.future_action * interval_s,
            "aggregation": "unique-anchor mean; ade_m uses valid prefix, ade_full_horizon_m/FDE require complete horizon; fixed-horizon main tables use ade_full_horizon_m",
            "displacement_metrics": "metrics.displacement; one ego, one mode; missing values excluded",
            "constant_action_baseline": "repeat the final pre-anchor local SE(2) action",
            "units": "positions and displacement errors: metres; headings: radians",
        }
        summary["run"] = {
            "checkpoint": str(Path(self.config["trainer"]["ckpt_path"]).resolve()), "config": str(self.config_path.resolve()),
            "data_root": str(Path(root).resolve()), "dataset_dir": dataset_dir,
            "manifest": dataset.processed_file_names_path, "manifest_samples": total,
            "partial_evaluation": count < total, "device": str(device),
            "batch_size": batch_size, "num_workers": workers, "seed": self.seed,
            "log_map": str(Path(options["log_map"]).resolve()) if options["log_map"] else None,
            "torch_version": str(torch.__version__), "model_hyperparameters": dict(model.hparams),
        }
        summary["timing"] = {
            "wall_s": time.perf_counter() - self.started,
            "model_batch_latency_ms": {
                f"p{q}": float(torch.tensor(model_times, dtype=torch.float64).quantile(q / 100) * 1000)
                for q in (50, 95, 99)},
            "model_amortized_ms_per_window": sum(model_times) * 1000 / count,
            "warmup_forwards": options["warmup_batches"],
            "measured_batches": len(model_times),
            "note": "Cached-BEV whole-window batch forward after warmup; excludes encoder, transfers and loading. Neither single-decision nor end-to-end latency.",
        }
        output = self.export(summary)
        with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if options["save_predictions"]:
            torch.save(predictions, output / "predictions.pt")

        for key, value in summary["metrics"].items():
            if value is not None:
                self.log("val/" + key, value, on_step=False, on_epoch=True)


class RefreshEvaluator(Evaluator):
    """Rolling refresh metrics, with the same reductions as model validation."""

    def validation_step(self, batch, batch_idx):
        metrics = self.model.cars_adapter_validation_metrics(batch)
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      batch_size=batch["planning_mask"].shape[0])

    def on_validation_end(self):
        self.export({
            "metrics": {key: float(value) for key, value in self.trainer.callback_metrics.items()},
            "config": self.config, "samples": self.count,
            "manifest_samples": len(self.dataset), "partial_evaluation": self.count < len(self.dataset),
            "checkpoint": str(Path(self.config["trainer"]["ckpt_path"]).resolve()),
            "seed": self.seed, "closed_loop": False,
        })
