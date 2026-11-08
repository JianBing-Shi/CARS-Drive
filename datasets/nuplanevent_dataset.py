"""Event-filtered nuPlan BEV sequences and independent planning annotations.

Future BEV and object annotations are replay/evaluation data, not anchor-time inputs.
"""
from __future__ import annotations

import random
import re
import sqlite3
import sys
from collections import Counter
from bisect import bisect_right
from contextlib import closing
from pathlib import Path

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from datasets import dataset_utils
from datasets.dataset_utils import (
    load_nuplan_camera_frame, extract_camera_features, camera_bev_features,
    load_nuplan_trajectory, load_nuplan_ego_state, load_nuplan_route_condition,
    load_nuplan_tracked_objects, load_nuplan_events,
)
from utils.camera import (
    load_backbone_encoder, load_bevformer_encoder,
    camera_embedding_indices_from_calibration,
)

class NuplanEventDataset(Dataset):
    """Cache shared BEVs and build event/control windows split by recording log."""

    # The shared preprocessing cache is deliberately wider than either model
    # input.  It covers [-2.5s, +1.5s] around each anchor at 10 Hz.
    CACHE_HISTORY_STEP = 25
    CACHE_FUTURE_STEP = 15
    CACHE_SCHEMA_VERSION = 5

    def __init__(self, root: str, dir: str, split: str, bevformer_root: str | Path,
                 history_step: int = 25, future_step: int = 15,
                 batch_size: int = 32, ratio: float = .1,
                 camera_embedding_indices=None, event_types=None, preprocess: bool = True,
                 cache_only: bool = False, mode: str = "pred"):
        self._data_root = Path(root).expanduser().resolve()
        if dir in ['train', 'val', 'test', 'mini']:
            self.dir = dir
        else:
            raise ValueError(dir + ' is not valid')
        if split in ['train', 'val']:
            self.split = split
        else:
            raise ValueError(split + ' is not valid')

        self.interval_us = dataset_utils.FRAME_INTERVAL_US
        self.batch_size = batch_size
        self.ratio = ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.history_step, self.future_step = history_step, future_step
        self.window_samples = history_step + 1 + future_step
        if mode not in {"pred", "plan1", "plan2"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")
        self.mode = mode
        self.bevformer_root = Path(bevformer_root).resolve()
        self.checkpoint = self.bevformer_root / "projects/ckpts/bevformer_tiny_epoch_24.pth"
        self.source_indices = camera_embedding_indices
        self._automatic_camera_mapping = camera_embedding_indices is None
        self.preprocess = preprocess
        self.cache_only = cache_only
        self.image_encoder = self.bev_encoder = None
        self._processed_file_names = []
        super().__init__(root=str(self._data_root))
        self._processed_file_names = torch.load(
            self.processed_file_names_path, map_location="cpu", weights_only=False)
        if event_types is not None:
            selected = set(event_types)
            self._processed_file_names = [name for name in self._processed_file_names
                                          if Path(name).parent.name in selected]
        self.samples = [Path(self.event_dir) / name for name in self._processed_file_names]

    @property
    def raw_img_dir(self) -> str:
        return str(self._data_root / "sensor_blobs" / self.dir)

    @property
    def raw_db_dir(self) -> str:
        return str(self._data_root / "splits" / self.dir)

    @property
    def processed_dir(self) -> str:
        return str(self._data_root / "preprocessed_dataset")

    @property
    def event_dir(self) -> str:
        return str(Path(self.processed_dir) / "events")

    @property
    def feature_dir(self) -> str:
        return str(Path(self.processed_dir) / "features")

    @property
    def bev_feature_dir(self) -> str:
        return str(Path(self.processed_dir) / "bev_features")

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.dir}-{split}.pt" for split in ("train", "val")]

    @property
    def processed_file_names_path(self) -> str:
        return str(Path(self.processed_dir) / f"{self.dir}-{self.split}.pt")

    def _process(self):
        if not self.preprocess and Path(self.processed_file_names_path).is_file():
            return
        if all(Path(path).is_file() for path in self.processed_paths):
            return
        if self.log and "pytest" not in sys.modules:
            print("Processing...", file=sys.stderr)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.process()
        if self.log and "pytest" not in sys.modules:
            print("Done!", file=sys.stderr)

    def _load_model(self):
        # Load once per preprocessing run, then reuse across all DBs/windows.
        self.image_encoder = load_backbone_encoder(self.checkpoint, device=self.device)
        self.bev_encoder = load_bevformer_encoder(
            self.checkpoint, device=self.device, bevformer_root=self.bevformer_root,
            camera_embedding_indices=self.source_indices,
        )

    def collect_database(self, database):
        """CPU: collect complete trajectory windows for the current DB."""
        windows, rejected = [], Counter()
        candidates = dataset_utils.load_window_candidates(
            database, self.history_step * self.interval_us, self.future_step * self.interval_us)
        for kind, anchor, source_timestamp, offset in candidates:
            trajectory = load_nuplan_trajectory(
                database, anchor,
                history_seconds=self.history_step * self.interval_us / 1e6,
                future_seconds=self.future_step * self.interval_us / 1e6,
            )
            if not trajectory["trajectory_mask"].all():
                rejected["incomplete_trajectory"] += 1
                continue
            if kind == "background_control":
                future = trajectory["future_expert_trajectory"]
                actions = dataset_utils.local_actions(future)
                if (future[:, 1].abs().max() > .3 or future[:, 2].abs().max() > .1
                        or (len(actions) > 1 and (actions[1:, :2] - actions[:-1, :2]).norm(dim=-1).max() > .15)):
                    rejected["nonsteady_control"] += 1
                    continue
            trajectory["window_metadata"] = {
                "window_role": "control" if kind == "background_control" else "event",
                "source_event_timestamp_us": source_timestamp,
                "anchor_offset_us": offset,
            }
            windows.append((kind, anchor, trajectory))
        return windows, rejected

    def encode(self, database, windows):
        """Encode each required timestamp once, reusing the same encoders."""
        timestamps = sorted({timestamp for _, _, trajectory in windows
                             for timestamp in trajectory["trajectory_timestamps_us"].tolist()})
        if self.cache_only:
            # Existing BEVs are keyed by the latest front-camera image filename.
            with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                rows = connection.execute(
                    "SELECT i.timestamp,i.filename_jpg FROM image i JOIN camera c "
                    "ON i.camera_token=c.token WHERE c.channel='CAM_F0' ORDER BY i.timestamp").fetchall()
            capture_times = [row[0] for row in rows]
            paths = {}
            for timestamp in timestamps:
                index = bisect_right(capture_times, timestamp) - 1
                path = (Path(self.bev_feature_dir) / database.stem / Path(rows[index][1]).with_suffix(".pt").name
                        if index >= 0 and timestamp - capture_times[index] <= 100_000 else None)
                paths[timestamp] = path if path is not None and path.is_file() else None
            return paths
        paths = {}
        with tqdm(total=len(timestamps), desc=f"Frames {database.stem}",
                  unit="frame", position=1, leave=True, dynamic_ncols=True) as progress:
            for start in range(0, len(timestamps), self.batch_size):
                group = timestamps[start:start + self.batch_size]
                pending = []
                for timestamp in group:
                    progress.set_postfix_str(f"read/extract timestamp={timestamp}", refresh=True)
                    pending.append(self._frame(database, timestamp, defer_bev=True))
                for timestamp, item in zip(group, pending):
                    if isinstance(item, Path):
                        paths[timestamp] = item
                valid = [item for item in pending if isinstance(item, dict)]
                if valid:
                    progress.set_postfix_str(f"BEV fusion batch={len(valid)}", refresh=True)
                    outputs = camera_bev_features(
                        [item["images"] for item in valid], self.bev_encoder,
                        [item["frame"] for item in valid], batch_size=self.batch_size)
                    progress.set_postfix_str("save BEV", refresh=True)
                    for item, bev in zip(valid, outputs):
                        destination = item["destination"]
                        payload = {**bev, "ego_state": item["state"],
                                   "tracked_objects": item["objects"]}
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = destination.with_suffix(".tmp")
                        torch.save(payload, temporary)
                        temporary.replace(destination)
                        paths[item["timestamp"]] = destination
                progress.update(len(group))
        paths.update({timestamp: None for timestamp in timestamps if timestamp not in paths})
        return paths

    def _frame(self, database, timestamp, defer_bev=False):
        """One deduplicated eight-view BEV plus annotation cache per DB/time."""
        frame = load_nuplan_camera_frame(
            database, self.raw_img_dir, timestamp, skip_unavailable=True)
        if frame is None:
            return None
        filename = Path(frame["image_paths"]["CAM_F0"]).with_suffix(".pt").name
        destination = Path(self.bev_feature_dir) / database.stem / filename
        if destination.is_file():
            return destination
        if self.cache_only:
            return None
        if not all(Path(path).is_file() for path in frame["image_paths"].values()):
            return None
        state = load_nuplan_ego_state(database, timestamp)
        objects = load_nuplan_tracked_objects(
            database, timestamp, reference_ego_to_global=frame["reference_ego_to_global"])
        if self._automatic_camera_mapping:
            indices = camera_embedding_indices_from_calibration(frame["camera_to_ego"])
            if indices != self.source_indices:
                self.source_indices = indices
                print(f"\nCamera embedding mapping ({database.stem}): "
                      f"{dict(zip(dataset_utils.CAMERAS, indices))}", flush=True)
                if self.bev_encoder is not None:
                    with torch.no_grad():
                        self.bev_encoder.transformer.cams_embeds.copy_(
                            self.bev_encoder.source_camera_embeddings[indices])
        if self.image_encoder is None:
            self._load_model()
        image_destinations = [
            Path(self.feature_dir) / database.stem / camera
            / Path(frame["image_paths"][camera]).with_suffix(".pt").name
            for camera in dataset_utils.CAMERAS
        ]
        tensor_fields = {"camera_features", "image_transforms"}
        camera_fields = tensor_fields | {
            "camera_order", "source_paths", "original_shapes", "resized_shapes",
            "image_shapes", "image_tokens",
        }
        views = [torch.load(path, map_location="cpu", weights_only=False)
                 for path in image_destinations] if all(path.is_file() for path in image_destinations) else []
        valid_views = bool(views) and all(
            isinstance(view, dict)
            and view.get("calibration_database") == frame["database"]
            and view.get("image_tokens") == (frame["image_tokens"][index],)
            and view.get("camera_order") == (dataset_utils.CAMERAS[index],)
            and view.get("rectification_applied") is True
            for index, view in enumerate(views)
        )
        if valid_views:
            images = {
                key: (torch.cat([view[key] for view in views]) if key in tensor_fields
                      else sum((view[key] for view in views), ()) if key in camera_fields
                      else value)
                for key, value in views[0].items()
            }
        else:
            images = extract_camera_features(frame["image_paths"], self.image_encoder,
                                             calibration=frame, batch_size=8 * self.batch_size)
            for index, path in enumerate(image_destinations):
                view = {key: value[index:index + 1] if key in camera_fields else value
                        for key, value in images.items()}
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                torch.save(view, temporary)
                temporary.replace(path)
        if defer_bev:
            return {"timestamp": timestamp, "frame": frame, "images": images,
                    "state": state, "objects": objects, "destination": destination}
        bev = camera_bev_features(images, self.bev_encoder, frame)
        payload = {**bev, "ego_state": state, "tracked_objects": objects}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination

    def save_event(self, database, kind, anchor, trajectory, frame_paths):
        """Save annotations and feature references; no encoding here."""
        timestamps = trajectory["trajectory_timestamps_us"]
        paths = [str(frame_paths[timestamp].relative_to(self._data_root))
                 for timestamp in timestamps.tolist()]
        route = load_nuplan_route_condition(database, anchor)
        # All data remain accessible for replay, but anchor training reads <= t.
        offsets = torch.arange(-self.history_step, self.future_step + 1)
        replay_causal = offsets[None, :] <= torch.arange(self.future_step)[:, None]
        future_trajectory = trajectory["future_expert_trajectory"]
        future_actions = dataset_utils.local_actions(future_trajectory)
        payload = {
            "schema_version": self.CACHE_SCHEMA_VERSION, "recording": database.stem, "event_type": kind,
            **trajectory.get("window_metadata", {}),
            "anchor_timestamp_us": anchor, "sampling_interval_us": self.interval_us,
            "timestamps_us": timestamps, "relative_time_us": offsets * self.interval_us,
            "anchor_index": self.history_step,
            "bev_paths": paths,
            "trajectory": trajectory, "route_condition": route,
            "future_actions": future_actions,
            "future_action_mask": (trajectory["trajectory_mask"][self.history_step:-1]
                                   & trajectory["trajectory_mask"][self.history_step + 1:]),
            "future_trajectory": future_trajectory.clone(),
            "anchor_input_mask": offsets <= 0,
            "replay_causal_mask": replay_causal,
            "causal_context_mask": replay_causal,
            "future_annotations_are_supervision_only": True,
            "future_bev_available_only_after_arrival": True,
        }
        relative = Path(database.stem) / kind / f"{anchor}.pt"
        destination = Path(self.event_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        return relative.as_posix()

    def process_database(self, database):
        """Keep window metadata local to this DB, returning only filenames."""
        windows, rejected = self.collect_database(database)
        frame_paths = self.encode(database, windows)
        names = []
        for kind, anchor, trajectory in windows:
            if any(frame_paths[timestamp] is None
                   for timestamp in trajectory["trajectory_timestamps_us"].tolist()):
                rejected["missing_or_stale_camera_frame"] += 1
                continue
            names.append(self.save_event(database, kind, anchor, trajectory, frame_paths))
        return names, rejected

    def process(self):
        if not self.preprocess:
            raise FileNotFoundError(
                f"Preprocess the dataset before training: {self.processed_dir}")
        self._databases = sorted(Path(self.raw_db_dir).glob("*.db"))
        if not self._databases:
            raise FileNotFoundError(f"No nuPlan DB files found in {self.raw_db_dir}")
        names, rejected = [], Counter()
        for database in tqdm(self._databases, desc="nuPlan DB"):
            database_names, database_rejected = self.process_database(database)
            names.extend(database_names)
            rejected.update(database_rejected)

        # Read authoritative log identities from DB metadata, never parse DB filenames.
        recording_logs = {}
        for database in self._databases:
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
                logfile = connection.execute("SELECT logfile FROM log").fetchone()[0]
                # nuPlan metadata may include the segment suffix in logfile itself.
                recording_logs[database.stem] = re.sub(r"_\d+_\d+$", "", logfile)
        logs = sorted(set(recording_logs.values()))
        rng = random.Random(42)
        rng.shuffle(logs)
        val_logs = set(logs[:max(1, int(len(logs) * self.ratio))]) if len(logs) > 1 else set()
        splits = {"train": [], "val": []}
        for name in names:
            split = "val" if recording_logs[Path(name).parts[0]] in val_logs else "train"
            splits[split].append(name)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        for split, events in splits.items():
            rng.shuffle(events)
            torch.save(events, Path(self.processed_dir) / f"{self.dir}-{split}.pt")
        if rejected:
            print(f"Skipped windows: {dict(rejected)}")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        event = torch.load(self.samples[idx], map_location="cpu", weights_only=False)
        count = len(event["bev_paths"])
        cache_anchor = int(event["anchor_index"])
        planning = self.mode in {"plan1", "plan2"}
        if planning:
            if count != 41 or cache_anchor != self.CACHE_HISTORY_STEP:
                raise ValueError("plan1/plan2 require the expanded 41-frame cache [-2.5s, +1.5s]")
            if self.history_step != 20 or self.future_step != 10:
                raise ValueError("plan1/plan2 require history_step=20 and future_step=10")
            begin, end = 0, count
        elif count == 41 and cache_anchor == self.CACHE_HISTORY_STEP:
            # The frozen predictor keeps its original [-2s, +1s] interface.
            begin, end = 5, 36
        elif count == self.history_step + 1 + self.future_step:
            # Existing 31-frame pred caches and small preprocessing fixtures.
            begin, end = 0, count
        else:
            raise ValueError("Unsupported event cache length; expected a 41-frame cache or configured pred window")

        anchor = cache_anchor - begin
        timestamps = event["timestamps_us"]
        event["timestamps_us"] = timestamps[begin:end]
        # Use the original anchor before slicing: original slot 20 is -0.5s in
        # a 41-frame cache, whereas the event anchor is original slot 25.
        event["relative_time_us"] = timestamps[begin:end] - timestamps[cache_anchor]
        event["bev_paths"] = event["bev_paths"][begin:end]
        event["anchor_index"] = anchor
        trajectory = dict(event["trajectory"])
        if "past_trajectory" in trajectory:
            trajectory["past_trajectory"] = trajectory["past_trajectory"][begin:cache_anchor].float()
            trajectory["anchor_trajectory"] = trajectory["anchor_trajectory"].float()
        trajectory["future_expert_trajectory"] = trajectory["future_expert_trajectory"][:end - cache_anchor - 1].float()
        for name in ("trajectory_mask", "trajectory_timestamps_us", "pose_timestamps_us"):
            if name in trajectory:
                trajectory[name] = trajectory[name][begin:end]
        if "future_expert_mask" in trajectory:
            trajectory["future_expert_mask"] = trajectory["future_expert_mask"][:end - cache_anchor - 1]
        interval_us = int(event.get("sampling_interval_us", self.interval_us))
        if "history_seconds" in trajectory:
            trajectory["history_seconds"] = anchor * interval_us / 1e6
        if "future_seconds" in trajectory:
            trajectory["future_seconds"] = (end - cache_anchor - 1) * interval_us / 1e6
        event["trajectory"] = trajectory

        frames = [torch.load(self._data_root / name, map_location="cpu", weights_only=False)
                  for name in event["bev_paths"]]
        event["bev_features"] = torch.stack([x["bev_features"] for x in frames])
        event["bev_valid_mask"] = torch.stack([x["bev_valid_mask"] for x in frames])
        event["reference_ego_to_global"] = torch.stack([x["reference_ego_to_global"] for x in frames])
        event["image_timestamps_us"] = torch.stack([x["image_timestamps_us"] for x in frames])
        event["ego_states"] = [x["ego_state"] for x in frames]
        event["tracked_objects"] = [x["tracked_objects"] for x in frames]
        event["bev_xy"], event["bev_range"] = frames[0]["bev_xy"], frames[0]["bev_range"]
        event["bev_range"] = torch.as_tensor(event["bev_range"], dtype=torch.float32)
        event["future_trajectory"] = trajectory["future_expert_trajectory"]
        event["future_actions"] = dataset_utils.local_actions(event["future_trajectory"])
        if "trajectory_mask" in trajectory:
            pose_mask = trajectory["trajectory_mask"].bool()
            event["future_action_mask"] = pose_mask[anchor:-1] & pose_mask[anchor + 1:]
        else:
            event["future_action_mask"] = event["future_action_mask"][:len(event["future_actions"])].bool()
        if "past_trajectory" in trajectory:
            past = torch.cat((trajectory["past_trajectory"], trajectory["anchor_trajectory"][None]), 0)
            # Preserve pred's established output. The predictor consumes the
            # final history_step genuine increments, discarding the first pad.
            event["history_actions"] = dataset_utils.local_actions(past)
        else:
            event["history_actions"] = event["future_actions"].new_zeros(self.history_step, 3)
        slots = torch.arange(len(event["bev_paths"]))
        event["anchor_input_mask"] = slots <= anchor
        event["replay_causal_mask"] = slots[None, :] <= anchor + torch.arange(self.future_step)[:, None]
        event["causal_context_mask"] = event["replay_causal_mask"]
        event["future_timestamps_us"] = event["timestamps_us"][anchor + 1:]
        event["anchor_timestamp_us"] = torch.as_tensor(event["anchor_timestamp_us"], dtype=torch.long)
        event["split"] = self.split

        if planning:
            if "past_trajectory" not in trajectory or "trajectory_mask" not in trajectory:
                raise ValueError("Planning requires recorded past/future poses and their validity mask")
            poses = torch.cat((trajectory["past_trajectory"], trajectory["anchor_trajectory"][None],
                               trajectory["future_expert_trajectory"]))
            # Initialize at -0.5s, then make ten causal decisions at -0.4s to
            # +0.5s. Every new candidate has ten future endpoints, up to +1.5s.
            start = anchor - 5
            decisions = torch.arange(start + 1, start + 11)
            endpoints = decisions[:, None] + torch.arange(1, self.future_step + 1)[None]
            delta = poses[endpoints, :2] - poses[decisions, None, :2]
            yaw = poses[decisions, 2, None]
            angles = poses[endpoints, 2] - yaw
            event["planning_targets"] = torch.stack((
                yaw.cos() * delta[..., 0] + yaw.sin() * delta[..., 1],
                -yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                torch.atan2(angles.sin(), angles.cos())), -1)
            mask = trajectory["trajectory_mask"].bool()
            # A valid cumulative endpoint requires every intervening pose.
            valid = mask[decisions, None] & mask[endpoints]
            event["planning_mask"] = valid.long().cumprod(-1).bool()
            # local_actions pads an initial origin; dropping it leaves exactly
            # the observed adjacent-pose increments. No candidate action enters
            # the real history supplied to later decisions.
            actions = dataset_utils.local_actions(poses)[1:]
            observed_at = torch.arange(start, start + 11)
            event["action_history_sequence"] = torch.stack([
                actions[time - self.history_step:time] for time in observed_at.tolist()])
            event["history_actions"] = event["action_history_sequence"][0]
            event["planning_start_index"] = start
            event["planning_decision_indices"] = decisions
            event["anchor_input_mask"] = slots <= start
            event["replay_causal_mask"] = slots[None, :] <= decisions[:, None]
            event["causal_context_mask"] = event["replay_causal_mask"]
        return event


if __name__ == "__main__":
    root = "../datasets/nuplan/nuplan-v1.1"
    bevformer_root = "3rdparty/BEVFormer"
    NuplanEventDataset(root=root, dir='mini', split='train', bevformer_root=bevformer_root)
    NuplanEventDataset(root=root, dir='mini', split='val', bevformer_root=bevformer_root)
