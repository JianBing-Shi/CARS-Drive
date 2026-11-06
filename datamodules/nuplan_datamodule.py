"""Lightning DataModule for the compact FRESH-Drive event cache."""

from __future__ import annotations

from typing import Iterable, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, default_collate

from datasets import NuplanEventDataset


def collate_events(samples):
    """Batch fixed-shape fields; retain variable annotations per event."""
    annotations = {"route_condition", "tracked_objects", "ego_states", "bev_paths"}
    return {key: [sample[key] for sample in samples] if key in annotations
            else default_collate([sample[key] for sample in samples])
            for key in samples[0]}


class NuplanDataModule(pl.LightningDataModule):
    """Load preprocessed events using train/val manifests in the Plan-R1 style."""

    def __init__(self,
                 root: str,
                 event_types: Iterable[str] | None = None,
                 train_batch_size: int = 8,
                 val_batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 dir: str = "mini",
                 bevformer_root: str = "3rdparty/BEVFormer",
                 mode: str = "action",
                 history_features: int = 20,
                 future_action: int = 10,
                 ratio: float = .1) -> None:
        super().__init__()
        self.root = root
        self.dir = dir
        self.mode = mode
        self.history_features = history_features
        self.future_action = future_action
        self.ratio = ratio
        self.bevformer_root = bevformer_root
        self.event_types = sorted(event_types) if event_types is not None else None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: Optional[str] = None) -> None:
        splits = ("train", "val") if stage in (None, "fit") else ("val",)
        for split in splits:
            dataset = NuplanEventDataset(
                root=self.root, dir=self.dir, split=split, bevformer_root=self.bevformer_root,
                history_features=self.history_features, future_action=self.future_action,
                ratio=self.ratio, event_types=self.event_types, preprocess=False)
            if self.mode == "refresh":
                dataset.samples = [path for path in dataset.samples
                                   if path.parent.name == "background_control"
                                   or torch.load(path, map_location="cpu", weights_only=False).get("anchor_offset_us", 0) == 0]
            dataset._processed_file_names = [str(path.relative_to(dataset.event_dir)) for path in dataset.samples]
            setattr(self, f"{split}_dataset", dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, collate_fn=collate_events,
                          **({"prefetch_factor": self.prefetch_factor} if self.num_workers > 0 else {}))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, collate_fn=collate_events,
                          **({"prefetch_factor": self.prefetch_factor} if self.num_workers > 0 else {}))
