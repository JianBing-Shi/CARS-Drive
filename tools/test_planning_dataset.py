"""Check cache slicing and causal ten-decision planning supervision."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from datamodules import NuplanDataModule
from datasets import NuplanEventDataset


def cached_window(root, *, count=41, name="recording/changing_lane/0.pt", offset=0):
    root = Path(root)
    cache = root / "preprocessed_dataset"
    anchor = 25 if count == 41 else 20
    times = torch.arange(count) - anchor
    motion = times.float()
    poses = torch.stack((motion * .1 + motion.square() * .001,
                         motion.square() * .002, motion * .03), -1)
    paths = []
    for index, time in enumerate(times.tolist()):
        relative = Path("preprocessed_dataset/bev_features") / f"{time}.pt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "bev_features": torch.full((4, 2, 2), float(time)),
            "bev_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "reference_ego_to_global": torch.eye(4),
            "image_timestamps_us": torch.full((8,), 10_000_000 + time * 100_000),
            "ego_state": {"speed": torch.tensor(1.)},
            "tracked_objects": {"boxes": torch.zeros(0, 7)},
            "bev_xy": torch.zeros(2, 2, 2), "bev_range": torch.arange(6.),
        }, path)
        paths.append(relative.as_posix())
    timestamps = 10_000_000 + times * 100_000
    mask = torch.ones(count, dtype=torch.bool)
    event = {
        "schema_version": 5 if count == 41 else 4,
        "recording": "recording", "event_type": Path(name).parent.name,
        "anchor_timestamp_us": 10_000_000, "anchor_offset_us": offset,
        "anchor_index": anchor, "timestamps_us": timestamps,
        "relative_time_us": times * 100_000, "bev_paths": paths,
        "anchor_input_mask": times <= 0,
        "future_actions": torch.zeros(count - anchor - 1, 3),
        "future_action_mask": mask[anchor + 1:],
        "future_trajectory": poses[anchor + 1:],
        "trajectory": {
            "past_trajectory": poses[:anchor], "anchor_trajectory": poses[anchor],
            "future_expert_trajectory": poses[anchor + 1:],
            "trajectory_mask": mask, "future_expert_mask": mask[anchor + 1:],
            "trajectory_timestamps_us": timestamps, "pose_timestamps_us": timestamps,
            "history_seconds": anchor * .1, "future_seconds": (count - anchor - 1) * .1,
        },
        "route_condition": {},
    }
    destination = cache / "events" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(event, destination)
    for split in ("train", "val"):
        torch.save([name], cache / f"mini-{split}.pt")
    return destination, event, poses


def load_dataset(root, mode="pred"):
    return NuplanEventDataset(str(root), "mini", "train", root,
                              history_step=20, future_step=10, preprocess=False, mode=mode)


class PlanningDatasetTests(unittest.TestCase):
    def test_pred_41_matches_legacy_31_and_preserves_anchor_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            expanded_root, legacy_root = Path(temporary) / "expanded", Path(temporary) / "legacy"
            cached_window(expanded_root)
            cached_window(legacy_root, count=31)
            expanded, legacy = load_dataset(expanded_root).get(0), load_dataset(legacy_root).get(0)
            for key in ("bev_features", "timestamps_us", "relative_time_us", "future_timestamps_us",
                        "history_actions", "future_actions", "future_trajectory", "future_action_mask",
                        "anchor_input_mask", "replay_causal_mask"):
                torch.testing.assert_close(expanded[key], legacy[key], msg=key)
            for key in expanded["trajectory"]:
                torch.testing.assert_close(expanded["trajectory"][key], legacy["trajectory"][key], msg=key)
            self.assertEqual(expanded["bev_features"].shape[0], 31)
            self.assertEqual(expanded["anchor_index"], 20)
            self.assertEqual(expanded["relative_time_us"][[0, 20, 30]].tolist(), [-2_000_000, 0, 1_000_000])
            self.assertEqual(expanded["history_actions"].shape, (21, 3))
            self.assertNotIn("planning_targets", expanded)

    def test_plan_modes_use_ten_full_horizons_and_twenty_observed_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, poses = cached_window(temporary)
            for mode in ("plan1", "plan2"):
                with self.subTest(mode=mode):
                    sample = load_dataset(temporary, mode).get(0)
                    self.assertEqual(sample["planning_start_index"], 20)
                    self.assertEqual(sample["planning_decision_indices"].tolist(), list(range(21, 31)))
                    self.assertEqual(sample["planning_targets"].shape, (10, 10, 3))
                    self.assertEqual(sample["planning_mask"].shape, (10, 10))
                    self.assertTrue(sample["planning_mask"].all())
                    self.assertEqual(sample["action_history_sequence"].shape, (11, 20, 3))
                    self.assertEqual(sample["history_actions"].shape, (20, 3))
                    self.assertEqual(sample["relative_time_us"][[20, 21, 30, 40]].tolist(),
                                     [-500_000, -400_000, 500_000, 1_500_000])
                    for step in range(11):
                        past = poses[step:step + 20]
                        delta = poses[step + 1:step + 21] - past
                        yaw = past[:, 2]
                        expected = torch.stack((yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
                                                -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1],
                                                torch.atan2(delta[:, 2].sin(), delta[:, 2].cos())), -1)
                        torch.testing.assert_close(sample["action_history_sequence"][step], expected)
                    delta = poses[40] - poses[30]
                    yaw = poses[30, 2]
                    expected = torch.stack((yaw.cos() * delta[0] + yaw.sin() * delta[1],
                                            -yaw.sin() * delta[0] + yaw.cos() * delta[1], delta[2]))
                    torch.testing.assert_close(sample["planning_targets"][-1, -1], expected)
                    self.assertEqual(sample["replay_causal_mask"].sum(1).tolist(), list(range(22, 32)))
                    self.assertFalse(sample["anchor_input_mask"][21:].any())

    def test_future_pose_changes_cannot_change_earlier_action_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, event, _ = cached_window(temporary)
            original = load_dataset(temporary, "plan1").get(0)
            # The first changed pose arrives at decision 2 (cache slot 22).
            event["trajectory"]["past_trajectory"][22:, 0] += 3
            torch.save(event, path)
            changed = load_dataset(temporary, "plan1").get(0)
            torch.testing.assert_close(original["action_history_sequence"][:2],
                                       changed["action_history_sequence"][:2])
            self.assertFalse(torch.equal(original["action_history_sequence"][2],
                                         changed["action_history_sequence"][2]))

    def test_pose_gaps_mask_every_later_endpoint_and_pred_adjacent_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, event, _ = cached_window(temporary)
            event["trajectory"]["trajectory_mask"][26] = False
            torch.save(event, path)
            sample = load_dataset(temporary, "plan2").get(0)
            self.assertEqual(sample["planning_mask"][0].tolist(), [True] * 4 + [False] * 6)
            self.assertFalse(sample["planning_mask"][4:6].any())
            self.assertTrue(sample["planning_mask"][6:].all())
            pred = load_dataset(temporary).get(0)
            self.assertEqual(pred["future_action_mask"].tolist(), [False, False] + [True] * 8)
            self.assertEqual(pred["trajectory"]["trajectory_mask"].shape, (31,))

    def test_plan_requires_expanded_cache_and_rejects_legacy_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            cached_window(temporary, count=31)
            with self.assertRaisesRegex(ValueError, "41-frame cache"):
                load_dataset(temporary, "plan1").get(0)
            with self.assertRaisesRegex(ValueError, "Unsupported dataset mode"):
                load_dataset(temporary, "plan")

    def test_plan_datamodules_keep_event_anchor_and_controls_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            names = ["recording/changing_lane/0.pt", "recording/changing_lane/1.pt",
                     "recording/background_control/2.pt"]
            for name, offset in zip(names, (0, 100_000, 100_000)):
                cached_window(temporary, name=name, offset=offset)
            for split in ("train", "val"):
                torch.save(names, Path(temporary) / "preprocessed_dataset" / f"mini-{split}.pt")
            for mode in ("plan1", "plan2"):
                module = NuplanDataModule(root=temporary, mode=mode, num_workers=0,
                                          train_batch_size=2, val_batch_size=2)
                with patch.object(NuplanEventDataset, "_load_model", side_effect=AssertionError("No preprocessing")):
                    module.setup("fit")
                    self.assertEqual(len(module.train_dataset), 2)
                    batch = next(iter(module.train_dataloader()))
                    self.assertEqual(batch["action_history_sequence"].shape, (2, 11, 20, 3))
                    self.assertEqual(batch["planning_targets"].shape, (2, 10, 10, 3))
                    self.assertEqual(batch["planning_start_index"].tolist(), [20, 20])


if __name__ == "__main__":
    unittest.main()
