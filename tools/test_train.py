"""Exercise training and checkpoint recovery with a tiny synthetic BEV cache."""

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml


class TrainTests(unittest.TestCase):
    def test_train_and_resume_pred_and_both_planning_modes(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="fresh-training-", dir="/tmp") as temporary:
            root = Path(temporary)
            cache = root / "preprocessed_dataset"
            bev = Path("preprocessed_dataset/bev_features/frame.pt")
            (root / bev).parent.mkdir(parents=True)
            torch.save({
                "bev_features": torch.randn(4, 6, 6),
                "bev_valid_mask": torch.ones(6, 6, dtype=torch.bool),
                "reference_ego_to_global": torch.eye(4),
                "image_timestamps_us": torch.ones(8, dtype=torch.long),
                "ego_state": {}, "tracked_objects": {},
                "bev_xy": torch.zeros(6, 6, 2), "bev_range": torch.arange(6.),
            }, root / bev)
            event = {
                "anchor_index": 1, "anchor_timestamp_us": 100_000,
                "timestamps_us": torch.arange(4) * 100_000,
                "bev_paths": [bev.as_posix()] * 4, "trajectory": {},
                "route_condition": {},
                "future_actions": torch.tensor([[.1, 0., 0.], [.1, 0., 0.]]),
                "future_trajectory": torch.tensor([[.1, 0., 0.], [.2, 0., 0.]]),
                "future_action_mask": torch.ones(2, dtype=torch.bool),
            }
            poses = torch.zeros(41, 3)
            poses[:, 0] = (torch.arange(41) - 25) * .1
            event.update(anchor_index=25, anchor_timestamp_us=2_500_000,
                         timestamps_us=torch.arange(41) * 100_000, bev_paths=[bev.as_posix()] * 41,
                         future_trajectory=poses[26:], future_actions=torch.tensor([[.1, 0., 0.]] * 15),
                         future_action_mask=torch.ones(15, dtype=torch.bool),
                         trajectory={"past_trajectory": poses[:25], "anchor_trajectory": poses[25],
                                     "future_expert_trajectory": poses[26:], "trajectory_mask": torch.ones(41, dtype=torch.bool)})
            for split in ("train", "val"):
                name = Path(split) / "changing_lane/sample.pt"
                destination = cache / "events" / name
                destination.parent.mkdir(parents=True)
                torch.save(event, destination)
                torch.save([name.as_posix()], cache / f"mini-{split}.pt")

            config = yaml.safe_load(Path("config/train/pred.yaml").read_text())
            config["datamodule"].update(
                root=str(root), dir="mini", history_step=20, future_step=10,
                train_batch_size=1, val_batch_size=1, num_workers=0, pin_memory=False)
            config["model"].update(
                history_step=20, future_step=10, feature_dim=4, hidden_dim=8,
                heads=2, layers=1, spatial_grid=[3, 3], macro_grid=[3, 3],
                dropout=0, anchor_dropout=0, warmup_epochs=1, T_max=4)
            config["trainer"].update(
                accelerator="cpu", max_epochs=1, num_sanity_val_steps=0,
                ckpt_path=None,
                enable_progress_bar=False, enable_model_summary=False,
                log_every_n_steps=1)
            config["trainer"]["csv_logger"]["save_dir"] = str(root / "logs")
            config_file = root / "train.yaml"

            def run():
                config_file.write_text(yaml.safe_dump(config))
                with patch.object(sys, "argv", ["train.py", "--config", str(config_file)]):
                    return runpy.run_path("train.py", run_name="__main__")["trainer"]

            action = run()
            action_path = Path(action.checkpoint_callback.last_model_path)
            original_action_dir = action_path.parent
            self.assertTrue(Path(action.checkpoint_callback.best_model_path).is_file())
            self.assertEqual(action.checkpoint_callback.best_model_score,
                             action.callback_metrics['val/ade'])
            self.assertEqual(Path(action.checkpoint_callback.best_model_path).name, 'best.ckpt')
            self.assertEqual(action_path.name, 'last.ckpt')
            self.assertEqual({path.name for path in action_path.parent.glob('*.ckpt')},
                             {'best.ckpt', 'last.ckpt'})
            action_state = torch.load(action_path, map_location="cpu", weights_only=False)
            self.assertEqual(action_state["global_step"], 1)
            self.assertEqual(action_state["hyper_parameters"]["mode"], "pred")
            for key, expected in {'action_loss': 'l1', 'action_loss_version': 2,
                                  'xy_loss_weight': .50, 'yaw_weight': .10,
                                  'trajectory_loss_weight': .35, 'heading_loss_weight': .05}.items():
                self.assertEqual(action_state['hyper_parameters'][key], expected)
            self.assertNotIn('smooth_l1_beta', action_state['hyper_parameters'])
            self.assertEqual(action_state['action_loss_config']['xy_loss_weight'], .50)
            self.assertEqual(action_state['action_loss_config']['action_loss_version'], 2)
            config["trainer"].update(ckpt_path=str(action_path), max_epochs=2)
            config['model']['xy_loss_weight'] = .60
            with self.assertRaisesRegex(ValueError, 'Checkpoint loss settings differ'):
                run()
            config['model']['xy_loss_weight'] = .50
            resumed = run()
            self.assertEqual(Path(resumed.checkpoint_callback.dirpath), original_action_dir)
            self.assertEqual(len(resumed.checkpoint_callback.best_k_models), 1)
            self.assertEqual({path.name for path in original_action_dir.glob('*.ckpt')},
                             {'best.ckpt', 'last.ckpt'})
            self.assertEqual(resumed.checkpoint_callback.best_model_score,
                             min(resumed.checkpoint_callback.best_k_models.values()))
            self.assertTrue(Path(resumed.checkpoint_callback.best_model_path).is_file())
            action_path = Path(resumed.checkpoint_callback.last_model_path)
            action_state = torch.load(action_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(action_state["epoch"], 1)
            self.assertEqual(action_state["lr_schedulers"][0]["last_epoch"], 2)
            self.assertTrue(all(int(state["step"]) == 2
                                for state in action_state["optimizer_states"][0]["state"].values()))

            # Schedule edits must not turn a full resume into weights-only loading.
            config['model']['learning_rate'] = 1e-5
            with self.assertRaisesRegex(ValueError, 'schedule settings differ'):
                run()
            config['model']['learning_rate'] = .0003

            # Each scheme starts independently from identical pred weights.
            for mode in ("plan1", "plan2"):
                config["model"].update(mode=mode, cache_anchor_step=25, plan_horizon=10,
                                       refresh_classification_weight=1. if mode == "plan1" else 0.,
                                       quality_loss_weight=1., gain_loss_weight=1.)
                config["datamodule"]["mode"] = mode
                config["trainer"]["ckpt"]["monitor"] = "val/ade"
                config["trainer"]["csv_logger"]["name"] = mode
                config["trainer"]["max_epochs"] = 1
                config["trainer"]["ckpt_path"] = str(action_path)
                planner = run()
                self.assertEqual(planner.global_step, 1)
                self.assertEqual(planner.lightning_module.mode, mode)
                for name, value in planner.lightning_module.state_dict().items():
                    if name.startswith(("tokenizer.", "memory.", "action_encoder.", "decoder.")):
                        torch.testing.assert_close(value, action_state["state_dict"][name])
                for name in ("tokenizer", "memory", "action_encoder", "decoder"):
                    component = getattr(planner.lightning_module, name)
                    self.assertFalse(component.training)
                    self.assertTrue(all(not p.requires_grad for p in component.parameters()))
                    self.assertTrue(all(p.grad is None for p in component.parameters()))
                planner_path = Path(planner.checkpoint_callback.last_model_path)
                original_planner_dir = planner_path.parent
                config["trainer"].update(ckpt_path=str(planner_path), max_epochs=2)
                resumed_planner = run()
                self.assertEqual(Path(resumed_planner.checkpoint_callback.dirpath), original_planner_dir)
                self.assertEqual(len(resumed_planner.checkpoint_callback.best_k_models), 1)
                self.assertTrue(Path(resumed_planner.checkpoint_callback.best_model_path).is_file())
                planner_path = Path(resumed_planner.checkpoint_callback.last_model_path)
                planner_state = torch.load(planner_path, map_location="cpu", weights_only=False)
                self.assertEqual(resumed_planner.global_step, 2)
                self.assertEqual(planner_state["epoch"], 1)
                self.assertEqual(planner_state["lr_schedulers"][0]["last_epoch"], 2)
                self.assertTrue(all(int(state["step"]) == 2
                                    for state in planner_state["optimizer_states"][0]["state"].values()))
                # Cross-scheme checkpoints must never masquerade as a resume.
                other = "plan2" if mode == "plan1" else "plan1"
                config["model"]["mode"] = config["datamodule"]["mode"] = other
                config["trainer"]["ckpt_path"] = str(planner_path)
                with self.assertRaisesRegex(ValueError, 'Checkpoint mode is incompatible'):
                    run()

    def test_rejects_non_l1_and_removed_options_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory(prefix='fresh-l1-config-', dir='/tmp') as temporary:
            path = Path(temporary) / 'train.yaml'
            for changes, message in (({'action_loss': 'mse'}, 'fixed to l1'),
                                     ({'smooth_l1_beta': 1.}, 'has been removed')):
                config = yaml.safe_load(Path('config/train/pred.yaml').read_text())
                config['model'].update(changes)
                config['trainer']['ckpt_path'] = '/tmp/unused-fresh-checkpoint.ckpt'
                path.write_text(yaml.safe_dump(config))
                with self.subTest(changes=changes), patch.object(
                        sys, 'argv', ['train.py', '--config', str(path)]):
                    with self.assertRaisesRegex(ValueError, message):
                        runpy.run_path('train.py', run_name='__main__')

    def test_mode_configs_and_cli(self):
        for mode in ("pred", "plan1", "plan2"):
            config = yaml.safe_load(Path(f"config/train/{mode}.yaml").read_text())
            self.assertEqual(config["model"]["mode"], mode)
            self.assertEqual(config["datamodule"]["mode"], mode)
            self.assertNotIn("init_ckpt_path", config["trainer"])
            self.assertNotIn("reset_training_state", config["trainer"])
        with patch.object(sys, "argv", ["train.py", "--resume", "unused.ckpt"]):
            with self.assertRaises(SystemExit):
                runpy.run_path("train.py", run_name="__main__")

    def test_checkpoint_path_values(self):
        from train import checkpoint_path
        for value in (None, '', '   '):
            self.assertIsNone(checkpoint_path(value))
        with tempfile.TemporaryDirectory(prefix='fresh-checkpoint-path-', dir='/tmp') as temporary:
            path = Path(temporary) / 'model.ckpt'
            path.touch()
            self.assertEqual(checkpoint_path(str(path)), str(path))
            with self.assertRaises(FileNotFoundError):
                checkpoint_path(str(path.with_name('missing.ckpt')))
        with self.assertRaises(ValueError):
            checkpoint_path(False)

if __name__ == "__main__":
    unittest.main()
