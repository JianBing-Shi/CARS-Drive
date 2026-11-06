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
    def test_train_and_resume_both_stages(self):
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
            poses = torch.zeros(31, 3)
            poses[:, 0] = (torch.arange(31) - 20) * .1
            event.update(anchor_index=20, anchor_timestamp_us=2_000_000,
                         timestamps_us=torch.arange(31) * 100_000, bev_paths=[bev.as_posix()] * 31,
                         future_trajectory=poses[21:], future_actions=torch.tensor([[.1, 0., 0.]] * 10),
                         future_action_mask=torch.ones(10, dtype=torch.bool),
                         trajectory={"past_trajectory": poses[:20], "anchor_trajectory": poses[20],
                                     "future_expert_trajectory": poses[21:], "trajectory_mask": torch.ones(31, dtype=torch.bool)})
            for split in ("train", "val"):
                name = Path(split) / "changing_lane/sample.pt"
                destination = cache / "events" / name
                destination.parent.mkdir(parents=True)
                torch.save(event, destination)
                torch.save([name.as_posix()], cache / f"mini-{split}.pt")

            config = yaml.safe_load(Path("config/train/action.yaml").read_text())
            config["datamodule"].update(
                root=str(root), dir="mini", history_features=20, future_action=10,
                train_batch_size=1, val_batch_size=1, num_workers=0, pin_memory=False)
            config["model"].update(
                history_features=20, future_action=10, feature_dim=4, hidden_dim=8,
                heads=2, layers=1, spatial_grid=[3, 3], macro_grid=[3, 3],
                dropout=0, anchor_dropout=0, warmup_epochs=1, T_max=4)
            config["trainer"].update(
                accelerator="cpu", max_epochs=1, num_sanity_val_steps=0,
                ckpt_path=None, reset_training_state=False,
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
            self.assertEqual(action_state["hyper_parameters"]["mode"], "action")
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

            # A finished schedule must not silently resume at zero LR when
            # the YAML requests a new plan. Start a separate weights-only run.
            original_action_path = action_path
            exhausted_state = torch.load(action_path, map_location="cpu", weights_only=False)
            for group in exhausted_state['optimizer_states'][0]['param_groups']:
                group['lr'] = 0.
            exhausted_state['lr_schedulers'][0]['last_epoch'] = 4
            exhausted_path = root / 'exhausted.ckpt'
            torch.save(exhausted_state, exhausted_path)
            config['trainer'].update(ckpt_path=str(exhausted_path), max_epochs=1)
            config['model'].update(learning_rate=1e-5, warmup_epochs=1, T_max=3)
            with self.assertRaisesRegex(ValueError, 'schedule settings differ'):
                run()
            config['trainer']['reset_training_state'] = True
            config['trainer']['csv_logger']['name'] = 'action_extend'
            extended = run()
            extended_path = Path(extended.checkpoint_callback.last_model_path)
            extended_state = torch.load(extended_path, map_location='cpu', weights_only=False)
            self.assertNotEqual(extended_path.parent, original_action_dir)
            self.assertEqual(extended_state['global_step'], 1)
            self.assertEqual(extended_state['epoch'], 0)
            self.assertEqual(extended_state['hyper_parameters']['learning_rate'], 1e-5)
            self.assertEqual(extended_state['hyper_parameters']['T_max'], 3)
            self.assertEqual(extended_state['lr_schedulers'][0]['last_epoch'], 1)
            self.assertTrue(all(group['lr'] > 0
                                for group in extended_state['optimizer_states'][0]['param_groups']))
            self.assertTrue(all(int(state['step']) == 1
                                for state in extended_state['optimizer_states'][0]['state'].values()))
            self.assertTrue(any(not torch.equal(value, exhausted_state['state_dict'][name])
                                for name, value in extended_state['state_dict'].items()
                                if name.startswith('decoder.')))
            config['trainer'].update(ckpt_path=str(extended_path), reset_training_state=False,
                                     max_epochs=2)
            continued = run()
            self.assertEqual(continued.global_step, 2)
            self.assertEqual(Path(continued.checkpoint_callback.dirpath), extended_path.parent)

            # Refresh remains independent: use the original action checkpoint.
            action_path = original_action_path
            config['model'].update(learning_rate=1e-5, warmup_epochs=1, T_max=4)
            config["model"]["mode"] = config["datamodule"]["mode"] = "refresh"
            config["trainer"]["ckpt"]["monitor"] = "val/loss"
            config["trainer"]["csv_logger"]["name"] = "refresh"
            config["trainer"]["max_epochs"] = 1
            config["trainer"]["ckpt_path"] = str(action_path)
            refresh = run()
            self.assertEqual(refresh.global_step, 1)
            self.assertEqual(refresh.lightning_module.mode, "refresh")
            for name, value in refresh.lightning_module.state_dict().items():
                if name.startswith(("tokenizer.", "memory.", "decoder.")):
                    torch.testing.assert_close(value, action_state["state_dict"][name])
            self.assertTrue(all(not p.requires_grad
                                for p in refresh.lightning_module.decoder.parameters()))
            refresh_path = Path(refresh.checkpoint_callback.last_model_path)
            original_refresh_dir = refresh_path.parent
            config["trainer"].update(ckpt_path=str(refresh_path), max_epochs=2)
            resumed_refresh = run()
            self.assertEqual(Path(resumed_refresh.checkpoint_callback.dirpath), original_refresh_dir)
            self.assertEqual(len(resumed_refresh.checkpoint_callback.best_k_models), 1)
            self.assertTrue(Path(resumed_refresh.checkpoint_callback.best_model_path).is_file())
            refresh_path = Path(resumed_refresh.checkpoint_callback.last_model_path)
            refresh_state = torch.load(refresh_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed_refresh.global_step, 2)
            self.assertEqual(refresh_state["epoch"], 1)
            self.assertEqual(refresh_state["lr_schedulers"][0]["last_epoch"], 2)
            self.assertTrue(all(int(state["step"]) == 2
                                for state in refresh_state["optimizer_states"][0]["state"].values()))

    def test_rejects_non_l1_and_removed_options_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory(prefix='fresh-l1-config-', dir='/tmp') as temporary:
            path = Path(temporary) / 'train.yaml'
            for changes, message in (({'action_loss': 'mse'}, 'fixed to l1'),
                                     ({'smooth_l1_beta': 1.}, 'has been removed')):
                config = yaml.safe_load(Path('config/train/action.yaml').read_text())
                config['model'].update(changes)
                config['trainer']['ckpt_path'] = '/tmp/unused-fresh-checkpoint.ckpt'
                path.write_text(yaml.safe_dump(config))
                with self.subTest(changes=changes), patch.object(
                        sys, 'argv', ['train.py', '--config', str(path)]):
                    with self.assertRaisesRegex(ValueError, message):
                        runpy.run_path('train.py', run_name='__main__')

    def test_mode_configs_and_cli(self):
        for mode in ("action", "refresh"):
            config = yaml.safe_load(Path(f"config/train/{mode}.yaml").read_text())
            self.assertEqual(config["model"]["mode"], mode)
            self.assertEqual(config["datamodule"]["mode"], mode)
            self.assertNotIn("init_ckpt_path", config["trainer"])
        with patch.object(sys, "argv", ["train.py", "--resume", "unused.ckpt"]):
            with self.assertRaises(SystemExit):
                runpy.run_path("train.py", run_name="__main__")

    def test_reset_requires_action_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='fresh-reset-config-', dir='/tmp') as temporary:
            path = Path(temporary) / 'train.yaml'
            for mode, checkpoint in (('action', None), ('refresh', '/tmp/unused.ckpt')):
                config = yaml.safe_load(Path(f'config/train/{mode}.yaml').read_text())
                config['trainer'].update(ckpt_path=checkpoint, reset_training_state=True)
                path.write_text(yaml.safe_dump(config))
                with self.subTest(mode=mode), patch.object(
                        sys, 'argv', ['train.py', '--config', str(path)]):
                    with self.assertRaisesRegex(ValueError, 'requires an action checkpoint'):
                        runpy.run_path('train.py', run_name='__main__')


if __name__ == "__main__":
    unittest.main()
