"""Shared YAML parameters must reach the data and model constructors."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import torch
import yaml

from datamodules import NuplanDataModule
from model import CARSDirve


class SharedConfigTests(unittest.TestCase):
    def test_validation_yaml_and_val_only_setup(self):
        for mode in ("pred", "plan1", "plan2"):
            with self.subTest(mode=mode), patch(
                    "datamodules.nuplan_datamodule.NuplanEventDataset") as dataset:
                config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                         f"config/val/{mode}.yaml").read_text())
                module = NuplanDataModule(**config["datamodule"])
                self.assertEqual(config["mode"], module.mode)
                module.setup("validate")
                dataset.assert_called_once()
                self.assertEqual(dataset.call_args.kwargs["split"], "val")
                self.assertFalse(hasattr(module, "train_dataset"))

    def test_shared_yaml_arguments_and_model_modes(self):
        config_dir = Path(__file__).resolve().parents[1] / "config/train"
        for mode in ("pred", "plan1", "plan2"):
            with self.subTest(mode=mode):
                config = yaml.safe_load((config_dir / f"{mode}.yaml").read_text())
                inspect.signature(NuplanDataModule).bind(**config["datamodule"])
                inspect.signature(CARSDirve).bind(**config["model"])
                for key in ("mode", "history_step", "future_step"):
                    self.assertEqual(config["datamodule"][key], config["model"][key])
                settings = dict(config["model"], hidden_dim=12, heads=4, layers=1)
                model = CARSDirve(**settings)
                self.assertEqual(model.mode, mode)
                if mode in {"plan1", "plan2"}:
                    self.assertEqual((model.history_step, model.future_step, model.cache_anchor_step,
                                      model.plan_horizon, model.cars_adapter_steps), (20, 10, 25, 10, 10))
                model.train()
                if mode in {"plan1", "plan2"}:
                    model.freeze_pred_model()
                for module in (model.tokenizer, model.memory, model.action_encoder, model.decoder):
                    self.assertEqual(module.training, mode == "pred")
                    self.assertTrue(all(p.requires_grad == (mode == "pred")
                                        for p in module.parameters()))

    def test_dataset_receives_nondefault_window_and_ratio(self):
        for mode in ("pred", "plan1", "plan2"):
            with self.subTest(mode=mode), patch(
                    "datamodules.nuplan_datamodule.NuplanEventDataset") as dataset:
                history, future = (3, 2) if mode == "pred" else (20, 10)
                module = NuplanDataModule(
                    root="/tmp/unused-nuplan", mode=mode, history_step=history,
                    future_step=future, ratio=.25, num_workers=0)
                module.setup("fit")
                self.assertEqual(dataset.call_count, 2)
                self.assertEqual([call.kwargs["split"] for call in dataset.call_args_list],
                                 ["train", "val"])
                for call in dataset.call_args_list:
                    self.assertEqual(call.kwargs["history_step"], history)
                    self.assertEqual(call.kwargs["future_step"], future)
                    self.assertEqual(call.kwargs["ratio"], .25)
                    self.assertFalse(call.kwargs["preprocess"])

    def test_mode_checkpoint_loads(self):
        for mode in ("pred", "plan1", "plan2"):
            with self.subTest(mode=mode):
                model = CARSDirve(hidden_dim=12, heads=4, layers=1, mode=mode,
                                  history_step=20, future_step=10,
                                  cache_anchor_step=20 if mode == "pred" else 25,
                                  plan_horizon=5 if mode == "pred" else 10)
                checkpoint = {"state_dict": model.state_dict(),
                              "hyper_parameters": dict(model.hparams),
                              "pytorch-lightning_version": pl.__version__}
                model.on_save_checkpoint(checkpoint)
                with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
                    path = Path(temporary) / f"{mode}.ckpt"
                    torch.save(checkpoint, path)
                    restored = CARSDirve.load_from_checkpoint(path, map_location="cpu")
                self.assertEqual(restored.mode, mode)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], tensor)
                self.assertTrue(all(p.requires_grad == (mode == "pred")
                                    for p in restored.decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
