"""Validate action or refresh checkpoints with YAML-configured Lightning."""
import argparse

import lightning.pytorch as pl
import torch
from torch.utils.data import Subset

from datamodules import NuplanDataModule
from model.CARSDirve import CARSDirve
from metrics.evaluator import ActionEvaluator, RefreshEvaluator
from metrics.binary_refresh_evaluator import BinaryRefreshEvaluator
from utils import load_config


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024, workers=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default='config/val/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    trainer_config = dict(config["trainer"])
    model = CARSDirve.load_from_checkpoint(
        trainer_config.pop("ckpt_path"), map_location="cpu", strict=True, action_loss="l1")
    trainer = pl.Trainer(**trainer_config)

    data = NuplanDataModule(**config["datamodule"])
    data.setup("validate")
    dataset = data.val_dataset
    data.val_dataset = Subset(dataset, range(len(dataset))[:config["evaluation"]["max_samples"]])
    count = len(data.val_dataset)

    if model.mode == "pred":
        evaluator = ActionEvaluator(model, dataset, config, args.config, count)
    elif model.mode == "plan":
        evaluator_class = BinaryRefreshEvaluator if model.cars_adapter_version >= 2 else RefreshEvaluator
        evaluator = evaluator_class(model, dataset, config, args.config, count)
    trainer.validate(evaluator, dataloaders=data.val_dataloader())
