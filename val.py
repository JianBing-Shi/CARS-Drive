"""Validate pred, plan1 or plan2 checkpoints with YAML-configured Lightning."""
import argparse

import lightning.pytorch as pl
import torch
from torch.utils.data import Subset

from datamodules import NuplanDataModule
from model.CARSDirve import CARSDirve
from metrics.evaluator import ActionEvaluator
from metrics.binary_refresh_evaluator import BinaryRefreshEvaluator
from utils import load_config


def validate_evaluation_mode(model, config):
    mode = config['datamodule'].get('mode', 'pred')
    if model.mode != mode or config.get('mode', mode) != mode:
        raise ValueError('Validation model and datamodule modes must agree (pred, plan1, or plan2)')
    if mode not in {'pred', 'plan1', 'plan2'}:
        raise ValueError('Legacy plan checkpoints require their original validation code')


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
    validate_evaluation_mode(model, config)
    trainer = pl.Trainer(**trainer_config)

    data = NuplanDataModule(**config["datamodule"])
    data.setup("validate")
    dataset = data.val_dataset
    data.val_dataset = Subset(dataset, range(len(dataset))[:config["evaluation"]["max_samples"]])
    count = len(data.val_dataset)
    if not count:
        raise ValueError('No validation anchors selected')

    if model.mode == "pred":
        evaluator = ActionEvaluator(model, dataset, config, args.config, count)
    elif model.mode in {"plan1", "plan2"}:
        evaluator = BinaryRefreshEvaluator(model, dataset, config, args.config, count)
    trainer.validate(evaluator, dataloaders=data.val_dataloader())
