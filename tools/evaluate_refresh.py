"""Compare learned refresh against paired acquisition/replanning controls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl
import torch

from datamodules import NuplanDataModule
from evaluate_action_diagnostics import select_samples, stage_inputs
from metrics.refresh_comparison import RefreshComparisonEvaluator
from model.CARSDirve import CARSDirve
from utils import load_config


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024, workers=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/val/plan_comparison.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    trainer_config = dict(config['trainer'])
    model = CARSDirve.load_from_checkpoint(trainer_config.pop('ckpt_path'), map_location='cpu', strict=True, action_loss='l1')
    data = NuplanDataModule(**config['datamodule'])
    data.setup('validate')
    dataset = data.val_dataset
    total = len(dataset)
    config['evaluation']['source_manifest'] = str(dataset.processed_file_names_path)
    config['evaluation']['manifest_windows'] = total
    indices = select_samples(dataset, config['evaluation']['samples_per_event'])
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset._processed_file_names = [str(p.relative_to(dataset.event_dir)) for p in dataset.samples]
    if not len(dataset):
        raise ValueError('No validation anchors selected')
    print(f'Selected {len(dataset)} unique anchors from {total} refresh validation windows', flush=True)
    if config['evaluation'].get('scratch_dir'):
        stage_inputs(dataset, config['evaluation']['scratch_dir'])
    evaluator = RefreshComparisonEvaluator(model, dataset, config, args.config, len(dataset))
    pl.Trainer(**trainer_config).validate(evaluator, dataloaders=data.val_dataloader(), verbose=False)
