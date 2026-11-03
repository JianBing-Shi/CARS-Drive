"""Lightning training entry point; resume restores optimizer and loop state."""

from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import torch
from utils import load_config
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datamodules import NuplanDataModule
from model.FRESHDrive import FRESHDrive


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser(description="Train FRESH-Drive model")
    parser.add_argument('--config', type=Path, default=Path('config/train/action.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    ckpt_path = config['trainer']['ckpt_path']

    if ckpt_path:
        print(f"Loading model from checkpoint: {ckpt_path}")
        source = FRESHDrive.load_from_checkpoint(ckpt_path, map_location='cpu')
        if source.mode == config['model']['mode']:
            model = source
        elif source.mode == 'action' and config['model']['mode'] == 'refresh':
            model = FRESHDrive(**config['model'])
            for name in ('tokenizer', 'memory', 'decoder'):
                getattr(model, name).load_state_dict(getattr(source, name).state_dict())
            ckpt_path = None
        else:
            raise ValueError('Checkpoint mode is incompatible with the configured mode')
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = FRESHDrive(**config['model'])
        if model.mode == 'refresh':
            raise ValueError('Refresh requires a trained action checkpoint')

    datamodule = NuplanDataModule(**config['datamodule'])

    model_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])
    
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])
    csv_logger = CSVLogger(**config['trainer']['csv_logger'])
    trainer_config = {key: value for key, value in config['trainer'].items()
                      if key not in {'ckpt', 'csv_logger', 'lr_monitor', 'ckpt_path'}}
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=[model_checkpoint, lr_monitor],           # 回调函数列表，包括模型保存和学习率监控
        logger=csv_logger                                   # 日志记录器       
    )
    
    print(f'Mode: {model.mode} | run: {csv_logger.log_dir}', flush=True)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f'Last checkpoint: {model_checkpoint.last_model_path}', flush=True)

