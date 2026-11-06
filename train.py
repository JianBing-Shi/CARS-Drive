"""Resume training state, or explicitly start a new action schedule from weights."""

from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import torch
from utils import load_config
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datamodules import NuplanDataModule
from model.FRESHDrive import FRESHDrive


def validate_action_components(source, target):
    # Shape-compatible changes (e.g. attention heads) can still alter semantics.
    keys = ('architecture_version', 'feature_dim', 'hidden_dim', 'spatial_grid',
            'heads', 'layers', 'decoder_layers', 'dropout', 'future_action', 'history_features')
    for key in keys:
        actual, expected = source.hparams[key], target.hparams[key]
        if key == 'spatial_grid':
            actual, expected = tuple(actual), tuple(expected)
        if actual != expected:
            raise ValueError(f'Action component setting {key} differs from checkpoint; preserve the trained action network')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser(description="Train FRESH-Drive model")
    parser.add_argument('--config', type=Path, default=Path('config/train/action.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    ckpt_path = config['trainer']['ckpt_path']
    reset_training_state = config['trainer'].get('reset_training_state', False)
    if not isinstance(reset_training_state, bool):
        raise ValueError('reset_training_state must be a boolean')
    if reset_training_state and (not ckpt_path or config['model']['mode'] != 'action'):
        raise ValueError('reset_training_state requires an action checkpoint and action mode')
    if config['model'].get('action_loss', 'l1') != 'l1':
        raise ValueError('action_loss is fixed to l1')
    if 'smooth_l1_beta' in config['model']:
        raise ValueError('smooth_l1_beta has been removed; action supervision uses only L1')

    if ckpt_path:
        print(f"Loading model from checkpoint: {ckpt_path}")
        source = FRESHDrive.load_from_checkpoint(ckpt_path, map_location='cpu', action_loss='l1')
        configured_version = config['model'].get('architecture_version', 1)
        if source.hparams.get('architecture_version', 1) != configured_version:
            raise ValueError('Checkpoint architecture differs from configuration; start a new action run with ckpt_path: null')
        if source.mode == config['model']['mode']:
            configured_refresh_version = config['model'].get('refresh_version', 1)
            if source.refresh_version != configured_refresh_version:
                raise ValueError('Refresh architecture differs; initialize the new binary gate from an action checkpoint')
            if source.mode == 'refresh' and source.refresh_version >= 2:
                configured = FRESHDrive(**config['model'])
                if source.checkpoint_refresh_protocol != configured.refresh_protocol():
                    raise ValueError('Refresh protocol differs from checkpoint; start a new refresh run from action weights')
            defaults = {'action_loss_version': 2, 'action_loss': 'l1', 'xy_loss_weight': 1.,
                        'yaw_weight': 1., 'trajectory_loss_weight': 0., 'heading_loss_weight': 0.}
            expected_loss = {key: config['model'].get(key, default) for key, default in defaults.items()}
            if source.checkpoint_loss_config != expected_loss:
                raise ValueError('Checkpoint loss settings differ from configuration; '
                                 'start the fixed L1 action run with ckpt_path: null')
            if reset_training_state:
                model = FRESHDrive(**config['model'])
                validate_action_components(source, model)
                model.load_state_dict(source.state_dict(), strict=True)
                ckpt_path = None
                print('Loaded action weights only; starting a new optimizer, learning-rate schedule, '
                      'epoch counter and checkpoint ranking.', flush=True)
            else:
                schedule_keys = ('learning_rate', 'weight_decay', 'warmup_epochs', 'T_max')
                if any(config['model'].get(key, source.hparams[key]) != source.hparams[key]
                       for key in schedule_keys):
                    raise ValueError('Optimizer or schedule settings differ from checkpoint; '
                                     'action training needs reset_training_state: true to start a new schedule; '
                                     'ordinary resume must preserve these settings')
                model = source
        elif source.mode == 'action' and config['model']['mode'] == 'refresh':
            model = FRESHDrive(**config['model'])
            validate_action_components(source, model)
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

    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])
    logger_config = dict(config['trainer']['csv_logger'])
    checkpoint_config = dict(config['trainer']['ckpt'])
    if ckpt_path:
        # Keep the original checkpoint directory: Lightning only restores the
        # full top-k ranking when ModelCheckpoint.dirpath has not changed.
        checkpoint_dir = Path(ckpt_path).resolve().parent
        checkpoint_config['dirpath'] = str(checkpoint_dir)
        # CSVLogger starts a new segment so existing CSV history is preserved.
    csv_logger = CSVLogger(**logger_config)
    model_checkpoint = ModelCheckpoint(**checkpoint_config)
    trainer_config = {key: value for key, value in config['trainer'].items()
                      if key not in {'ckpt', 'csv_logger', 'lr_monitor', 'ckpt_path', 'reset_training_state'}}
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=[model_checkpoint, lr_monitor],           # 回调函数列表，包括模型保存和学习率监控
        logger=csv_logger                                   # 日志记录器       
    )
    
    print(f'Mode: {model.mode} | run: {csv_logger.log_dir}', flush=True)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f'Best checkpoint ({model_checkpoint.monitor}, {model_checkpoint.mode}): '
          f'{model_checkpoint.best_model_path} | score={model_checkpoint.best_model_score}', flush=True)
    print(f'Last checkpoint: {model_checkpoint.last_model_path}', flush=True)

