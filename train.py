"""Initialize from an empty ckpt_path, or fully resume a training checkpoint."""

from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
import torch
from utils import load_config
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datamodules import NuplanDataModule
from model.CARSDirve import CARSDirve


def checkpoint_path(value):
    """Empty YAML values start a run; a supplied path must be a real file."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('trainer.ckpt_path must be a file path, null, or an empty string')
    value = value.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint file does not exist: {path}')
    return str(path)


def validate_action_components(source, target):
    # Shape-compatible changes (e.g. attention heads) can still alter semantics.
    keys = ('architecture_version', 'feature_dim', 'hidden_dim', 'spatial_grid',
            'heads', 'layers', 'decoder_layers', 'dropout', 'future_step', 'history_step')
    for key in keys:
        actual, expected = source.hparams[key], target.hparams[key]
        if key == 'spatial_grid':
            actual, expected = tuple(actual), tuple(expected)
        if actual != expected:
            raise ValueError(f'Action component setting {key} differs from checkpoint; preserve the trained action network')


def validate_stage_config(config):
    """Model and data must use the same, explicit research protocol."""
    mode = config['model']['mode']
    if mode not in {'pred', 'plan1', 'plan2'}:
        raise ValueError('model.mode must be pred, plan1, or plan2; legacy plan checkpoints require their original code')
    if config['datamodule'].get('mode', 'pred') != mode:
        raise ValueError('Model and datamodule modes must agree')
    for key in ('history_step', 'future_step'):
        if config['model'].get(key, 20 if key == 'history_step' else 10) != config['datamodule'].get(
                key, 20 if key == 'history_step' else 10):
            raise ValueError(f'Model and datamodule {key} must agree')


def initialize_model(model_config, path):
    """Return a model and full-resume path, or start planning from frozen pred weights.

    A pred checkpoint initializes each planning experiment independently.
    Optimizer, epoch and schedule state are restored only for the same mode.
    """
    target_mode = model_config['mode']
    if target_mode not in {'pred', 'plan1', 'plan2'}:
        raise ValueError('model.mode must be pred, plan1, or plan2')
    if path is None:
        if target_mode != 'pred':
            raise ValueError('plan1 and plan2 require a trained pred checkpoint')
        print('No checkpoint path provided, initializing a new model.')
        return CARSDirve(**model_config), None

    print(f'Loading model from checkpoint: {path}')
    source = CARSDirve.load_from_checkpoint(path, map_location='cpu', action_loss='l1')
    configured_version = model_config.get('architecture_version', 1)
    if source.hparams.get('architecture_version', 1) != configured_version:
        raise ValueError('Checkpoint architecture differs from configuration; preserve the trained pred architecture')
    if source.mode != target_mode and not (source.mode == 'pred' and target_mode in {'plan1', 'plan2'}):
        raise ValueError('Checkpoint mode is incompatible: plan1 and plan2 must initialize independently '
                         'from pred; full resume requires the same mode')
    configured = CARSDirve(**model_config)
    validate_action_components(source, configured)
    if source.mode == 'pred' and target_mode in {'plan1', 'plan2'}:
        for name in ('tokenizer', 'memory', 'action_encoder', 'decoder'):
            getattr(configured, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
        configured.freeze_pred_model()
        print(f'Initialized {target_mode} from frozen pred weights; starting a new optimizer and schedule.')
        return configured, None

    if source.cars_adapter_version != configured.cars_adapter_version:
        raise ValueError('CARS architecture differs; initialize a new planning run from a pred checkpoint')
    if source.mode in {'plan1', 'plan2'} and source.checkpoint_refresh_protocol != configured.cars_adapter_protocol():
        raise ValueError('CARS protocol differs from checkpoint; initialize a new planning run from pred weights')
    defaults = {'action_loss_version': 2, 'action_loss': 'l1', 'xy_loss_weight': 1.,
                'yaw_weight': 1., 'trajectory_loss_weight': 0., 'heading_loss_weight': 0.}
    expected_loss = {key: model_config.get(key, default) for key, default in defaults.items()}
    if source.checkpoint_loss_config != expected_loss:
        raise ValueError('Checkpoint loss settings differ from configuration; resume must preserve the training objective')
    schedule_keys = ('learning_rate', 'weight_decay', 'warmup_epochs', 'T_max', 'lr_min_ratio')
    if any(model_config.get(key, source.hparams[key]) != source.hparams[key] for key in schedule_keys):
        raise ValueError('Optimizer or schedule settings differ from checkpoint; resume must preserve these settings')
    for key, expected in model_config.items():
        actual = source.hparams.get(key)
        if key in {'spatial_grid', 'macro_grid'}:
            actual, expected = tuple(actual), tuple(expected)
        if actual != expected:
            raise ValueError(f'Checkpoint setting {key} differs from configuration; full resume must preserve model settings')
    return source, path


def _multiple_devices(devices) -> bool:
    """Return whether a concrete YAML device setting requests >1 device."""
    if isinstance(devices, int):
        return devices > 1 or devices == -1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return False


def prepare_trainer_config(config: dict) -> dict:
    """Prepare Lightning settings while keeping the default single-card run.

    Lightning injects a distributed sampler for DDP.  A user can therefore
    change only ``trainer.devices`` to ``2`` or ``[0, 1]`` in YAML; this entry
    point selects DDP for multiple requested GPUs (including ``-1``), and
    leaves strategy selection to Lightning for ``devices: auto``.
    """
    trainer_config = {key: value for key, value in config['trainer'].items()
                      if key not in {'ckpt', 'csv_logger', 'lr_monitor', 'ckpt_path'}}
    accelerator = trainer_config.get('accelerator', 'auto')
    strategy = trainer_config.get('strategy', 'auto')
    devices = trainer_config.get('devices', 1)
    if strategy == 'auto' and trainer_config.get('num_nodes', 1) > 1:
        trainer_config['strategy'] = 'ddp'
    elif strategy == 'auto' and _multiple_devices(devices) and accelerator in {'gpu', 'cuda'}:
        trainer_config['strategy'] = 'ddp'
    if trainer_config.get('num_nodes', 1) < 1:
        raise ValueError('trainer.num_nodes must be at least 1')
    return trainer_config


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser(description="Train CARS-Drive model")
    parser.add_argument('--config', type=Path, default=Path('config/train/pred.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    validate_stage_config(config)
    ckpt_path = config['trainer'].get('ckpt_path')
    if 'reset_training_state' in config['trainer']:
        raise ValueError('reset_training_state has been removed; use ckpt_path only')
    if config['model'].get('action_loss', 'l1') != 'l1':
        raise ValueError('action_loss is fixed to l1')
    if 'smooth_l1_beta' in config['model']:
        raise ValueError('smooth_l1_beta has been removed; action supervision uses only L1')
    ckpt_path = checkpoint_path(ckpt_path)

    model, ckpt_path = initialize_model(config['model'], ckpt_path)

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
    trainer_config = prepare_trainer_config(config)
    print('Trainer devices:', trainer_config.get('devices', 1),
          '| strategy:', trainer_config.get('strategy', 'auto'),
          '| nodes:', trainer_config.get('num_nodes', 1), flush=True)
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

