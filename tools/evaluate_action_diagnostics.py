"""Run paired visual, stale-input and fixed-context checkpoint diagnostics."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl
import torch

from datamodules import NuplanDataModule
from metrics.action_diagnostics import ActionDiagnosticsEvaluator
from model.CARSDirve import CARSDirve
from utils import load_config


def stage_inputs(dataset, directory):
    """Copy only referenced BEVs to local scratch, preserving source metadata."""
    directory=Path(directory)
    if directory.resolve()==dataset._data_root:
        raise ValueError('Scratch directory must differ from the source dataset')
    names=set()
    staged_samples=[]
    for path in dataset.samples:
        event=torch.load(path,map_location='cpu',weights_only=False)
        names.update(event['bev_paths'])
        staged=directory/path.relative_to(dataset._data_root)
        staged.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,staged)
        staged_samples.append(staged.resolve())
    def copy(name):
        destination=directory/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        source=dataset._data_root/name
        if not destination.exists() or destination.stat().st_size!=source.stat().st_size:
            shutil.copyfile(source,destination)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(copy,sorted(names)),1):
            if i%50==0 or i==len(names):
                print(f'Scratch BEVs: {i}/{len(names)}',flush=True)
    dataset._data_root=directory.resolve()
    dataset.samples=staged_samples


def select_samples(dataset, per_event, seed=1024):
    """Sample each label before global dedup; never select using model errors."""
    groups = defaultdict(list)
    if per_event is not None and per_event <= 0:
        raise ValueError('samples_per_event must be positive or null')
    for i,path in enumerate(dataset.samples):
        groups[path.parent.name].append(i)
    rng = random.Random(seed)
    selected = []
    for label, indices in sorted(groups.items()):
        selected.extend(sorted(rng.sample(indices,min(per_event,len(indices)))) if per_event else indices)
    seen = set()
    indices = []
    for i in selected:
        path = dataset.samples[i]
        key = (path.parent.parent.name,path.stem)
        if key not in seen:
            indices.append(i); seen.add(key)
    return indices


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024,workers=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/val/action_diagnostics.yaml')
    args=parser.parse_args()
    config=load_config(args.config)
    trainer_config=dict(config['trainer'])
    model=CARSDirve.load_from_checkpoint(trainer_config.pop('ckpt_path'),map_location='cpu',strict=True,action_loss='l1')
    data=NuplanDataModule(**config['datamodule']); data.setup('validate')
    dataset=data.val_dataset
    indices=select_samples(dataset,config['evaluation']['samples_per_event'])
    dataset.samples=[dataset.samples[i] for i in indices]
    dataset._processed_file_names=[str(p.relative_to(dataset.event_dir)) for p in dataset.samples]
    print(f'Selected {len(dataset)} unique anchors',flush=True)
    if config['evaluation'].get('scratch_dir'):
        stage_inputs(dataset,config['evaluation']['scratch_dir'])
    evaluator=ActionDiagnosticsEvaluator(model,dataset,config,args.config,len(dataset))
    pl.Trainer(**trainer_config).validate(evaluator,dataloaders=data.val_dataloader(),verbose=False)
