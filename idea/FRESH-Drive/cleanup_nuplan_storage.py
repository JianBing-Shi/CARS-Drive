from pathlib import Path
import shutil

split_root = Path(r'H:\nuplan\dataset\nuplan-v1.1\splits').resolve()
targets = [split_root / name for name in ('train_singapore', 'train_boston', 'train_pittsburgh', 'test')]
protected = {split_root / name for name in ('mini', 'train', 'val', 'mini-processed-plan-train-PlanR1', 'mini-processed-plan-val-PlanR1', 'train-processed-plan-train-PlanR1', 'train-processed-plan-val-PlanR1')}
for target in targets:
    target = target.resolve()
    if target in protected:
        raise RuntimeError(f'protected target: {target}')
    if split_root not in target.parents:
        raise RuntimeError(f'unsafe target: {target}')
    if target.exists():
        print(f'deleting {target}')
        shutil.rmtree(target)
print('cleanup complete')
