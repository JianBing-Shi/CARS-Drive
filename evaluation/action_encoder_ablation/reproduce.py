import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'tools'))
from collections import defaultdict
import torch
import lightning.pytorch as pl
from datamodules import NuplanDataModule
from datamodules.nuplan_datamodule import collate_events
from model.CARSDirve import CARSDirve
from metrics.action import sample_metrics
from metrics.statistics import aggregate
from evaluate_action_diagnostics import select_samples
pl.seed_everything(1024,workers=True)
m=CARSDirve.load_from_checkpoint('lightning_logs/action_encoder/version_0/checkpoints/best.ckpt',map_location='cpu',strict=True).eval()
data=NuplanDataModule(root='../datasets/nuplan/nuplan-v1.1',dir='mini',mode='pred',history_step=20,future_step=10,num_workers=0,pin_memory=False,persistent_workers=False)
data.setup('validate');ds=data.val_dataset
indices=select_samples(ds,2)
samples=ds.samples
# All referenced files have already been staged for the full evaluation.
scratch=Path('/tmp/fresh_action_encoder_cache')
ds.samples=[scratch/samples[i].relative_to(ds._data_root) for i in indices]
ds._data_root=scratch
rows=defaultdict(list)
keys=('bev_features','bev_valid_mask','timestamps_us','anchor_timestamp_us','reference_ego_to_global','bev_range','history_actions')
with torch.inference_mode():
 for start in range(0,len(ds.samples),4):
  b=collate_events([ds.get(i) for i in range(start,min(start+4,len(ds.samples)))])
  inputs={k:b[k] for k in keys}
  target=b['future_trajectory'].float(); mask=b['future_action_mask'].bool()
  variants={'normal':inputs,'zero_history':{**inputs,'history_actions':torch.zeros_like(inputs['history_actions'])},'no_vision':{**inputs,'bev_features':torch.zeros_like(inputs['bev_features']),'bev_valid_mask':torch.zeros_like(inputs['bev_valid_mask'])}}
  for name,inp in variants.items():
   result=m(**inp,return_details=True,teacher_forcing=False)
   for horizon,a in [('replay',result['actions']),('fixed5',result['fixed_actions'])]:
    scores=sample_metrics(a,target[:,:a.shape[1]],mask[:,:a.shape[1]],m.action_loss,.1)
    for i in range(len(a)):
     row={'recording':b['recording'][i],'anchor_timestamp_us':int(b['anchor_timestamp_us'][i]),'event_type':b['event_type'][i]}
     row.update({k:float(v[i]) if math.isfinite(float(v[i])) else None for k,v in scores.items()});rows[name+'/'+horizon].append(row)
  print('Motion ablation',min(start+4,len(ds.samples)),'/',len(ds.samples),flush=True)
report={'checkpoint':'lightning_logs/action_encoder/version_0/checkpoints/best.ckpt','samples':len(ds.samples),'selection':'seed=1024, 2 per event type, then unique anchor dedup; selected independent of model errors','scope':'small inference ablation, distribution shift; not a retrained no-history or no-vision baseline','scenarios':{k:aggregate(v,[n for n in v[0] if n not in ('recording','anchor_timestamp_us','event_type')]) for k,v in rows.items()}}
output=Path('evaluation/action_encoder_ablation');output.mkdir(parents=True,exist_ok=True);(output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:{n:v['metrics'][n] for n in ('ade_m','fde_m')} for k,v in report['scenarios'].items()},indent=2),flush=True)
