"""Analyze existing action predictions by label/log; optionally audit cached motion."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def analyze(report_path, output, motion_groups=False):
    report=json.loads(report_path.read_text())
    rows=list(csv.DictReader(report_path.with_name('per_sample.csv').open()))
    unique={}
    for row in rows:
        unique.setdefault((row['recording'],row['anchor_timestamp_us']),row)
    groups=defaultdict(list)
    for row in unique.values():
        labels=['all','recording:'+row['recording']]
        if motion_groups:
            event=torch.load(Path(report['run']['data_root'])/'preprocessed_dataset/events'/row['sample'],map_location='cpu')
            past=event['trajectory']['past_trajectory'].float()
            speed=float(past[-1,:2].norm()/.1)
            yaw_rate=abs(float(past[-1,2]))/.1
            labels+=['speed:'+('stationary_<0.5' if speed<.5 else 'slow_0.5-2' if speed<2 else 'medium_2-5' if speed<5 else 'fast_>=5'),
                     'turn:'+('turning_>=0.1rad_s' if yaw_rate>=.1 else 'near_straight')]
            truth=event['future_trajectory'].float()
            step_speed=torch.cat((truth[:1,:2],truth[1:,:2]-truth[:-1,:2])).norm(dim=-1)/.1
            change=float(step_speed[-1]-speed)
            labels+=['expert_future_speed_change:'+('braking_<-.5m_s' if change<-.5 else
                       'accelerating_>.5m_s' if change>.5 else 'steady')]
        for label in labels:
            groups[label].append(row)
    # Each label retains its own anchors, including multi-label scenes.
    event_seen=defaultdict(set)
    for row in rows:
        key='event:'+row['event_type']
        identity=(row['recording'],row['anchor_timestamp_us'])
        if identity not in event_seen[key]:
            groups[key].append(row)
            event_seen[key].add(identity)
    result={}
    for key,items in groups.items():
        data={'samples':len(items),'recordings':len({i['recording'] for i in items})}
        for metric in ['ade_full_horizon_m','fde_m','heading_mae_rad','action_dx_mae_m','action_dy_mae_m','action_dyaw_mae_rad']:
            for name in ['model','constant_action']:
                values=[float(i[name+'_'+metric]) for i in items if i[name+'_'+metric]!='']
                data[name+'_'+metric]=math.fsum(values)/len(values) if values else None
        result[key]=data
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'Analyzed {len(unique)} anchors: {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('evaluation/action/metrics.json'))
    parser.add_argument('--output',type=Path,default=Path('evaluation/action_diagnostics/existing_action_analysis.json'))
    parser.add_argument('--motion-groups',action='store_true')
    args=parser.parse_args()
    analyze(args.report,args.output,args.motion_groups)
