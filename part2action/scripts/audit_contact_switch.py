"""Audit demonstration label validity and exact model switch eligibility on CPU."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
from audit_contact_labels import _segments
from data.geometry import near_contact_mask, tcp_position_numpy
from data.targets import classify_skill_instruction, _detect_contact_step, derive_contact_xyz, derive_skill_contact_target, _part_points_world


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-demos',type=int,default=32)
    p.add_argument('--threshold',type=float,default=.03)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    summary=defaultdict(lambda:dict(segments=0,frames=0,valid_frames=0,eligible_frames=0,segments_ever_eligible=0))
    count=0
    with (a.out/'segments.jsonl').open('w') as output:
        for obj in ['bottle','mug','pliers','scissors']:
            with h5py.File(a.data_root/f'{obj}.hdf5','r') as f:
                keys=sorted(f['data'],key=lambda k:int(k.split('_')[-1]))[:a.max_demos]
                for key in keys:
                    demo=f['data'][key]; obs=demo['obs']; actions=demo['actions'][:]; poses=obs['tcp_pose'][:]; cloud=obs['agentview_part_pcd'][:]
                    for start,end,instruction in _segments(demo['skill_instructions'][:]):
                        kind=classify_skill_instruction(instruction)
                        event=derive_skill_contact_target(actions=actions,part_pcd=cloud,tcp_pose=poses,skill_kind=kind,segment_start=start,segment_end=end,current_t=start)
                        event_cache={k:event[k] for k in ['contact_t','method','reason','tcp_to_part_distance_m']}
                        if event_cache['contact_t']>=0: event_cache['reason']=''
                        record=dict(object=obj,demo=key,instruction=instruction,kind=kind,start=start,end=end,skill_event=int(event['contact_t']),methods={})
                        surface=[]
                        for mode in ['legacy','skill_specific']:
                            contacts=[]; valid=[]; events=[]
                            for t in range(start,end+1):
                                if mode=='legacy':
                                    et=_detect_contact_step(actions,t); xyz,v=derive_contact_xyz(cloud,poses,et,current_t=t)
                                else:
                                    result=derive_skill_contact_target(actions=actions,part_pcd=cloud,tcp_pose=poses,skill_kind=kind,segment_start=start,segment_end=end,current_t=t,contact_event=event_cache)
                                    et=result['contact_t'];xyz=result['contact_xyz'];v=result['valid']
                                contacts.append(xyz);valid.append(bool(v));events.append(int(et))
                                if mode=='legacy':
                                    points=_part_points_world(cloud,t)
                                    surface.append(float(np.linalg.norm(points-tcp_position_numpy(poses[t]),axis=-1).min())*100 if len(points) else None)
                            near=near_contact_mask(torch.as_tensor(poses[start:end+1]),torch.as_tensor(np.asarray(contacts)),a.threshold,torch.as_tensor(valid)).numpy()
                            distances=np.linalg.norm(np.asarray(contacts)-tcp_position_numpy(poses[start:end+1]),axis=-1)*100
                            eligible=[start+i for i,v in enumerate(near) if v]; invalid=[start+i for i,v in enumerate(valid) if not v]
                            transitions=int(np.count_nonzero(near[1:]!=near[:-1]))
                            record['methods'][mode]=dict(valid_frames=sum(valid),invalid_frames=invalid,eligible_frames=eligible,first_eligible=eligible[0] if eligible else None,switch_transitions=transitions,contact_events=events,distance_cm=[float(d) if v else None for d,v in zip(distances,valid)])
                            stat=summary[f'{obj}/{kind}/{mode}'];stat['segments']+=1;stat['frames']+=len(valid);stat['valid_frames']+=sum(valid);stat['eligible_frames']+=len(eligible);stat['segments_ever_eligible']+=bool(eligible)
                        record['nearest_surface_cm']=surface
                        output.write(json.dumps(record,allow_nan=False)+'\n');count+=1
                    print(obj,key,flush=True)
    report=dict(threshold_m=a.threshold,max_demos_per_object=a.max_demos,segments=count,summary=dict(summary),scope='All frames; labels only, not predicted-contact rollout. Deterministic first-N sample.')
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Demonstration contact-switch audit','',f'Threshold: {a.threshold*100:g} cm. First {a.max_demos} demos per object. All frames; no model inference.','', '| Object / skill / labels | Valid frames | Eligible frames | Segments ever eligible |','|---|---:|---:|---:|']
    for k,s in sorted(summary.items()): lines.append(f"| {k} | {s['valid_frames']}/{s['frames']} | {s['eligible_frames']}/{s['frames']} | {s['segments_ever_eligible']}/{s['segments']} |")
    lines+=['','TCP-to-label proximity is not physical finger contact. The training sample stride is 2; this report includes unsampled frames too. Existing labels and configurations are unchanged. Per-frame events, missing geometry, distances and eligibility are in segments.jsonl.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
