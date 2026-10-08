"""Small CPU-only pilot: compare label events and project diagnostic markers.

Run with PYTHONPATH=part2action. This does not change training labels.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.geometry import _AGENTVIEW_TO_WORLD, tcp_position_numpy
from data.targets import derive_contact_xyz, derive_skill_contact_target, _detect_contact_step
from audit_contact_labels import _segments

OUT = Path('results/diagnostics/contact_label_audit')
OUT.mkdir(parents=True, exist_ok=True)
world_to_camera = np.linalg.inv(_AGENTVIEW_TO_WORLD)


def project(point, width, height):
    gl = world_to_camera @ np.r_[point, 1.0]
    x, y, z = gl[:3] * np.array([1, -1, -1])
    if z <= 0:
        return None
    focal = height / (2 * np.tan(np.pi / 6))
    return (focal * x / z + width / 2, focal * y / z + height / 2)


rows = []
for obj in ['bottle', 'mug', 'pliers', 'scissors']:
    selected = {}
    with h5py.File(f'datasets/PartInstruct/demos/{obj}.hdf5', 'r') as f:
        keys = sorted(f['data'], key=lambda k: int(k.split('_')[-1]))[:8]
        for key in keys:
            demo = f['data'][key]
            for start, end, instruction in _segments(demo['skill_instructions'][:]):
                kind = 'grasp_obj' if instruction.lower().startswith('grasp ') else 'touch_obj' if instruction.lower().startswith('touch ') else None
                if kind and kind not in selected:
                    selected[kind] = (key, start, end, instruction)
        for kind, (key, start, end, instruction) in selected.items():
            demo = f['data'][key]; obs = demo['obs']
            actions = demo['actions'][:]; poses = obs['tcp_pose'][:]; cloud = obs['agentview_part_pcd'][:]
            legacy_t = _detect_contact_step(actions, start)
            target = derive_skill_contact_target(actions=actions, part_pcd=cloud, tcp_pose=poses, skill_kind=kind, segment_start=start, segment_end=end, current_t=start)
            event_t = int(target['contact_t'])
            if event_t < 0:
                continue
            times = [start, max(start, event_t - 2), event_t, min(end, event_t + 2)]
            fig, axes = plt.subplots(1, len(times), figsize=(15, 4))
            record = dict(object=obj, demo=key, skill=kind, instruction=instruction, segment=[start,end], legacy_contact_t=int(legacy_t), skill_contact_t=event_t, event_distance_cm=None if target['tcp_to_part_distance_m'] is None else float(target['tcp_to_part_distance_m'])*100, frames=[])
            for ax, t in zip(axes, times):
                rgb = obs['agentview_rgb'][t]; height, width = rgb.shape[:2]
                ax.imshow(rgb)
                mask = np.asarray(obs['agentview_part_mask'][t]).squeeze()
                if mask.shape == (height,width) and np.any(mask):
                    ax.contour(mask.astype(float), levels=[0.5], colors=['lime'], linewidths=0.7)
                # Legacy event depends on current t, exactly as the loader does.
                lt = _detect_contact_step(actions,t)
                lp, lv = derive_contact_xyz(cloud,poses,lt,current_t=t)
                sp, sv = derive_contact_xyz(cloud,poses,event_t,current_t=t)
                for point, valid, color, marker, label in [(lp,lv,'red','x','legacy'),(sp,sv,'cyan','+','skill-specific'),(tcp_position_numpy(poses[t]),True,'yellow','o','TCP')]:
                    uv = project(point,width,height)
                    if valid and uv is not None:
                        ax.scatter(*uv,c=color,marker=marker,s=55,label=label)
                ax.set_title(f't={t}; legacy event={lt}')
                ax.axis('off')
                record['frames'].append(dict(t=t,legacy_contact_t=int(lt),label_difference_cm=float(np.linalg.norm(lp-sp))*100 if lv and sv else None))
            axes[0].legend(fontsize=7,loc='lower left')
            fig.suptitle(f'{obj}/{key}: {instruction}\nsegment {start}–{end}; skill event {event_t}. Green=dataset part; marker projections assume PartGym fixed camera.')
            fig.tight_layout()
            name=f'{obj}_{kind}.png'; fig.savefig(OUT/name,dpi=130); plt.close(fig)
            record['figure']=name; rows.append(record)
(OUT/'visual_examples.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
