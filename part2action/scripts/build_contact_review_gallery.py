"""Build an offline, synchronized contact-label review gallery from recorded demos."""
import argparse
import base64
import io
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image, ImageDraw
from audit_contact_labels import _segments
from data.targets import _detect_contact_step, derive_contact_xyz, derive_skill_contact_target, _part_points_world
from data.geometry import tcp_position_numpy
from visualize_contact_projection import project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results/diagnostics/contact_label_audit/review'))
    parser.add_argument('--search-demos', type=int, default=100)
    parser.add_argument('--max-frames', type=int, default=45)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    examples = []
    for obj in ['bottle', 'mug', 'pliers', 'scissors']:
        with h5py.File(f'datasets/PartInstruct/demos/{obj}.hdf5', 'r') as f:
            selected = {}
            for key in sorted(f['data'], key=lambda k: int(k.split('_')[-1]))[:args.search_demos]:
                demo = f['data'][key]
                for start, end, instruction in _segments(demo['skill_instructions'][:]):
                    kind = 'grasp_obj' if instruction.lower().startswith('grasp ') else 'touch_obj' if instruction.lower().startswith('touch ') else None
                    if kind and kind not in selected:
                        selected[kind] = (key, start, end, instruction)
                if len(selected) == 2:
                    break
            for kind, (key, start, end, instruction) in selected.items():
                demo = f['data'][key]; obs = demo['obs']
                actions = demo['actions'][:]; poses = obs['tcp_pose'][:]; cloud = obs['agentview_part_pcd'][:]
                target = derive_skill_contact_target(actions=actions, part_pcd=cloud, tcp_pose=poses, skill_kind=kind, segment_start=start, segment_end=end, current_t=start)
                event = int(target['contact_t']); legacy = _detect_contact_step(actions,start)
                times = sorted(set(np.linspace(start,end,min(args.max_frames,end-start+1),dtype=int).tolist()+[legacy]+([event] if event >= 0 else [])))
                times = [t for t in times if start <= t <= end]
                timeline = []
                for t in range(start,end+1):
                    tcp = tcp_position_numpy(poses[t]); pts = _part_points_world(cloud,t)
                    lt = _detect_contact_step(actions,t); lp,lv = derive_contact_xyz(cloud,poses,lt,current_t=t)
                    sp,sv = derive_contact_xyz(cloud,poses,event,current_t=t) if event>=0 else (np.zeros(3),False)
                    timeline.append(dict(t=t, surface_cm=float(np.linalg.norm(pts-tcp,axis=-1).min())*100 if len(pts) else None,legacy_cm=float(np.linalg.norm(lp-tcp))*100 if lv else None,skill_cm=float(np.linalg.norm(sp-tcp))*100 if sv else None,command=float(actions[t,6]),opening=float(np.asarray(obs['gripper_state'][t]).reshape(-1)[0]) if 'gripper_state' in obs else None,legacy_event=int(lt)))
                frames=[]
                for t in times:
                    rgb=obs['agentview_rgb'][t]; h,w=rgb.shape[:2]; mask=np.asarray(obs['agentview_part_mask'][t]).squeeze()>0
                    images=[]; lt=_detect_contact_step(actions,t)
                    lp,lv=derive_contact_xyz(cloud,poses,lt,current_t=t)
                    sp,sv=derive_contact_xyz(cloud,poses,event,current_t=t) if event>=0 else (np.zeros(3),False)
                    for title,point,valid,color in [('Recorded RGB',None,False,'white'),('Legacy contact',lp,lv,'#ff5656'),('Skill-specific contact',sp,sv,'#00ddff')]:
                        im=Image.fromarray(rgb.astype('uint8')).convert('RGB')
                        if point is not None and mask.shape==(h,w):
                            arr=np.array(im); arr[mask]=(arr[mask]*.65+np.array([65,220,100])*.35).astype('uint8'); im=Image.fromarray(arr)
                        draw=ImageDraw.Draw(im)
                        if point is not None:
                            for xyz,c,r in [(tcp_position_numpy(poses[t]),'#ffe25c',5)]+([(point,color,6)] if valid else []):
                                uv=project(xyz,w,h)
                                if uv is not None:
                                    x,y=uv; draw.ellipse((x-r,y-r,x+r,y+r),outline=c,width=3)
                            # Actual recorded TCP history, not predicted approach output.
                            trail=[project(tcp_position_numpy(poses[k]),w,h) for k in range(max(start,t-12),t+1)]
                            trail=[v for v in trail if v is not None]
                            if len(trail)>1: draw.line(trail,fill='#ffe25c',width=2)
                        draw.rectangle((0,0,w,24),fill='#101820'); draw.text((6,6),title,fill='white')
                        b=io.BytesIO(); im.save(b,format='JPEG',quality=80); images.append(base64.b64encode(b.getvalue()).decode())
                    frames.append(dict(t=t,images=images,legacy_event=int(lt),skill_event=event))
                examples.append(dict(object=obj,demo=key,kind=kind,instruction=instruction,start=start,end=end,legacy_start_event=int(legacy),skill_event=event,frames=frames,timeline=timeline))
                print(obj,kind,key,len(frames),flush=True)
    payload=json.dumps(examples,allow_nan=False)
    template=Path(__file__).with_name('contact_review_template.html').read_text()
    (args.out/'index.html').write_text(template.replace('__DATA__',payload))
    (args.out/'review_metadata.json').write_text(json.dumps([{k:v for k,v in e.items() if k!='frames'} for e in examples],indent=2)+'\n')
    print('Gallery:',args.out/'index.html',flush=True)

if __name__=='__main__': main()
