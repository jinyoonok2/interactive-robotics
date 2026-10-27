"""Matched normalized-contact rollout metrics and side-by-side GIF review."""
import argparse
import csv
import html
import json
import statistics
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from build_policy_comparison_videos import key, render_panels
from export_evaluation_gifs import write_bundle
from ground_truth_reference import DEFAULT_DATA_ROOT, GroundTruthReferences


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def analyze(trial, model):
    rows = []
    previous_rotation = None
    for i, (event, diagnostic, action) in enumerate(zip(trial['phase_events'],trial['diagnostics'],trial['actions'])):
        alignment = event['initial_target_part_alignment']
        assert alignment['supported']
        commanded = Rotation.from_euler('xyz',action[3:6])
        measured = Rotation.from_quat(diagnostic['tcp_quat'])
        error = (commanded.inv()*measured).magnitude()*180/np.pi
        jump = (previous_rotation.inv()*commanded).magnitude()*180/np.pi if previous_rotation is not None else None
        previous_rotation = commanded
        rows.append({'model':model,'object':trial['obj_class'],'task':trial['task_type'],'trial':trial['trial_index'],
                     'seed':trial['seed'],'episode_id':trial['info']['ep_id'],'step':i,'phase':event['phase'],
                     'tcp_to_part_cm':100*alignment['tcp_to_part_min_distance_m'],
                     'contact_to_part_cm':100*alignment['predicted_contact_to_part_min_distance_m'],
                     'action_to_part_cm':100*alignment['action_target_to_part_min_distance_m'],
                     'position_tracking_error_cm':float(np.linalg.norm(np.array(action[:3])-diagnostic['tcp_xyz'])*100),
                     'wrist_tracking_error_deg':float(error),'command_rotation_jump_deg':float(jump) if jump is not None else None,
                     'finger_command_mm':action[6]*1000,'measured_left_finger_mm':diagnostic['measured_finger_position_m']*1000,
                     'learned_contact_weight':diagnostic.get('contact_conditioning_weight')})
    assert len(rows)==len(trial['actions'])
    grasp = trial['phase_events'][0]['initial_target_part_alignment']['skill_name']=='grasp_obj'
    near5 = next((r for r in rows if r['tcp_to_part_cm']<=5),None)
    near10 = next((r for r in rows if r['tcp_to_part_cm']<=10),None)
    closed = next((r for r in rows if r['finger_command_mm']<=1),None)
    measured_closed = next((r for r in rows if r['measured_left_finger_mm']<=1),None)
    summary = {k:rows[0][k] for k in ['model','object','task','trial','seed','episode_id']}
    summary.update(success=trial['success'],steps=len(rows),initial_skill='grasp' if grasp else 'touch',
                   first_within_5cm_step=near5['step'] if near5 else None,
                   first_within_10cm_step=near10['step'] if near10 else None,
                   finger_command_at_first_10cm_mm=near10['finger_command_mm'] if near10 else None,
                   measured_finger_at_first_10cm_mm=near10['measured_left_finger_mm'] if near10 else None,
                   first_closed_command_step=closed['step'] if closed else None,
                   tcp_distance_at_first_closed_command_cm=closed['tcp_to_part_cm'] if closed else None,
                   first_measured_closed_step=measured_closed['step'] if measured_closed else None,
                   tcp_distance_at_first_measured_closed_cm=measured_closed['tcp_to_part_cm'] if measured_closed else None,
                   mean_tcp_to_part_cm=mean([r['tcp_to_part_cm'] for r in rows]),
                   last_third_tcp_to_part_cm=mean([r['tcp_to_part_cm'] for r in rows[2*len(rows)//3:]]),
                   min_tcp_to_part_cm=min(r['tcp_to_part_cm'] for r in rows),
                   mean_contact_to_part_cm=mean([r['contact_to_part_cm'] for r in rows]),
                   mean_wrist_tracking_error_deg=mean([r['wrist_tracking_error_deg'] for r in rows]),
                   mean_command_rotation_jump_deg=mean([r['command_rotation_jump_deg'] for r in rows]),
                   mean_position_tracking_error_cm=mean([r['position_tracking_error_cm'] for r in rows]),
                   clipped_gripper_commands=trial['gripper_clipped_count'],
                   effective_contact_weight=rows[0]['learned_contact_weight'] if model=='contact' else 0.)
    return summary,rows


def csv_file(path, rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def gif_from_video(video, gif):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(video),
                    '-filter_complex_threads','1','-filter_complex',
                    'fps=5,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer',
                    '-threads','2','-loop','0',str(gif)],check=True)
    with Image.open(gif) as image:
        assert image.size==(1440,720) and image.n_frames>1
        duration=0
        for i in range(image.n_frames):
            image.seek(i);duration+=image.info.get('duration',0)
    return {'gif_frames':image.n_frames,'gif_duration_s':duration/1000}


def fmt(value, unit=''):
    return 'never' if value is None else f'{value:.1f}{unit}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    parser.add_argument('--reuse-media',action='store_true',help='Refresh measurements and gallery without re-encoding existing GIFs and MP4s.')
    args=parser.parse_args();root=args.source_dir;out=root;out.mkdir(exist_ok=True)
    (out/'_raw'/'composites').mkdir(parents=True,exist_ok=True)
    references=GroundTruthReferences(out/'_raw'/'ground_truth',args.data_root)
    previous = {r['gif']:r for r in json.loads((out/'manifest.json').read_text())} if args.reuse_media else {}
    data={};summaries={};all_steps=[];all_trials=[];aggregate={}
    for name,folder in [('control','unified_control'),('contact','unified_contact')]:
        payload=json.loads(next((root/'_raw'/folder/'hierarchical').glob('rollout_results_*.json')).read_text())
        assert payload['gripper_command_bounds_m']==[0,.04] and payload['termination_mode']=='legacy'
        trials=payload['results'];data[name]={key(t):t for t in trials};summaries[name]={}
        for t in trials:
            summary,rows=analyze(t,name);summaries[name][key(t)]=summary;all_trials.append(summary);all_steps.extend(rows)
        selected=list(summaries[name].values());grasps=[s for s in selected if s['initial_skill']=='grasp']
        aggregate[name]={'trials':len(trials),'successes':sum(t['success'] for t in trials),
            'reached_within_5cm':sum(s['first_within_5cm_step'] is not None for s in selected),
            'reached_within_10cm':sum(s['first_within_10cm_step'] is not None for s in selected),
            'grasp_trials':len(grasps),'grasp_command_closed_outside_10cm':sum(s['tcp_distance_at_first_closed_command_cm'] is not None and s['tcp_distance_at_first_closed_command_cm']>10 for s in grasps),
            'grasp_never_commanded_closed':sum(s['first_closed_command_step'] is None for s in grasps),
            **{f:mean([s[f] for s in selected]) for f in ['mean_tcp_to_part_cm','last_third_tcp_to_part_cm','mean_contact_to_part_cm','mean_wrist_tracking_error_deg','mean_command_rotation_jump_deg','mean_position_tracking_error_cm','effective_contact_weight']}}
    assert set(data['control'])==set(data['contact']) and len(data['control'])==39
    csv_file(out/'per_trial.csv',all_trials);csv_file(out/'per_step.csv',all_steps)
    focus=[k for k in data['control'] if data['control'][k]['success'] and not data['contact'][k]['success']]
    (out/'summary.json').write_text(json.dumps({'aggregate':aggregate,'control_success_contact_failure':[{'key':k,'control':summaries['control'][k],'contact':summaries['contact'][k]} for k in focus],
        'definitions':{'distance':'Nearest sampled surface of the first ground-truth target part throughout motion; not physical contact.',
                       'expert_reference':'Leftmost panel is a recorded dataset expert demonstration with the same complete skill chain and parameters, preferring the same object instance. It is a different episode/starting scene when exact test recordings are unavailable; elapsed sample time is aligned, not task phase.',
                       'closed':'Command or measured left finger position at most 1mm; measured right finger is unavailable in these rollouts.',
                       'wrist_error':'Pre-action measured versus commanded wrist SO(3) angle; not error against a ground-truth grasp orientation.',
                       'aggregation':'Mean of per-trial means. Same timestep across diverging trajectories does not imply identical observations.'}},indent=2)+'\n')
    episodes=sorted(data['control'],key=lambda k:(k not in focus,k))
    manifest=[];cards=[]
    for k in episodes:
        left,right=data['control'][k],data['contact'][k]
        truth=references.trial(left)
        assert left['instruction']==right['instruction']
        assert np.allclose(left['diagnostics'][0]['tcp_xyz'],right['diagnostics'][0]['tcp_xyz'])
        obj,task,trial,seed,ep=k;stem=f'{obj}_task{task}_trial{trial:02d}_episode{ep}'
        video=out/'_raw'/'composites'/f'{stem}.mp4';gif=out/f'{stem}.gif'
        if args.reuse_media:
            assert video.is_file() and gif.is_file()
            entry=previous[str(gif.relative_to(out))]
            if 'ground_truth_reference' not in entry:
                raise ValueError('Existing media lacks ground truth; rebuild without --reuse-media')
            info={f:entry[f] for f in ['frames','input_fps','output_fps','input_frame_counts','gif_frames','gif_duration_s']}
        else:
            temporary_video=video.with_name(video.stem+'.tmp.mp4')
            temporary_gif=gif.with_name(gif.stem+'.tmp.gif')
            info=render_panels([truth,left,right],['GROUND-TRUTH REFERENCE','UNIFIED CONTROL','UNIFIED CONTACT'],temporary_video,.5,show_action_state=True)
            info.update(gif_from_video(temporary_video,temporary_gif))
            temporary_video.replace(video);temporary_gif.replace(gif)
        capture=cv2.VideoCapture(str(video));assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==info['frames']
        capture.set(cv2.CAP_PROP_POS_FRAMES,info['frames']-1);assert capture.read()[0];capture.release()
        manifest.append({'object':obj,'task':task,'trial':trial,'seed':seed,'episode_id':ep,'focus':k in focus,
                         'control_success':left['success'],'contact_success':right['success'],
                         'panel_order':['ground_truth_reference','control','contact'],
                         'ground_truth_reference':truth['ground_truth_reference'],
                         'video':str(video.relative_to(out)),'gif':str(gif.relative_to(out)),
                         'source_videos':[truth['rollout_video'],left['rollout_video'],right['rollout_video']],**info})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        rows=''
        for metric,label,unit in [('min_tcp_to_part_cm','Nearest TCP to true part',' cm'),('first_within_5cm_step','First within 5cm, step',''),
                                 ('finger_command_at_first_10cm_mm','Finger command on first entering 10cm',' mm'),
                                 ('measured_finger_at_first_10cm_mm','Measured finger on first entering 10cm',' mm'),
                                 ('first_closed_command_step','First closed command, step',''),('tcp_distance_at_first_closed_command_cm','TCP distance at first close',' cm'),
                                 ('mean_wrist_tracking_error_deg','Mean wrist tracking error','°'),('mean_command_rotation_jump_deg','Mean wrist command change','°')]:
            rows+=f'<tr><td>{label}</td>'+''.join(f'<td>{fmt(summaries[m][k][metric],unit)}</td>' for m in ['control','contact'])+'</tr>'
        outcomes=f'Control: {"SUCCESS" if left["success"] else "failure"}; Contact: {"SUCCESS" if right["success"] else "failure"}'
        reference=truth['ground_truth_reference']
        match_note='same evaluation episode' if reference['match_type']=='exact_episode' else 'same complete skill chain, different episode'
        note=f'Left: expert {reference["demo_key"]}, episode {reference["episode_id"]}; {match_note}. Object instance: {"same" if reference["same_object_instance"] else "different"}.'
        cards.append(f'<article data-object="{obj}" data-focus="{str(k in focus).lower()}"><h2>{obj} task {task}, trial {trial} — episode {ep}</h2><p>{html.escape(left["instruction"])} | {outcomes}</p><p>{html.escape(note)}</p><img loading="lazy" src="{gif.relative_to(out)}" alt="Ground-truth reference, control and contact GIF"><p><a href="{gif.relative_to(out)}" download>Download GIF</a></p><table><tr><th>Metric</th><th>Control</th><th>Contact</th></tr>{rows}</table></article>')
        print(f'[{len(manifest)}/39] {stem}',flush=True)
    aggregate_rows=''
    for field,label in [('successes','Successes /39'),('reached_within_5cm','Trials reaching within 5cm /39'),('last_third_tcp_to_part_cm','Final-third TCP distance, cm'),
                        ('mean_contact_to_part_cm','Mean predicted-contact error, cm'),('mean_wrist_tracking_error_deg','Mean wrist tracking error, degrees'),
                        ('grasp_command_closed_outside_10cm','Grasp trials first closing outside 10cm')]:
        aggregate_rows+='<tr><td>'+label+'</td>'+''.join(f'<td>{aggregate[m][field]:.2f}</td>' for m in ['control','contact'])+'</tr>'
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Normalized contact policy review</title><style>body{font:16px system-ui;background:#17191d;color:#eee;max-width:1600px;margin:auto;padding:24px}article{padding:18px;background:#24272d;margin:20px 0;border-radius:10px}img,video{width:100%;max-width:1440px}a{color:#9bd0ff}td,th{padding:8px;text-align:left}select,input{font:inherit;margin:8px}h2{font-size:20px}</style><h1>Normalized contact: matched control/contact GIFs</h1><p>39 matched test1 trials using the 40-epoch continued control and normalized-contact checkpoints. Ground-truth reference is left; control is center; contact is right. The four control-success/contact-failure trials are shown first. They cover three unique episode IDs because one mug episode is repeated with different seeds. Half-speed, aligned by elapsed sample time from each video start; earlier finishes hold the final frame. The expert reference matches the complete skill chain and parameters but comes from a different episode and starting scene; it is not a replay of this test episode or phase-aligned ground truth. Green tint on the expert panel marks the dataset target part.</p><table><tr><th>Aggregate</th><th>Control</th><th>Contact</th></tr>'''+aggregate_rows+'''</table><p>Blue: predicted contact; green: nearest sampled point on the first true target part. The 5/10cm distances are descriptive approach thresholds, not physical-contact tests. Finger command/measurement refers to one finger joint, not total opening. Wrist error is the angle between the current measured orientation and commanded orientation, not error against an expert grasp pose. Low wrist tracking error can simply reflect stationary commands. A grasped object can keep measured finger joints above 1mm even during a successful grasp. First-close timing records the first threshold crossing, not necessarily sustained closure. No physical collision forces or ground-truth wrist poses were recorded in these runs.</p><p>Metrics compare different executed trajectories. They do not isolate the causal effect of the contact input on identical observations. Distance aggregates track the first instructed part even after phase changes, and longer trajectories can include more failure steps.</p><label><input id="focus" type="checkbox">Four control successes only</label><label>Object <select id="object"><option value="">All</option><option>mug</option><option>scissors</option><option>bottle</option><option>pliers</option></select></label><p><a href="per_trial.csv">Per-trial CSV</a> · <a href="per_step.csv">Per-step CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="comparison_bundle.zip">Download all GIFs and measurements</a></p>'''+''.join(cards)+'''<script>function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=(document.querySelector('#focus').checked&&a.dataset.focus!=='true')||(document.querySelector('#object').value&&a.dataset.object!==document.querySelector('#object').value))}document.querySelectorAll('input,select').forEach(e=>e.onchange=filter);filter()</script>'''
    (out/'index.html').write_text(page)
    write_bundle(out)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
