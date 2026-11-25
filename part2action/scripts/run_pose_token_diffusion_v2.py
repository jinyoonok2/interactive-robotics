"""Prepare and queue a pinned, fresh-policy v2 pilot with component validation."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

from _common import ROOT as PART
from models.pose_token_diffusion_v2.pilot import prepare_pilot
from models.pose_token_diffusion_v2.dataset import build_dataset
from models.pose_token_diffusion_v2.training import source_paths, verify_plan
from models.pose_token_diffusion_v2.completion import CompletionDataset
from diagnostics.evaluation_protocol import execution_settings
from register_result import register

ROOT = PART.parent
PACKAGE = PART/'models/pose_token_diffusion_v2'
LAUNCHER = PART/'slurm/pose_token_diffusion_v2.sbatch'


def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    recipe = yaml.safe_load(args.config.read_text())
    manifest = json.loads((args.reference_study/'demonstrations.json').read_text()) if args.reference_study else prepare_pilot(recipe['data']['train_hdf5'],per_object=args.per_object,
        validation_fraction=recipe['data']['validation_fraction'],seed=recipe['train']['seed'],
        action_chunk=recipe['data']['action_chunk'],max_skill_slots=recipe['data']['max_skill_slots'])
    args.out_dir.mkdir(parents=True,exist_ok=False)
    manifest_path = args.out_dir/'demonstrations.json'
    write(manifest_path,manifest)
    recipe['model_config'] = str((args.config.parent/recipe['model_config']).resolve(strict=True))
    recipe['data']['demo_manifest'] = str(manifest_path)
    config = args.out_dir/'pilot_resolved.yaml'
    config.write_text(yaml.safe_dump(recipe,sort_keys=False))
    audit = {}
    for split in ('train','validation'):
        dataset = build_dataset({**recipe['data'],'split':split},recipe['goal_labels'])
        seen,counts,valid,reasons = set(),Counter(),Counter(),Counter()
        for spec in dataset._samples:
            key=spec.file_path,spec.demo_key
            if key in seen:continue
            seen.add(key)
            for phase in dataset.targets_for(spec):
                counts[phase['kind']]+=1
                valid[phase['kind']]+=int(phase['component_valid'].any())
                reasons[phase['reason'] or 'valid']+=1
        required={'grasp_obj','touch_obj','move_gripper','rotate_obj','release_obj'}
        if not required<=set(valid) or any(valid[k]<1 for k in required):
            raise ValueError('Pilot split lacks valid supervision for every supported skill: '+split)
        audit[split]={'demonstrations':len(seen),'samples':len(dataset),'phase_counts':dict(counts),
            'valid_goal_counts':dict(valid),'label_reasons':dict(reasons)}
        dataset.close()
    train={(r['file'],r['demo']) for r in manifest['demonstrations'] if r['split']=='train'}
    validation={(r['file'],r['demo']) for r in manifest['demonstrations'] if r['split']=='validation'}
    if train&validation:raise ValueError('Pilot train/validation demonstration leakage')
    write(args.out_dir/'label_audit.json',{'splits':audit,'demo_overlap':0,
        'limits':'Touch is a geometry proxy; later goals are demonstrated TCP endpoint proxies. Not verified contact/retention.'})
    completion_audit = {}
    extra_paths = []
    if recipe.get('completion'):
        for split in ('train','validation'):
            completion = CompletionDataset(manifest_path,split=split,
                n_obs_steps=recipe['data']['n_obs_steps'],**recipe['completion'])
            completion_audit[split] = completion.audit
            if any(completion.audit['by_skill'].get(k,{}).get('positive',0)<1
                   for k in ('grasp_obj','touch_obj','move_gripper','rotate_obj','release_obj')):
                raise ValueError('Missing positive completion proxy coverage: '+split)
            extra_paths.append(completion.metadata_path)
        write(args.out_dir/'completion_label_audit.json',completion_audit)
    if recipe['train'].get('reference_initialization'):
        extra_paths.append(Path(recipe['train']['reference_initialization']))
    if args.reference_study:extra_paths.append(args.reference_study/'demonstrations.json')
    if args.development_reference:extra_paths.append(args.development_reference/'plan.json')
    paths=source_paths()+[config,manifest_path,args.config.resolve()]+extra_paths
    sources={str(p.resolve()):sha(p) for p in sorted(set(paths))}
    snapshot=args.out_dir/'source_snapshot';snapshot.mkdir()
    for i,name in enumerate(sources):shutil.copy2(name,snapshot/f'{i:03d}_{Path(name).name}')
    data_files={}
    for name in recipe['data']['train_hdf5']:
        stat=Path(name).stat();data_files[name]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    write(args.out_dir/'plan.json',{'created_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'architecture':'pose_token_diffusion_v2','scope':'Development joint-training pilot, not the official test benchmark.',
        'config':str(config),'source_sha256':sources,'data_files':data_files,'label_audit':audit,
        'execution_settings':execution_settings('test1'),'initialization':'random_policy_frozen_backbones',
        'epochs':recipe['train']['epochs'],'per_object':args.per_object,'data_manifest':str(manifest_path),
        'completion_label_audit':completion_audit,
        'reference_study':str(args.reference_study) if args.reference_study else None,
        'development_reference':str(args.development_reference) if args.development_reference else None,
        'development_output':str(args.out_dir.with_name(args.out_dir.name+'_development')) if args.development_reference else None,
        'component_reports':['grounding','goal_pose','applicability','phase_termination','diffusion_action_loss'],
        'validation_schedule':'Untrained baseline and every training epoch; entire selected held-out demonstration frames.',
        'checkpoint_selection':'Development validation only; no official Test1 scenarios are used for tuning.',
        'simulator_rollouts':'Same 24 development rollouts queued after training.' if args.development_reference else 'Not part of this training submission; saved checkpoints support separate 250/8 evaluation.'})
    register('diagnostics',args.out_dir)
    (args.out_dir/'README.md').write_text('# Pose-token Diffusion v2 pilot\n\nPrepared fresh-policy joint training with frozen pretrained DINOv2/T5.\n\n'
        f"Selected {sum(x['demonstrations'] for x in audit.values())} training-release demonstrations: {audit['train']['demonstrations']} train and {audit['validation']['demonstrations']} held out. No official benchmark episodes.\n\n"
        'A full-backbone GPU smoke update, held-out component report and action sampling must pass before training starts. '
        'Training produces epoch-0 and per-epoch JSON/CSV/Markdown reports and PNG overlays.\n\n'
        '[Selection](demonstrations.json) | [Label audit](label_audit.json) | [Recipe](pilot_resolved.yaml) | [Plan](plan.json)\n\n'
        '[Smoke](smoke/README.md) | [Training progress](training/README.md)\n')
    print(json.dumps({'study':str(args.out_dir),'data':audit,'epochs':recipe['train']['epochs']}),flush=True)


def submit(args,plan):
    receipt=args.out_dir/'submissions.json'
    if receipt.exists():raise FileExistsError('Pilot already has submission receipts; do not submit duplicate jobs')
    jobs=[]
    def queue(mode,extra):
        command=['sbatch','--parsable','--partition=gpu','--no-requeue','--gres=gpu:1',
            '--cpus-per-task=4','--mem=32G','--constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4000|a4500|rtx_4000_ada',
            '--job-name=p2a-v2-'+mode,'--output='+str(args.out_dir/(mode+'_%j.out')),
            '--error='+str(args.out_dir/(mode+'_%j.err')),
            '--export=ALL,STUDY='+str(args.out_dir)+',MODE='+mode,*extra,str(LAUNCHER)]
        job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
        jobs.append({'mode':mode,'job_id':job,'command':command})
        receipt.write_text(json.dumps(jobs,indent=2)+'\n')
        return job
    smoke=queue('smoke',['--time=00:30:00'])
    train=queue('train',['--time=04:00:00','--dependency=afterok:'+smoke,'--kill-on-invalid-dep=yes'])
    if plan.get('development_reference'):
        command=['sbatch','--parsable','--partition=cpu','--no-requeue','--cpus-per-task=2','--mem=8G','--time=00:15:00',
            '--job-name=p2a-v2-dev-prepare','--output='+str(args.out_dir/'development_prepare_%j.out'),
            '--error='+str(args.out_dir/'development_prepare_%j.err'),
            '--dependency=afterok:'+train,'--kill-on-invalid-dep=yes',
            '--export=ALL,STUDY='+str(args.out_dir)+',MODE=prepare-development',str(LAUNCHER)]
        prepare_job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
        jobs.append({'mode':'prepare-development','job_id':prepare_job,'command':command})
        receipt.write_text(json.dumps(jobs,indent=2)+'\n')
        launcher=PART/'slurm/evaluate_pose_token_v2_development.sbatch'
        command=['sbatch','--parsable','--partition=gpu','--no-requeue','--gres=gpu:1',
            '--cpus-per-task=4','--mem=32G','--time=02:00:00','--array=0-1%2',
            '--constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4000|a4500|rtx_4000_ada',
            '--job-name=p2a-v2-completion-eval','--dependency=afterok:'+prepare_job,'--kill-on-invalid-dep=yes',
            '--output='+str(args.out_dir/'evaluation_%A_%a.out'),'--error='+str(args.out_dir/'evaluation_%A_%a.err'),
            '--export=ALL,P2A_V2_DEV_STUDY='+plan['development_output']+',MODE=evaluate',str(launcher)]
        evaluate_job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
        jobs.append({'mode':'evaluate','job_id':evaluate_job,'command':command})
        receipt.write_text(json.dumps(jobs,indent=2)+'\n')
        command=['sbatch','--parsable','--partition=cpu','--no-requeue','--cpus-per-task=2','--mem=8G','--time=00:10:00',
            '--job-name=p2a-v2-completion-report','--dependency=afterany:'+evaluate_job,'--kill-on-invalid-dep=yes',
            '--output='+str(args.out_dir/'comparison_%j.out'),'--error='+str(args.out_dir/'comparison_%j.err'),
            '--export=ALL,P2A_V2_DEV_STUDY='+plan['development_output']+',MODE=report',str(launcher)]
        report_job=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
        jobs.append({'mode':'report','job_id':report_job,'command':command})
        receipt.write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','submit','smoke','train','prepare-development'),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PACKAGE/'configs/pilot.yaml')
    parser.add_argument('--per-object',type=int,default=64)
    parser.add_argument('--reference-study',type=Path)
    parser.add_argument('--development-reference',type=Path)
    args=parser.parse_args();args.out_dir=args.out_dir.resolve();args.config=args.config.resolve()
    if args.mode=='prepare':prepare(args);return
    plan=verify_plan(args.out_dir/'plan.json')
    if args.mode=='submit':submit(args,plan);return
    if args.mode=='prepare-development':
        subprocess.run([sys.executable,str(PART/'scripts/run_pose_token_v2_development.py'),
            '--mode','prepare','--out-dir',plan['development_output'],
            '--training-study',str(args.out_dir),'--reference-development',plan['development_reference']],cwd=ROOT,check=True)
        return
    command=[sys.executable,str(PART/'scripts/train_pose_token_diffusion_v2.py'),
        '--config',plan['config'],'--out-dir',str(args.out_dir/('smoke' if args.mode=='smoke' else 'training')),
        '--expected-plan',str(args.out_dir/'plan.json')]
    if args.mode=='smoke':command.append('--smoke')
    else:command+=['--smoke-receipt',str(args.out_dir/'smoke/smoke.json')]
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
