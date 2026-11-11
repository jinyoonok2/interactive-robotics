"""Matched fine-tuning of the existing visual Transformer with oracle grasp pose.

Stage one only: pose-estimator training is deliberately absent. Every policy
action, including the full task after grasp, comes from the original diffusion
head. Native expert execution occurs only in a separate oracle-reference scene.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('control', 'oracle_pose')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, plan):
    for name, expected in plan['source_sha256'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    for name, expected in plan['input_sha256'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen input changed: ' + name)


def prepare(args):
    import yaml
    from diagnostics.evaluation_scenarios import load_scenarios, rollout_plan
    config = read(args.config)
    data = Path(config['data_root'])
    scenarios = load_scenarios(data/'episodes_meta_test.json', 'test1', config['objects'])
    if len(scenarios) != 149:
        raise ValueError('Expected original 149-scenario Test1 population')
    trials = [{'scenario_index': i, 'seed': seed,
               'needs_grasp_pose': bool(s['episode']['chain_params'] and
                    s['episode']['chain_params'][0]['skill_name'] == 'grasp_obj')}
              for i, s in enumerate(scenarios) for seed in config['evaluation']['seeds']]
    rollout_plan(scenarios, config['evaluation']['seeds'])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT/config['baseline_config']).read_text())
    configs = {}
    for arm in ARMS:
        cfg = copy.deepcopy(base)
        cfg['name'] = 'visual_oracle_pose_v1_' + arm
        cfg['output_dir'] = str(args.out_dir/'models'/arm)
        cfg['train'].update(epochs=config['train']['epochs'], lr=config['train']['lr'],
                            init_checkpoint=config['init_checkpoint'], seed=config['train']['seed'])
        cfg['pose_guidance'] = {'enabled': arm == 'oracle_pose', 'scale_m': .1,
            'target': 'static measured preclosure TCP for first-skill grasp, throughout task',
            'privileged': True, 'reference_clock': False}
        path = args.out_dir / (arm + '.yaml')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False)); configs[arm] = str(path)
    names = [Path(__file__), ROOT/'part2action/scripts/train_visual_pose_v1.py',
             ROOT/'part2action/scripts/rollout_partgym.py', ROOT/'part2action/scripts/_common.py',
             ROOT/'part2action/scripts/generate_logged_references.py',
             ROOT/'part2action/slurm/visual_oracle_pose_v1.sbatch', args.config.resolve(),
             ROOT/'PartInstruct/PartInstruct/PartGym/env/bullet_env.py',
             ROOT/'PartInstruct/PartInstruct/PartGym/env/backend/planner/bullet_planner.py']
    names += list((ROOT/'part2action/models').glob('*.py')) + list((ROOT/'part2action/data').glob('*.py'))
    names += list((ROOT/'part2action/diagnostics').glob('*.py'))
    names = list(dict.fromkeys(path.resolve() for path in names))
    source_sha = {str(p): sha(p) for p in names}
    inputs = [Path(config['init_checkpoint']), data/'episodes_meta_train.json', data/'episodes_meta_test.json',
              *(Path(v) for v in configs.values())]
    plan = {'config': config, 'configs': configs, 'scenarios': scenarios, 'trials': trials,
            'oracle_indices': [i for i,t in enumerate(trials) if t['needs_grasp_pose']],
            'source_sha256': source_sha, 'input_sha256': {str(p): sha(p) for p in inputs},
            'protocol': 'Same original visual Transformer, shared pretrained checkpoint, '
              'two matched fine-tuning epochs, full four-category data, same minibatch seed. '
              'Only target-TCP-pose feature conditioning is enabled versus disabled. '
              'No reference timing, replayed prefix, replayed tail or analytic action controller.',
            'limits': [
                'One training seed in this initial screen; evaluation seeds 42/43/44.',
                'Oracle pose is privileged. This does not evaluate a learned pose estimator.',
                'The static first-grasp goal is supplied for the whole task; no GT phase or close time is an input.',
                'Tasks not beginning with grasp have no target in either arm and remain in both populations.',
                'Oracle references require a successful separate native expert grasp in the exact reset/seed.',
                'Failed/missing oracle references are coverage gaps, not policy failures, and are never silently replaced.',
                'The native part-grasp predicate permits contact with either finger; sustained retention is not implied.',
                'The historical 64/447 checkpoint is an initialization, not the matched fine-tuned control result.',
            ]}
    dump(args.out_dir/'plan.json', plan); dump(args.out_dir/'config.json', config)
    snap = args.out_dir/'source_snapshot'; snap.mkdir()
    for i,p in enumerate(names):
        shutil.copyfile(p, snap/f'{i:03d}_{p.name}')
    dump(args.out_dir/'status.json', {'status': 'prepared'})
    report(args, plan)
    print(json.dumps({'scenarios':len(scenarios), 'rollouts_per_arm':len(trials),
                      'oracle_references':len(plan['oracle_indices'])}))


def audit(args, plan):
    from collections import Counter
    import train_visual_pose_v1 as train
    from _common import load_yaml, resolve_paths
    cfg = resolve_paths(load_yaml(plan['configs']['control']), train.ROOT)
    ds = train.build_dataset(cfg['data'])
    unique = {}
    for spec in ds._samples:
        unique.setdefault((spec.file_path, spec.demo_key), spec)
    rows = []
    for spec in unique.values():
        label = ds.target_for(spec)
        rows.append({'object':Path(spec.file_path).stem, 'demo':spec.demo_key,
                     **{k:v.tolist() if hasattr(v,'tolist') else v for k,v in label.items()}})
    counts = Counter(r['reason'] for r in rows)
    result = {'status':'completed', 'demos':len(unique), 'samples':len(ds),
              'target_reasons':dict(counts), 'valid_grasp_demos':sum(r['valid'] for r in rows),
              'per_object':dict(Counter(r['object'] for r in rows)), 'labels':rows}
    if len(unique) != 4053 or result['valid_grasp_demos'] < 1:
        raise ValueError('Unexpected dataset population or no valid preclosure targets')
    dump(args.out_dir/'label_audit.json', result)
    print({k:v for k,v in result.items() if k != 'labels'}, flush=True)


def load_initial(model, path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(payload['model'], strict=False)
    bad = [k for k in missing if not k.startswith(('visual.model.', 'text.model.', 'pose_conditioner.'))]
    if bad or unexpected:
        raise ValueError(f'Incompatible baseline checkpoint: {bad}, {unexpected}')


def preflight(args, plan):
    import torch
    import train_visual_pose_v1 as train
    from _common import load_yaml, resolve_paths, set_seed
    from data.partinstruct_loader import collate_part2action
    if not torch.cuda.is_available():
        raise RuntimeError('Real-data visual preflight requires a Slurm GPU')
    if read(args.out_dir/'label_audit.json')['status'] != 'completed':
        raise ValueError('Target audit did not complete')
    results = {}
    for arm in ARMS:
        cfg = resolve_paths(load_yaml(plan['configs'][arm]), train.ROOT)
        set_seed(cfg['train']['seed'])
        ds = train.build_dataset(cfg['data'])
        selected = []
        for i,s in enumerate(ds._samples):
            if ds.target_for(s)['valid']:
                selected.append(i)
            if len(selected) == 2:
                break
        batch = collate_part2action([ds[i] for i in selected])
        batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        model = train.build_model(cfg).cuda(); load_initial(model, plan['config']['init_checkpoint'])
        train.freeze_prefixes(model, cfg['train']['freeze_prefixes'])
        model.train(); optimizer = torch.optim.AdamW(train.optimizer_groups(model,cfg['train']['lr'],{}))
        _, probability, weights = train.curriculum_state(0,cfg)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(batch['rgb'],batch['task_instruction'],target_action=batch['action_chunk'],
                        **train.forward_kwargs(batch,True,probability,True))
            loss, _ = train.compute_losses(out,batch,weights,set(cfg['heads']))
        loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ValueError('Nonfinite preflight gradient: '+name)
        pose_grad = float(model.pose_conditioner.projection[-1].weight.grad.abs().sum())
        if arm == 'oracle_pose' and not pose_grad > 0:
            raise ValueError('Action loss does not train the pose-conditioning module')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); model.eval()
        obs = {k:batch[k] for k in ('agentview_pcd','tcp_pose','gripper_state','goal_pose','goal_valid')}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            sample = model(batch['rgb'],batch['task_instruction'],**obs)
        if sample['action_chunk'].shape != (2,8,7) or not torch.isfinite(sample['action_chunk']).all():
            raise ValueError('Invalid visual-policy samples')
        results[arm] = {'loss':float(loss), 'pose_action_gradient':pose_grad,
                        'action_shape':[2,8,7], 'finite_gradients':True,
                        'original_action_head':type(model.action_head).__name__}
        del model,optimizer,batch,out,loss,sample,ds; gc.collect(); torch.cuda.empty_cache()
    dump(args.out_dir/'preflight.json', {'status':'completed','arms':results})


def oracle_path(root, index):
    return root/'oracle_references'/f'{index:03d}.json'


def collect_oracle(args, plan):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from diagnostics.evaluation_scenarios import pin_scenario
    from diagnostics.execution_trace import physical_state
    from generate_logged_references import native_classes
    index = plan['oracle_indices'][args.index]
    trial = plan['trials'][index]; scenario = plan['scenarios'][trial['scenario_index']]
    path = oracle_path(args.out_dir,index)
    if path.exists():
        raise FileExistsError(path)
    env = None
    try:
        from _common import set_seed
        set_seed(trial['seed'])
        env_cls, planner_cls = native_classes()
        cfg = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
        cfg.data_root = plan['config']['data_root']; cfg.meta_path = 'episodes_meta_test.json'; cfg.device = 'cpu'
        cfg.shape_meta.obs = {k:v for k,v in cfg.shape_meta.obs.items()
                              if k in ('tcp_pose','gripper_state','joint_states')}
        env = env_cls(config=cfg,gui=False,record=False,evaluation=True,skill_mode=False,
                      obj_class=scenario['obj_class'],task_type=scenario['task_type'],split='test1')
        pin_scenario(env,scenario['episode'],trial['seed']); env.reset()
        initial = physical_state(env); goal = []
        apply = env.robot.apply_action; previous = [float(env.robot.get_gripper_state())]
        def capture(action,*values,**kwargs):
            if previous[0] > .02 >= float(action[6]):
                goal.append(np.asarray(env.robot.get_tcp_pose().to_list(),dtype=float).tolist())
            previous[0] = float(action[6])
            return apply(action,*values,**kwargs)
        env.robot.apply_action = capture
        planner = planner_cls(env,planner_model_path=Path(cfg.data_root)/'assets/models/vgn_conv.pth',generation_mode=True)
        planner.grasp_planner.net.eval()
        # CPU logging/render speed must not replace the native iteration limits.
        native_pose = planner.pose_tcp
        def pose(*values,**kwargs):
            kwargs['timeout'] = None
            return native_pose(*values,**kwargs)
        planner.pose_tcp = pose
        part = scenario['episode']['chain_params'][0]['params'].get('part_grasp','')
        success = bool(planner.grasp_obj(scenario['obj_class'],**scenario['episode']['chain_params'][0]['params']))
        grasp = bool(planner_cls.predicate_grasping(env,part_name=part))
        complete = success and grasp and len(goal) == 1
        dump(path, {'status':'completed' if complete else 'reference_unavailable',
                    'trial_index':index, **trial, 'initial_state':initial,
                    'goal_pose':goal[0] if complete else None, 'native_grasp':grasp,
                    'planner_success':success, 'closure_events':len(goal),
                    'source':'measured TCP immediately before native expert closure',
                    'wall_timeout_disabled':True, 'native_iteration_limits_unchanged':True})
    except BaseException as error:
        dump(path, {'status':'runtime_error', 'trial_index':index, 'error':repr(error)})
        raise
    finally:
        if env is not None:
            env.close()


def evaluate(args, plan):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from _common import load_yaml, resolve_paths
    import train_visual_pose_v1 as train
    import rollout_partgym as rollout
    from diagnostics.execution_trace import physical_state
    from diagnostics.phase_controller import should_advance_phase
    from generate_logged_references import native_classes
    arm = ARMS[args.index // plan['config']['evaluation']['shards']]
    shard = args.index % plan['config']['evaluation']['shards']
    cfg = resolve_paths(load_yaml(plan['configs'][arm]),train.ROOT)
    checkpoint = args.out_dir/'models'/arm/'last.pt'
    receipt = checkpoint.parent/'completed.json'
    if not checkpoint.exists() or not receipt.exists() or read(receipt)['checkpoint_sha256'] != sha(checkpoint):
        raise RuntimeError('Completed training checkpoint missing or changed: '+str(checkpoint))
    model = train.build_model(cfg).cuda()
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    missing, unexpected = model.load_state_dict(payload['model'],strict=False)
    if unexpected or any(not k.startswith(('visual.model.','text.model.')) for k in missing):
        raise ValueError('Evaluation checkpoint lacks policy/module weights')
    model.eval(); device = torch.device('cuda')
    cls = rollout._load_partgym_env(ROOT/'PartInstruct')
    _, planner_cls = native_classes()
    simcfg = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    simcfg.data_root = plan['config']['data_root']; simcfg.device = 'cpu'
    fullcfg = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    rollout.configure_partgym_agentview_pcd(simcfg,fullcfg)
    for index in range(shard,len(plan['trials']),plan['config']['evaluation']['shards']):
        trial = plan['trials'][index]; scenario = plan['scenarios'][trial['scenario_index']]
        path = args.out_dir/'evaluations'/arm/f'{index:03d}.json'
        if path.exists():
            raise FileExistsError(path)
        ref = read(oracle_path(args.out_dir,index)) if oracle_path(args.out_dir,index).exists() else None
        available = bool(ref and ref['status'] == 'completed')
        if arm == 'oracle_pose' and trial['needs_grasp_pose'] and not available:
            dump(path, {'status':'blocked_missing_oracle','trial_index':index,**trial,
                        'reference_status':ref['status'] if ref else 'missing'})
            continue
        tracker = {'phase':0,'grasp_acquired':False,'initial_phase_last_grasp':False,'grasp_steps':[]}
        part = scenario['episode']['chain_params'][0]['params'].get('part_grasp','')
        class MonitoredEnv(cls):
            def reset(self,*values,**kwargs):
                obs = super().reset(*values,**kwargs)
                state = physical_state(self)
                if available:
                    for key in ('tcp_xyzw_xyz','robot_q','object_xyz','object_xyzw'):
                        np.testing.assert_allclose(state[key],ref['initial_state'][key],atol=1e-6,rtol=0)
                return obs
            def step(self,action,*values,**kwargs):
                result = super().step(action,*values,**kwargs)
                if trial['needs_grasp_pose'] and tracker['phase'] == 0:
                    before = self.world.sim_time
                    grasp = bool(planner_cls.predicate_grasping(self,part_name=part)) and float(action[6]) <= .02
                    if self.world.sim_time != before:
                        raise ValueError('Read-only grasp scoring advanced physics')
                    tracker['grasp_acquired'] |= grasp
                    tracker['initial_phase_last_grasp'] = grasp
                    tracker['grasp_steps'].append(grasp)
                return result
        goal = torch.tensor(ref['goal_pose'] if available else [0.,0.,0.,1.,0.,0.,0.],device=device).reshape(1,7)
        valid = torch.tensor([float(available)],device=device)
        class Policy(torch.nn.Module):
            use_action_deltas = model.use_action_deltas
            def forward(self,rgb,instructions,**kwargs):
                out = model(rgb,instructions,goal_pose=goal,goal_valid=valid,**kwargs)
                tracker['phase'] = int(out['predicted_phase'][0])
                return out
        try:
            result = rollout._run_trial(env_cls=MonitoredEnv,model=Policy(),model_cfg=cfg,partgym_cfg=simcfg,
                config_path=Path(plan['configs'][arm]),obj_class=scenario['obj_class'],task_type=scenario['task_type'],
                split='test1',trial_index=index,episode_seed=trial['seed'],device=device,
                max_steps=plan['config']['evaluation']['max_steps'],execute_steps=1,record=False,
                gif_fps=10.,gif_width=400.,gif_every_n=2,diagnostics=False,predicate_diagnostics=False,
                contact_diagnostics=False,diagnostic_every=20,termination_mode='legacy',min_phase_cycles=2,
                out_dir=args.out_dir/'evaluations'/arm,selected_episode=scenario['episode'])
            dump(path, {'status':'completed','trial_index':index,**trial, 'arm':arm,
                        'oracle_available':available, 'checkpoint_sha256':sha(checkpoint),
                        'grasp':tracker, **result})
        except BaseException as error:
            dump(path, {'status':'runtime_error','trial_index':index,**trial,'error':repr(error)})
            raise


def report(args, plan):
    groups = {}; paired = []; unknown = []
    for arm in ARMS:
        rows = [read(p) for p in sorted((args.out_dir/'evaluations'/arm).glob('*.json'))]
        done = [r for r in rows if r['status'] == 'completed']
        grasp = [r for r in done if r['needs_grasp_pose']]
        groups[arm] = {'completed':len(done),'planned':len(plan['trials']),
            'task_success':sum(r['success'] for r in done),
            'grasp_trials':len(grasp),'grasp_acquired':sum(r['grasp']['grasp_acquired'] for r in grasp),
            'initial_phase_end_grasp':sum(r['grasp']['initial_phase_last_grasp'] for r in grasp)}
        by_index = {r['trial_index']:r for r in rows}
        unknown += [{'arm':arm,**by_index.get(i,{'status':'missing','trial_index':i})}
                    for i in range(len(plan['trials']))
                    if by_index.get(i,{}).get('status') != 'completed']
    for i,t in enumerate(plan['trials']):
        paths = [args.out_dir/'evaluations'/arm/f'{i:03d}.json' for arm in ARMS]
        if all(p.exists() for p in paths):
            rows = [read(p) for p in paths]
            if all(r['status']=='completed' for r in rows):
                paired.append({'trial_index':i,'seed':t['seed'],
                    'control_task':rows[0]['success'],'oracle_task':rows[1]['success'],
                    'grasp_eligible':t['needs_grasp_pose'],
                    'control_grasp':rows[0]['grasp']['initial_phase_last_grasp'],
                    'oracle_grasp':rows[1]['grasp']['initial_phase_last_grasp']})
    refs = [read(p) for p in (args.out_dir/'oracle_references').glob('*.json')]
    training_audit = {'status':'incomplete'}
    if all((args.out_dir/'models'/a/'completed.json').exists() for a in ARMS):
        starts = [read(args.out_dir/'models'/a/'initialization_verification.json') for a in ARMS]
        histories = [read(args.out_dir/'models'/a/'history.json') for a in ARMS]
        shared = starts[0]['shared_policy_sha256'] == starts[1]['shared_policy_sha256']
        order = [[(r['epoch'],r['optimizer_steps'],r['batch_order_sha256']) for r in h] for h in histories]
        training_audit = {'status':'matched' if shared and order[0]==order[1] else 'mismatch',
                          'shared_initial_weights_equal':shared, 'batch_order_and_budget_equal':order[0]==order[1]}
    status = 'completed' if len(paired)==len(plan['trials']) and not unknown and training_audit['status']=='matched' else 'incomplete'
    if training_audit['status']=='mismatch':
        status = 'invalid_training_match'
    primary = {}
    eligible = [r for r in paired if r['grasp_eligible']]
    for arm in ('control','oracle'):
        primary[arm] = {'paired_task_trials':len(paired),'paired_task_success':sum(r[arm+'_task'] for r in paired),
                        'paired_grasp_trials':len(eligible),'paired_grasp_success':sum(r[arm+'_grasp'] for r in eligible)}
    summary = {'status':status,'groups':groups,'paired_completed':len(paired),'paired':paired,
               'primary_paired_comparison':primary,'training_match_audit':training_audit,
               'unknown':unknown,'oracle_reference_status':dict(Counter(r['status'] for r in refs)),
               'limits':plan['limits']}
    dump(args.out_dir/'summary.json',summary); dump(args.out_dir/'status.json',summary)
    lines = ['# Original visual Transformer + GT grasp-pose guidance','',
             f'**Status: {status}.**', '', plan['protocol'], '',
             '## Primary comparison: identical completed case/seed pairs', '',
             '| Arm | Full task success | Initial-phase-end part grasp |',
             '| --- | ---: | ---: |']
    for arm,r in primary.items():
        lines.append(f'| {arm} | {r["paired_task_success"]}/{r["paired_task_trials"]} | '
                     f'{r["paired_grasp_success"]}/{r["paired_grasp_trials"]} |')
    lines += ['', 'Coverage gaps restrict this paired population; consult the full denominator and oracle-reference status below.', '',
             '## All completed trials and coverage', '',
             '| Arm | Completed / planned | Full task success | Initial grasp acquired | Grasp at initial-phase end |',
             '| --- | ---: | ---: | ---: | ---: |']
    for arm,r in groups.items():
        lines.append(f'| {arm} | {r["completed"]}/{r["planned"]} | {r["task_success"]}/{r["completed"]} | '
                     f'{r["grasp_acquired"]}/{r["grasp_trials"]} | {r["initial_phase_end_grasp"]}/{r["grasp_trials"]} |')
    lines += ['', f'{len(paired)} matched completed comparisons. Missing/error/blocked trials are unknown, not task failures.',
              '', '## Scope', ''] + ['- '+v for v in plan['limits']]
    lines += ['', '[Jobs](submissions.json) · [Plan](plan.json) · [Label audit](label_audit.json) · [Preflight](preflight.json)',
              '[Raw paired outcomes and coverage](summary.json)']
    (args.out_dir/'README.md').write_text('\n'.join(lines)+'\n')


def submit(args,plan):
    if (args.out_dir/'submissions.json').exists():
        raise FileExistsError('Already submitted')
    jobs = {}
    def launch(name,mode,dependencies=(),array=None,gpu=False):
        command = ['sbatch','--parsable','--no-requeue',f'--job-name=p2a-vpose-{name}',
            f'--export=ALL,MODE={mode},P2A_VISUAL_POSE_STUDY={args.out_dir}',
            f'--output={args.out_dir}/{name}_%A_%a.out',f'--error={args.out_dir}/{name}_%A_%a.err',
            '--partition=gpu' if gpu else '--partition=cpu',
            '--cpus-per-task=8' if gpu else '--cpus-per-task=2',
            '--mem=48G' if gpu else '--mem=12G',
            '--time=2-00:00:00' if gpu else '--time=01:00:00']
        if gpu:
            command += ['--gres=gpu:1','--constraint=a40|a100_40gb|a100_80gb|h100_94gb']
        if dependencies:
            command += ['--dependency=afterany:'+':'.join(dependencies)]
        if array:
            command += ['--array='+array]
        command += [str(ROOT/'part2action/slurm/visual_oracle_pose_v1.sbatch')]
        job = subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Invalid Slurm response')
        jobs[name] = job; dump(args.out_dir/'submissions.json',jobs)
        return job
    audit_job = launch('audit','audit')
    smoke = launch('preflight','preflight',[audit_job],gpu=True)
    refs = launch('oracle','collect_oracle',array=f'0-{len(plan["oracle_indices"])-1}%8')
    training = launch('train','train',[smoke],array='0-1%2',gpu=True)
    evaluation = launch('evaluate','evaluate',[training,refs],array=f'0-{2*plan["config"]["evaluation"]["shards"]-1}%2',gpu=True)
    # Report even when upstream failed, so unknowns remain visible.
    command = ['sbatch','--parsable','--no-requeue','--partition=cpu','--cpus-per-task=2','--mem=12G',
        '--time=00:20:00','--job-name=p2a-vpose-report',f'--dependency=afterany:{evaluation}',
        f'--export=ALL,MODE=report,P2A_VISUAL_POSE_STUDY={args.out_dir}',
        f'--output={args.out_dir}/report_%j.out',f'--error={args.out_dir}/report_%j.err',
        str(ROOT/'part2action/slurm/visual_oracle_pose_v1.sbatch')]
    jobs['report'] = subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
    dump(args.out_dir/'submissions.json',jobs); dump(args.out_dir/'status.json',{'status':'queued','jobs':jobs})
    print(json.dumps(jobs,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','audit','preflight','train','collect_oracle','evaluate','report','submit'),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'part2action/configs/pose_learning/visual_oracle_pose_v1.json')
    parser.add_argument('--index',type=int,default=0)
    args = parser.parse_args(); args.out_dir=args.out_dir.resolve()
    if args.mode=='prepare':
        prepare(args); return
    plan=read(args.out_dir/'plan.json'); verify(args.out_dir,plan)
    if args.mode=='train':
        if read(args.out_dir/'preflight.json')['status']!='completed':
            raise ValueError('Visual preflight missing or failed')
        arm=ARMS[args.index]; target=args.out_dir/'models'/arm
        target.mkdir(parents=True,exist_ok=False)
        import sys
        subprocess.run([sys.executable,str(ROOT/'part2action/scripts/train_visual_pose_v1.py'),
            '--config',plan['configs'][arm]],check=True)
        dump(target/'completed.json',{'status':'completed','checkpoint_sha256':sha(target/'last.pt')})
    else:
        globals()[args.mode](args,plan)


if __name__=='__main__':
    main()
