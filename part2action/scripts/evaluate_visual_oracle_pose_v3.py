"""Evaluation recovery V3: import the reference-path helper and gate full runs.

V1 models/data/training and the failed V2 adapter stay frozen. A real full-budget
rollout smoke check in each arm must pass before the recovered evaluation starts.
"""
import argparse
from pathlib import Path
from run_visual_oracle_pose_v1 import ROOT, ARMS, read, dump, sha, verify, report, oracle_path

def evaluate(args, plan):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from _common import load_yaml, resolve_paths
    import train_visual_pose_v1 as train
    import rollout_partgym as rollout
    from diagnostics.execution_trace import physical_state
    from generate_logged_references import native_classes
    arm = ARMS[args.index // plan['config']['evaluation']['shards']]
    shard = args.index % plan['config']['evaluation']['shards']
    smoke = args.mode == 'smoke'
    if not smoke:
        require_smoke(args.out_dir)
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
    trial_indices = [0] if smoke else range(shard,len(plan['trials']),plan['config']['evaluation']['shards'])
    output = args.out_dir/'evaluation_recovery_v3'/'smoke'/arm if smoke else args.out_dir/'evaluations'/arm
    for index in trial_indices:
        trial = plan['trials'][index]; scenario = plan['scenarios'][trial['scenario_index']]
        path = output/f'{index:03d}.json'
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
            result = rollout.rollout_one(env_cls=MonitoredEnv,model=Policy(),model_cfg=cfg,partgym_cfg=simcfg,
                config_path=Path(plan['configs'][arm]),obj_class=scenario['obj_class'],task_type=scenario['task_type'],
                split='test1',trial_index=index,episode_seed=trial['seed'],device=device,
                max_steps=plan['config']['evaluation']['max_steps'],execute_steps=1,record=False,
                gif_fps=10.,gif_width=400.,gif_every_n=2,diagnostics=False,predicate_diagnostics=False,
                contact_diagnostics=False,diagnostic_every=20,termination_mode='legacy',min_phase_cycles=2,
                out_dir=output,selected_episode=scenario['episode'])
            dump(path, {'status':'completed','trial_index':index,**trial, 'arm':arm,
                        'oracle_available':available, 'checkpoint_sha256':sha(checkpoint),
                        'grasp':tracker, 'evaluation_adapter':'v3', **result})
            if smoke:
                if int(result['num_actions']) < 1:
                    raise RuntimeError('Smoke test must execute policy-generated actions')
                dump(output/'completed.json', {'status':'completed', 'arm':arm,
                    'checkpoint_sha256':sha(checkpoint), 'trial_index':index,
                    'max_steps':plan['config']['evaluation']['max_steps'],
                    'num_actions':result['num_actions'],
                    'scope':'full-budget runtime smoke only; not a benchmark success-rate result'})
        except BaseException as error:
            dump(path, {'status':'runtime_error','trial_index':index,**trial,'error':repr(error)})
            raise


def require_smoke(root):
    for arm in ARMS:
        path = root/'evaluation_recovery_v3'/'smoke'/arm/'completed.json'
        checkpoint = root/'models'/arm/'last.pt'
        if not path.exists() or not checkpoint.exists():
            raise RuntimeError('Both trained-arm rollout smoke tests must finish first')
        receipt = read(path)
        if receipt.get('status') != 'completed' or receipt.get('checkpoint_sha256') != sha(checkpoint):
            raise RuntimeError('Missing or stale trained-arm smoke receipt: '+arm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','evaluate','report'),required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args(); args.out_dir=args.out_dir.resolve()
    plan=read(args.out_dir/'plan.json'); verify(args.out_dir,plan)
    revision=read(args.out_dir/'evaluation_revision_v3.json')
    for name,expected in revision['source_sha256'].items():
        if sha(name)!=expected:
            raise RuntimeError('Evaluation revision source changed: '+name)
    if args.mode == 'smoke':
        if args.index not in (0,1):
            raise ValueError('Smoke array indexes select control=0 and oracle_pose=1')
        args.index *= plan['config']['evaluation']['shards']
    {'smoke':evaluate,'evaluate':evaluate,'report':report}[args.mode](args,plan)


if __name__=='__main__':
    main()
