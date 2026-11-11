"""Observation-only PartGym evaluation of the shared visual pose architecture."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import torch
from omegaconf import OmegaConf

from _common import ROOT, load_yaml, resolve_paths, save_json
from train_grounded_visual_pose_v1 import build_model
from diagnostics.evaluation_scenarios import load_scenarios, rollout_plan
from rollout_partgym import _load_partgym_env, configure_partgym_agentview_pcd, rollout_one


def load_model(cfg, checkpoint, device):
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('architecture') != 'grounded_visual_pose_v1':
        raise ValueError('Expected a trained grounded_visual_pose_v1 checkpoint')
    missing, extra = model.load_state_dict(state['model'], strict=False)
    if extra or any(not key.startswith(('visual.model.', 'text.model.')) for key in missing):
        raise ValueError(f'Incomplete trained checkpoint: {missing}, {extra}')
    model.to(device).eval()
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--ckpt',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--eval-seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--num-episodes',type=int,help='Explicit debugging subset only')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('Require 0 <= shard-index < shards')
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation for visual policy evaluation')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Use a fresh per-shard output directory')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    cfg=resolve_paths(load_yaml(args.config),ROOT)
    cfg['model']['text_device']='cuda'
    model=load_model(cfg,args.ckpt,torch.device('cuda'))
    framework=ROOT.parent/'PartInstruct'
    simpath=framework/'PartInstruct/PartGym/config/config_oracle.yaml'
    sim=OmegaConf.load(simpath)
    sim.data_root='/p/part2action/data/PartInstruct'
    sim.device='cpu'
    full=OmegaConf.load(framework/'PartInstruct/PartGym/config/config.yaml')
    configure_partgym_agentview_pcd(sim,full)
    env_cls=_load_partgym_env(framework)
    from generate_logged_references import native_classes
    _, planner_cls=native_classes()
    scenarios=load_scenarios(Path(sim.data_root)/sim.meta_path,'test1',
        ['bottle','mug','pliers','scissors'],None,args.num_episodes)
    if args.num_episodes is None and len(scenarios) != 149:
        raise ValueError(f'Expected all 149 Test1 scenarios; found {len(scenarios)}')
    checkpoint_sha=hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    save_json(args.out_dir/'plan.json',dict(scenarios=scenarios,seeds=args.eval_seeds,
        shards=args.shards,shard_index=args.shard_index,checkpoint_sha256=checkpoint_sha,
        policy_inputs='RGB history, observed full-scene PCD, task, measured TCP/opening; predicted skill/grounding/pose'))
    records=[]
    for index,(scenario,trial,seed) in enumerate(rollout_plan(scenarios,args.eval_seeds)):
        if index % args.shards != args.shard_index:
            continue
        first_skill=scenario['episode']['chain_params'][0]
        needs_grasp=first_skill['skill_name']=='grasp_obj'
        part=first_skill['params'].get('part_grasp','')
        tracker=dict(phase=0,grasp_acquired=False,initial_phase_last_grasp=False)
        pose_trace=[]
        class MonitoredEnv(env_cls):
            def step(self, action, *values, **kwargs):
                result=super().step(action,*values,**kwargs)
                if needs_grasp and tracker['phase']==0:
                    before=self.world.sim_time
                    grasp=bool(planner_cls.predicate_grasping(self,part_name=part)) and float(action[6])<=.02
                    if self.world.sim_time!=before:
                        raise ValueError('Read-only grasp scoring advanced physics')
                    tracker['grasp_acquired'] |= grasp
                    tracker['initial_phase_last_grasp']=grasp
                return result
        class MonitoredPolicy(torch.nn.Module):
            use_action_deltas=model.use_action_deltas
            def forward(self,rgb,instructions,**kwargs):
                out=model(rgb,instructions,**kwargs)
                tracker['phase']=int(out['predicted_phase'][0])
                pose_trace.append(dict(position_world=out['predicted_pose_position_world'][0].float().cpu().tolist(),
                    rotation_world=out['predicted_pose_rotation_world'][0].float().cpu().tolist(),
                    applicability=float(out['pose_applicability_logits'][0].sigmoid()),
                    phase=tracker['phase']))
                return out
        try:
            result=rollout_one(env_cls=MonitoredEnv,model=MonitoredPolicy(),model_cfg=cfg,partgym_cfg=sim,
                config_path=args.config,obj_class=scenario['obj_class'],task_type=scenario['task_type'],
                split='test1',trial_index=trial,episode_seed=seed,device=torch.device('cuda'),
                max_steps=120,execute_steps=1,record=False,gif_fps=10.,gif_width=400,gif_every_n=2,
                diagnostics=False,predicate_diagnostics=False,contact_diagnostics=False,
                diagnostic_every=20,termination_mode='legacy',min_phase_cycles=2,
                out_dir=args.out_dir,selected_episode=scenario['episode'])
            record=dict(status='completed',population_index=index,needs_grasp_pose=needs_grasp,
                        grasp=tracker,predicted_pose_trace=pose_trace,**result)
        except Exception as error:
            record=dict(status='runtime_error',population_index=index,seed=seed,error=repr(error))
        records.append(record)
        save_json(args.out_dir/f'trial_{index:03d}.json',record)
    completed=[r for r in records if r['status']=='completed']
    grasp_rows=[r for r in completed if r['needs_grasp_pose']]
    save_json(args.out_dir/'summary.json',dict(planned=len(rollout_plan(scenarios,args.eval_seeds)[args.shard_index::args.shards]),
        completed=len(completed),unknown=len(records)-len(completed),
        completed_task_successes=sum(bool(r['success']) for r in completed),
        grasp_trials=len(grasp_rows),grasp_acquired=sum(r['grasp']['grasp_acquired'] for r in grasp_rows),
        initial_phase_end_grasp=sum(r['grasp']['initial_phase_last_grasp'] for r in grasp_rows),
        note='Native part-grasp predicate allows either finger; sustained retention is not implied. Runtime errors are unknown, not physical policy failures; combine shards and matched starts before comparison.'))


if __name__=='__main__':
    main()
