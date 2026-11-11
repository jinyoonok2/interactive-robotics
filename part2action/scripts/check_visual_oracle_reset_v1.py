"""Verify a separate oracle scene matches the ORIGINAL full visual reset."""
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

from _common import set_seed
from diagnostics.evaluation_scenarios import pin_scenario
from diagnostics.execution_trace import physical_state
from run_visual_oracle_pose_v1 import ROOT, read, dump, verify
from rollout_partgym import _load_partgym_env, configure_partgym_agentview_pcd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.out_dir.resolve()
    plan=read(root/'plan.json'); verify(root,plan)
    index=plan['oracle_indices'][0]
    reference=read(root/'oracle_references'/f'{index:03d}.json')
    if reference['status']!='completed':
        raise ValueError('First reference is unavailable; no reset equivalence can be claimed')
    trial=plan['trials'][index]; scenario=plan['scenarios'][trial['scenario_index']]
    set_seed(trial['seed'])
    cfg=OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    cfg.data_root=plan['config']['data_root']; cfg.device='cpu'
    full=OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    configure_partgym_agentview_pcd(cfg,full)
    cls=_load_partgym_env(ROOT/'PartInstruct')
    env=cls(config=cfg,gui=False,record=False,evaluation=True,skill_mode=False,
            obj_class=scenario['obj_class'],task_type=scenario['task_type'],split='test1',track_samples=False)
    try:
        pin_scenario(env,scenario['episode'],trial['seed']); obs=env.reset()
        state=physical_state(env); errors={}
        for key in ('tcp_xyzw_xyz','robot_q','object_xyz','object_xyzw'):
            errors[key]=float(np.max(np.abs(np.asarray(state[key])-reference['initial_state'][key])))
            np.testing.assert_allclose(state[key],reference['initial_state'][key],atol=1e-6,rtol=0)
        assert {'agentview_rgb','agentview_pcd','tcp_pose','gripper_state'} <= set(obs)
        dump(root/'simulation_reset_preflight.json',{'status':'completed','trial_index':index,
            'max_absolute_differences':errors,'visual_inputs_present':True,
            'RGB_shape':list(np.asarray(obs['agentview_rgb']).shape),
            'scene_PCD_shape':list(np.asarray(obs['agentview_pcd']).shape)})
    finally:
        env.close()


if __name__=='__main__':
    main()
