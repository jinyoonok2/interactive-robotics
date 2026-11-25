"""Exercise the actual rollout loop: command budgets, feedback, early termination."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diagnostics.evaluation_protocol import execution_settings, align_native_environment
from rollout_partgym import rollout_one


class Environment:
    success_at = None
    truncate_at = None
    instances = []

    def __init__(self,**kwargs):
        self.task_instruction = 'Touch the mug'
        self.steps = 0
        self.closed = False
        self.instances.append(self)

    def observation(self):
        return {'agentview_rgb':np.full((4,4,3),self.steps%255,dtype=np.uint8)}

    def reset(self):
        return self.observation()

    def step(self,action):
        self.steps += 1
        success = self.steps == self.success_at
        return self.observation(),0.,success or self.steps == self.truncate_at,{
            'Success':success,'Steps':self.steps,'Completion Rate':float(success)}

    def close(self):
        self.closed = True


class Policy(torch.nn.Module):
    use_action_deltas = False

    def __init__(self,horizon=8):
        super().__init__()
        self.horizon = horizon
        self.observations = []

    def forward(self,rgb,instructions,**kwargs):
        self.observations.append(rgb.clone())
        return {'action_chunk':torch.zeros(1,self.horizon,7),
            'plan_slots':torch.zeros(1,2,4),'slot_valid_logits':torch.ones(1,2),
            'predicted_phase':torch.zeros(1,dtype=torch.long),
            'termination_logits':torch.full((1,),-100.)}


def run(policy,env=Environment,**settings):
    with tempfile.TemporaryDirectory() as directory:
        return rollout_one(env_cls=env,model=policy,model_cfg={'data':{'n_obs_steps':2}},
            partgym_cfg={},config_path=Path('unused.yaml'),obj_class='mug',task_type='1',split='test1',
            trial_index=0,episode_seed=42,device=torch.device('cpu'),max_steps=settings.get('max_steps',250),
            execute_steps=settings.get('execute_steps',8),record=False,gif_fps=10,gif_width=400,
            gif_every_n=2,diagnostics=False,predicate_diagnostics=False,contact_diagnostics=False,
            diagnostic_every=20,termination_mode='legacy',min_phase_cycles=2,out_dir=Path(directory))


class ExecutionTests(unittest.TestCase):
    def test_250_actions_are_32_predictions_with_a_two_action_final_chunk(self):
        model = Policy()
        row = run(model)
        self.assertEqual(row['num_actions'],250)
        self.assertEqual(len(model.observations),32)
        self.assertEqual(row['num_policy_predictions'],32)
        self.assertFalse(row['success'])
        self.assertFalse(torch.equal(model.observations[0],model.observations[1]))
        self.assertTrue(Environment.instances[-1].closed)

    def test_success_stops_inside_the_chunk_and_truncation_is_not_success(self):
        class Successful(Environment): success_at=11
        row = run(Policy(),Successful)
        self.assertEqual(row['num_actions'],11)
        self.assertTrue(row['success'])
        class Truncated(Environment): truncate_at=11
        row = run(Policy(),Truncated)
        self.assertEqual(row['num_actions'],11)
        self.assertFalse(row['success'])

    def test_legacy_cadence_remains_explicit_and_short_chunks_fail(self):
        row = run(Policy(),max_steps=7,execute_steps=1)
        self.assertEqual(row['num_actions'],7)
        self.assertEqual(row['num_policy_predictions'],7)
        with self.assertRaises(ValueError): run(Policy(horizon=2))

    def test_native_thresholds_preserve_policy_observations_and_runtime_paths(self):
        ours={'release_distance_world_z':.1,'shape_meta':{'tcp':7},'data_root':'local','device':'cpu'}
        author={'release_distance_world_z':.15,'shape_meta':{'part_mask':1},'data_root':'other','device':'cuda:0'}
        align_native_environment(ours,author)
        self.assertEqual(ours,{'release_distance_world_z':.15,'shape_meta':{'tcp':7},'data_root':'local','device':'cpu'})
        self.assertEqual(execution_settings('test1')['max_steps'],250)
        self.assertEqual(execution_settings('test4')['max_steps'],400)
        self.assertEqual(execution_settings('test1')['execute_steps'],8)


if __name__ == '__main__':
    unittest.main()
