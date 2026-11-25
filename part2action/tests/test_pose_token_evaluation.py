"""Coverage and pairing checks for the new trained-checkpoint benchmark."""
import unittest
import json
import tempfile
from pathlib import Path

from run_pose_token_evaluation import aggregate, metric, require_smoke


class PoseTokenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'scope': 'test', 'seeds': [42, 43], 'reference_gaps': [2],
            'scenarios': [{'obj_class': 'mug'}, {'obj_class': 'bottle'}],
            'trials': [{'scenario_index': 0, 'seed': 42, 'needs_grasp_pose': True},
                       {'scenario_index': 1, 'seed': 42, 'needs_grasp_pose': False},
                       {'scenario_index': 0, 'seed': 43, 'needs_grasp_pose': True}]}

    def row(self, index, success):
        return {'status': 'completed', **self.plan['trials'][index], 'success': success,
            'grasp': {'grasp_acquired': success, 'initial_phase_last_grasp': success}}

    def test_reference_gaps_are_excluded_from_both_paired_denominators(self):
        rows = {'disabled': {0: self.row(0, False), 1: self.row(1, True), 2: self.row(2, True)},
                'provided': {0: self.row(0, True), 1: self.row(1, True),
                    2: {'status': 'blocked_missing_oracle', **self.plan['trials'][2]}}}
        result = aggregate(self.plan, rows)
        self.assertEqual(result['paired_trial_indices'], [0, 1])
        self.assertEqual(result['task']['successes'], {'disabled': 1, 'provided': 2})
        self.assertEqual(result['task']['difference_percentage_points'], 50.)
        self.assertEqual(result['initial_phase_end_grasp']['trials'], 1)
        self.assertEqual(result['coverage']['provided']['blocked_missing_oracle'], 1)

    def test_missing_and_runtime_errors_remain_unknown_not_failures(self):
        rows = {'disabled': {0: self.row(0, False)},
                'provided': {0: {'status': 'runtime_error', **self.plan['trials'][0]}}}
        result = aggregate(self.plan, rows)
        self.assertEqual(result['task']['trials'], 0)
        self.assertIsNone(result['task']['rates']['provided'])
        self.assertEqual(result['coverage']['disabled']['not_recorded'], 2)
        self.assertEqual(result['coverage']['provided']['runtime_error'], 1)

    def test_mismatched_scenario_seed_is_rejected(self):
        row = self.row(0, True)
        row['seed'] = 999
        with self.assertRaises(ValueError):
            aggregate(self.plan, {'disabled': {0: row}, 'provided': {}})

    def test_discordant_successes_are_counted_by_pair(self):
        rows = {'disabled': {0: self.row(0, True), 1: self.row(1, False)},
                'provided': {0: self.row(0, False), 1: self.row(1, True)}}
        result = metric(rows, [0, 1], 'task')
        self.assertEqual(result['provided_only_success'], 1)
        self.assertEqual(result['disabled_only_success'], 1)
        self.assertEqual(result['difference_percentage_points'], 0.)

    def test_different_execution_protocol_cannot_enter_a_new_study(self):
        plan = {**self.plan,'max_steps':250,'execute_steps':8,
                'execution_settings':{'max_steps':250,'execute_steps':8}}
        row = {**self.row(0,True),'max_steps':120,'execute_steps':1,'num_actions':120}
        with self.assertRaises(ValueError):
            aggregate(plan,{'disabled':{0:row},'provided':{}})
        row.update(max_steps=250,execute_steps=8,num_actions=251)
        with self.assertRaises(ValueError):
            aggregate(plan,{'disabled':{0:row},'provided':{}})

    def test_old_smoke_receipts_cannot_gate_the_new_execution_protocol(self):
        plan={'execution_settings':{},'max_steps':250,'execute_steps':8,
              'checkpoints':{arm:{'sha256':arm} for arm in ('disabled','provided')}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for arm in plan['checkpoints']:
                path=root/'smoke'/arm/'runtime_passed.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'status':'passed','checkpoint_sha256':arm,
                    'max_steps':120,'execute_steps':1}))
            with self.assertRaises(ValueError):require_smoke(root,plan)
            for arm in plan['checkpoints']:
                path=root/'smoke'/arm/'runtime_passed.json'
                path.write_text(json.dumps({'status':'passed','checkpoint_sha256':arm,
                    'max_steps':250,'execute_steps':8}))
            require_smoke(root,plan)


if __name__ == '__main__':
    unittest.main()
