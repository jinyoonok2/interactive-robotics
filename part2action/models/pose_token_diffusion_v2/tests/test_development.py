"""Protect scenario selection and paired simulator-result denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from models.pose_token_diffusion_v2.development import (
    ARMS, FAMILIES, paired_summary, scenario_family, select_scenarios, write_report)


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = []
        for obj in ('mug', 'bottle'):
            for family in FAMILIES:
                first = 'touch_obj' if family == 'touch' else 'grasp_obj'
                chain = [{'skill_name': first, 'params': {}}]
                if family == 'grasp_lift_sequence':
                    chain += [{'skill_name': 'move_gripper', 'params': {'dir_move': 'top', 'grasping': True}},
                              {'skill_name': 'rotate_obj', 'params': {}}]
                self.scenarios.append({'obj_class': obj, 'source_demo': 'demo_1', 'family': family,
                    'episode': {'ep_id': obj+family, 'chain_params': chain, 'task_instruction': 'Original full instruction'}})
        self.plan = {'scope': 'development', 'scenarios': self.scenarios,
            'trials': [{'scenario_index': i, 'seed': 42} for i in range(len(self.scenarios))],
            'checkpoints': {arm: {'sha256': arm} for arm in ARMS},
            'execution_settings': {'max_steps': 250, 'execute_steps': 8}}

    def row(self, arm, index, success=False):
        state = {'tcp_xyzw_xyz': [0.]*7, 'robot_q': [0.]*9,
                 'object_xyz': [0., 0., .2], 'object_xyzw': [0., 0., 0., 1.]}
        return {'trial_index': index, **self.plan['trials'][index],
            'status': 'completed', 'checkpoint_sha256': arm, 'success': success,
            'max_steps': 250, 'execute_steps': 8, 'num_actions': 250,
            'phase_advances': 0, 'rollout_video': '/tmp/example.mp4',
            'monitor': {'initial_state': state, 'grasp_acquired': success, 'grasp_lift_prefix': False}}

    def test_selection_is_deterministic_and_preserves_full_lift_task(self):
        original = copy.deepcopy(self.scenarios)
        a = select_scenarios(self.scenarios, ('bottle', 'mug'))
        b = select_scenarios(list(reversed(self.scenarios)), ('bottle', 'mug'))
        self.assertEqual(a, b)
        self.assertEqual(self.scenarios, original)
        self.assertEqual(len(a), 6)
        self.assertEqual(len(a[2]['episode']['chain_params']), 3)
        self.assertEqual(a[2]['episode']['task_instruction'], 'Original full instruction')

    def test_sideways_motion_is_not_a_lift(self):
        episode = copy.deepcopy(self.scenarios[2]['episode'])
        episode['chain_params'][1]['params']['dir_move'] = 'left'
        self.assertIsNone(scenario_family(episode))

    def test_missing_family_cannot_silently_shrink_selection(self):
        with self.assertRaises(ValueError):
            select_scenarios(self.scenarios[:-1], ('bottle', 'mug'))

    def test_only_matching_completed_trials_enter_rates(self):
        records = {ARMS[0]: {0: self.row(ARMS[0], 0, True), 1: self.row(ARMS[0], 1)},
                   ARMS[1]: {0: self.row(ARMS[1], 0), 1: {**self.row(ARMS[1], 1), 'status': 'runtime_error'}}}
        summary = paired_summary(self.plan, records)
        self.assertEqual(summary['paired_trial_indices'], [0])
        self.assertEqual(summary['by_family']['grasp']['checkpoints'][ARMS[0]]['task_successes'], 1)
        self.assertEqual(summary['coverage'][ARMS[1]]['runtime_error'], 1)
        self.assertEqual(summary['coverage'][ARMS[0]]['not_recorded'], 4)

    def test_different_resets_are_reported_as_coverage_gaps(self):
        rows = {arm: {0: self.row(arm, 0)} for arm in ARMS}
        rows[ARMS[1]][0]['monitor']['initial_state']['object_xyz'][0] = .001
        summary = paired_summary(self.plan, rows)
        self.assertEqual(summary['paired_trial_indices'], [])
        self.assertEqual(summary['reset_mismatch_indices'], [0])

    def test_wrong_seed_or_protocol_cannot_enter_report(self):
        for change in ({'seed': 43}, {'execute_steps': 1}, {'num_actions': 251}, {'checkpoint_sha256': 'wrong'}):
            row = {**self.row(ARMS[0], 0), **change}
            with self.assertRaises(ValueError):
                paired_summary(self.plan, {ARMS[0]: {0: row}})

    def test_pending_report_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = write_report(Path(directory), self.plan, {})
            self.assertEqual(summary['status'], 'incomplete')
            self.assertTrue((Path(directory)/'index.html').exists())

    def test_corrected_report_compares_only_same_reset_before_after_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'before';output=root/'after'
            output.mkdir();records={arm:{} for arm in ARMS}
            for arm in ARMS:
                (reference/arm).mkdir(parents=True)
                for i in range(len(self.scenarios)):
                    old=self.row(arm,i,True)
                    (reference/arm/f'{i:03d}.json').write_text(json.dumps(old))
                    row=self.row(arm,i,False)
                    row['rollout_video']=str(output/arm/'rollout.mp4')
                    records[arm][i]=row
            plan={**self.plan,'reference_development':str(reference)}
            summary=write_report(output,plan,records)
            self.assertEqual(summary['status'],'complete')
            item=summary['before_after'][ARMS[0]+'/grasp']
            self.assertEqual(item['matched_resets'],2)
            self.assertEqual(item['previous_task_successes'],2)
            self.assertEqual(item['corrected_task_successes'],0)


if __name__ == '__main__':
    unittest.main()
