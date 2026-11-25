import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from diagnostics.partgym_skill_oracle import OracleDecision
from models.pose_token_diffusion_v3.evaluation import SuppliedPlanDiagnostic
from models.pose_token_diffusion_v3.planner import PlannerRequest
from run_pose_token_v3_phase_diagnostic import (
    ARMS, OracleDiagnosticPlanner, OracleDiagnosticRuntime,
    ROOT, checkpoint_for_study, paired_summary, preserved_completion_query,
    sha, training_evidence, verify_training_sources)


class PhaseDiagnosticTests(unittest.TestCase):
    def test_checkpoint_selection_follows_training_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            (root/'train/config_resolved.json').write_text(json.dumps({'train': {'epochs': 10}}))
            path, epoch = checkpoint_for_study(root)
            self.assertEqual(epoch, 10)
            self.assertEqual(path.name, 'epoch_010.pt')
            self.assertEqual(checkpoint_for_study(root, 7)[0].name, 'epoch_007.pt')
            for invalid in (0, 11, True):
                with self.assertRaises(ValueError):
                    checkpoint_for_study(root, invalid)

    def test_historical_two_epoch_studies_remain_explicitly_selectable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            (root/'train/config_resolved.json').write_text(json.dumps({'train': {'epochs': 2}}))
            self.assertEqual(checkpoint_for_study(root)[0].name, 'epoch_002.pt')

    def test_interrupted_training_requires_explicit_opt_in_and_completed_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            (root/'run_status.json').write_text(json.dumps({'status': 'failed'}))
            row = {'epoch': 2, 'optimizer_steps': 3980}
            (root/'train/history.json').write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                training_evidence(root, 2)
            evidence = training_evidence(root, 2, allow_interrupted=True)
            self.assertEqual(evidence['training_status'], 'failed')
            self.assertTrue(evidence['interrupted_checkpoint_diagnostic'])
            self.assertEqual(evidence['completed_epoch'], row)
            with self.assertRaises(ValueError):
                training_evidence(root, 3, allow_interrupted=True)
            (root/'run_status.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaises(ValueError):
                training_evidence(root, 2, allow_interrupted=True)

    def test_operational_changes_need_matching_hash_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            name = str(ROOT/'part2action/models/pose_token_diffusion_v3/evaluation.py')
            (root/'train/source_hashes.json').write_text(json.dumps({name: 'old'}))
            review = root/'review.json'
            item = {'path': name, 'training_sha256': 'old', 'evaluation_sha256': sha(name),
                    'reason': 'Reviewed output/cache routing changes only'}
            review.write_text(json.dumps({'schema': 1, 'changes': [item]}))
            with self.assertRaises(ValueError):
                verify_training_sources(root)
            evidence = verify_training_sources(root, review)
            self.assertEqual(evidence['reviewed_operational_changes'], [item])
            item['evaluation_sha256'] = 'stale'
            review.write_text(json.dumps({'schema': 1, 'changes': [item]}))
            with self.assertRaises(ValueError):
                verify_training_sources(root, review)

    def test_policy_changes_cannot_be_waived_by_operational_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'train').mkdir()
            name = str(ROOT/'part2action/models/pose_token_diffusion_v3/policy.py')
            (root/'train/source_hashes.json').write_text(json.dumps({name: 'old'}))
            review = root/'review.json'
            review.write_text(json.dumps({'schema': 1, 'changes': [{'path': name,
                'training_sha256': 'old', 'evaluation_sha256': sha(name), 'reason': 'Not allowed'}]}))
            with self.assertRaises(ValueError):
                verify_training_sources(root, review)

    def environment(self):
        return SimpleNamespace(world=SimpleNamespace(sim_time=3., recording=True),
            parser=SimpleNamespace(part_pcds={'handle': np.array([[1., 2., 3.]])}))

    def request(self, reason):
        return PlannerRequest('Lift the mug', None, (), .04, None, .99, (), reason)

    def test_each_condition_advances_only_from_its_declared_signal(self):
        skills = ['Grasp the mug at its handle', 'Move upwards']
        learned, oracle = SuppliedPlanDiagnostic(skills), OracleDiagnosticPlanner(skills)
        self.assertEqual(learned.plan(self.request('oracle_skill_complete')).skill.kind, 'grasp_obj')
        self.assertEqual(oracle.plan(self.request('monitor_requested_review')).skill.kind, 'grasp_obj')
        self.assertEqual(learned.plan(self.request('monitor_requested_review')).skill.kind, 'move_gripper')
        self.assertEqual(oracle.plan(self.request('oracle_skill_complete')).skill.kind, 'move_gripper')

    def test_oracle_query_restores_clock_recording_parser_and_random_streams(self):
        env = self.environment()
        class Oracle:
            def is_skill_complete(self, environment, phase, **kwargs):
                assert kwargs == {'exhaustive': True, 'preserve_sim_state': True}
                assert environment.world.recording is False
                environment.world.sim_time += 99.
                environment.parser.part_pcds['handle'] *= 2
                np.random.random()
                random.random()
                return OracleDecision(True, True, 'grasp_obj', phase)
        np.random.seed(19)
        random.seed(19)
        expected_np, expected_py = np.random.random(), random.random()
        np.random.seed(19)
        random.seed(19)
        decision = preserved_completion_query(env, Oracle(), 0)
        self.assertTrue(decision.complete)
        self.assertEqual(env.world.sim_time, 3.)
        self.assertTrue(env.world.recording)
        np.testing.assert_array_equal(env.parser.part_pcds['handle'], [[1., 2., 3.]])
        self.assertEqual(np.random.random(), expected_np)
        self.assertEqual(random.random(), expected_py)

    def test_exception_also_restores_diagnostic_side_effects(self):
        env = self.environment()
        class Oracle:
            def is_skill_complete(self, environment, *args, **kwargs):
                environment.world.sim_time = 99.
                raise ValueError('unsupported')
        with self.assertRaises(ValueError):
            preserved_completion_query(env, Oracle(), 0)
        self.assertEqual(env.world.sim_time, 3.)
        self.assertTrue(env.world.recording)

    def test_monitor_cannot_advance_oracle_condition_without_a_physical_predicate(self):
        class Oracle:
            complete = False
            def is_skill_complete(self, env, phase, **kwargs):
                return OracleDecision(self.complete, True, 'grasp_obj', phase)
        oracle = Oracle()
        runtime = OracleDiagnosticRuntime(object(), OracleDiagnosticPlanner(
            ['Grasp the mug at its handle', 'Move upwards']), self.environment(), oracle)
        runtime.skill = runtime.planner.skills[0]
        runtime.skill_chunks = 2
        runtime.chunks_since_review = 2
        runtime.progress_probability = .99
        reason = runtime._review_reason()
        self.assertEqual(reason, 'monitor_requested_review')
        self.assertEqual(runtime.planner.plan(self.request(reason)).skill.kind, 'grasp_obj')
        oracle.complete = True
        self.assertEqual(runtime._review_reason(), 'oracle_skill_complete')

    def plan_and_records(self):
        settings = {'max_steps': 250, 'execute_steps': 8}
        plan = {'scope': 'diagnostic', 'checkpoint_sha256': 'same',
            'execution_settings': settings, 'scenarios': [{'family': 'grasp_lift_sequence'}],
            'trials': [{'seed': 42, 'scenario_index': 0}]}
        row = {'status': 'completed', 'trial_index': 0, 'scenario_index': 0,
            'seed': 42, 'checkpoint_sha256': 'same', 'execution_settings': settings,
            'initial_state': {'object_xyz': [0., 0., .2]}, 'num_actions': 250,
            'success': False, 'phase_advances': 0}
        return plan, {arm: {0: copy.deepcopy(row)} for arm in ARMS}

    def test_comparison_counts_only_matching_complete_resets(self):
        plan, records = self.plan_and_records()
        records[ARMS[1]][0].update(success=True, phase_advances=1)
        summary = paired_summary(plan, records)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['by_family']['grasp_lift_sequence'][ARMS[1]]['task_successes'], 1)
        records[ARMS[1]][0]['initial_state']['object_xyz'][0] += .01
        summary = paired_summary(plan, records)
        self.assertEqual(summary['status'], 'incomplete')
        self.assertEqual(summary['paired_trial_indices'], [])

    def test_errors_are_coverage_gaps_and_protocol_mismatches_are_rejected(self):
        plan, records = self.plan_and_records()
        records[ARMS[1]][0]['status'] = 'runtime_error'
        self.assertEqual(paired_summary(plan, records)['paired_trial_indices'], [])
        for change in ({'seed': 43}, {'checkpoint_sha256': 'other'}, {'num_actions': 251}):
            plan, records = self.plan_and_records()
            records[ARMS[1]][0].update(change)
            with self.assertRaises(ValueError):
                paired_summary(plan, records)


if __name__ == '__main__':
    unittest.main()
