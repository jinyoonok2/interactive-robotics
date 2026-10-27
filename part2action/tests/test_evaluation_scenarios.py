import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from diagnostics.evaluation_scenarios import load_scenarios, pin_scenario, rollout_plan, success_statistics


class ScenarioTests(unittest.TestCase):
    def test_every_scenario_runs_once_for_every_seed(self):
        scenarios = [{'episode': {'ep_id': str(i)}} for i in range(149)]
        plan = rollout_plan(scenarios)
        self.assertEqual(len(plan), 447)
        keys = {(row['episode']['ep_id'], seed) for row, index, seed in plan}
        self.assertEqual(len(keys), 447)
        self.assertEqual({seed for _, _, seed in plan}, {42,43,44})
        self.assertEqual(plan[:3], [(scenarios[0],0,42),(scenarios[0],1,43),(scenarios[0],2,44)])
        with self.assertRaises(ValueError): rollout_plan(scenarios, [42,42])
        with self.assertRaises(ValueError): rollout_plan(scenarios, [-1])

    def test_report_separates_seed_variation_and_detects_duplicate_results(self):
        rows = [{'obj_class':'mug','task_type':'1','requested_episode_id':str(i),
                 'seed': seed,'success':i<n} for seed,n in [(42,0),(43,1),(44,2)] for i in range(2)]
        report = success_statistics(rows)
        self.assertEqual(report['mean_seed_success_rate'], .5)
        self.assertEqual(report['std_seed_success_rate'], .5)
        self.assertEqual(report['per_seed']['43']['successes'], 1)
        with self.assertRaises(ValueError): success_statistics(rows+[rows[0]])

    def test_release_count_unique_and_order_independent(self):
        path = Path(__file__).resolve().parents[2] / 'datasets/PartInstruct/episodes_meta_test.json'
        rows = load_scenarios(path, 'test1', ['bottle','mug','pliers','scissors'])
        self.assertEqual(len(rows), 149)
        keys = [(r['obj_class'], r['task_type'], r['episode']['ep_id']) for r in rows]
        self.assertEqual(len(set(keys)), 149)
        self.assertEqual(rows, load_scenarios(path, 'test1', ['scissors','pliers','mug','bottle']))
        self.assertEqual(load_scenarios(path, 'test1', ['bottle'], limit=2), rows[:2])

    def test_explicit_reset_preserves_metadata_and_upstream_pose(self):
        episode = {'ep_id':'123', 'obj_id':'object', 'chain_params':[{'skill_name':'grasp_obj'}],
                   'task_instruction':'grasp', 'skill_instructions':['grasp'],
                   'obj_pose':[0,0,0,1,.01,.02,.15], 'obj_scale':.1}
        original = copy.deepcopy(episode)
        for _ in range(2):
            env = SimpleNamespace(seed=lambda value: self.assertEqual(value, 42))
            pin_scenario(env, episode, 42)
            env.sample_chain_params()
            self.assertEqual(env.ep_id, '123')
            self.assertEqual(env.obj_init_position, [.01,.02,.2])
            self.assertEqual(env.obj_init_orientation, [0,0,0,1])
            self.assertEqual(env.chain_params, episode['chain_params'])
            env.chain_params.clear()
        self.assertEqual(episode, original)

    def test_duplicates_and_empty_selection_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'meta.json'
            path.write_text(json.dumps({'mug':{'test1':{'1':[{'ep_id':'1'},{'ep_id':'1'}]}}}))
            with self.assertRaises(ValueError): load_scenarios(path, 'test1')
            with self.assertRaises(ValueError): load_scenarios(path, 'test2')


if __name__ == '__main__':
    unittest.main()
