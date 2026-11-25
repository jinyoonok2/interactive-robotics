import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from models.pose_token_diffusion_v3.dataset import build_dataset, ProgressDataset, collate, observation_from_batch


def create_data(root):
    (root/'demos').mkdir()
    path = root/'demos/mug.hdf5'
    texts = ['Grasp the mug at its handle']*5 + ['Move upwards']*3 + ['Release']*4
    with h5py.File(path, 'w') as handle:
        for number in (1, 2):
            demo = handle.create_group(f'data/demo_{number}')
            actions = np.zeros((12, 7), dtype=np.float32)
            actions[:, 0] = np.arange(12)
            actions[:3, 6] = .04
            actions[8:, 6] = .04
            demo.create_dataset('actions', data=actions)
            demo.create_dataset('skill_instructions', data=[s.encode() for s in texts])
            demo.create_dataset('task_instructions', data=[b'Grasp lift release']*12)
            demo.create_dataset('metadata/ep_id', data=str(number).encode())
            obs = demo.create_group('obs')
            obs.create_dataset('agentview_rgb', data=np.zeros((12, 28, 28, 3), dtype=np.uint8))
            obs.create_dataset('agentview_part_mask', data=np.ones((12, 28, 28, 1), dtype=np.uint8))
            points = np.tile([[0., 0., 1.], [.1, .1, 1.], [-.1, -.1, 1.]], (12, 1, 1)).astype(np.float32)
            obs.create_dataset('agentview_pcd', data=points)
            part = np.concatenate([points, np.ones((12, 3, 1), dtype=np.float32)], -1)
            obs.create_dataset('agentview_part_pcd', data=part)
            poses = np.tile([0., 0., 0., 1., 0., 0., .4], (12, 1)).astype(np.float32)
            poses[:, 6] += np.arange(12)*.01
            obs.create_dataset('tcp_pose', data=poses)
            obs.create_dataset('gripper_state', data=actions[:, 6:7])
    (root/'episodes_meta_train.json').write_text(json.dumps({'mug': {'train': {'x': [
        {'ep_id': str(n), 'skill_instructions': list(dict.fromkeys(texts))} for n in (1, 2)]}}}))
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'demonstrations': [
        {'file': str(path), 'object': 'mug', 'demo': f'demo_{n}', 'split': split}
        for n, split in [(1, 'train'), (2, 'validation')]]}))
    data = {'train_hdf5': [str(path)], 'demo_manifest': str(manifest), 'action_chunk': 8,
        'n_obs_steps': 2, 'max_skill_slots': 4, 'action_target_offset': 1,
        'use_pcd': True, 'use_tcp_pose': True, 'use_hierarchy': True,
        'use_point_part_labels': True, 'use_part_pcd': False, 'use_contact_xyz': False}
    return data, manifest


class DatasetTests(unittest.TestCase):
    def test_short_skill_chunks_are_masked_and_never_contain_next_skill_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = create_data(Path(directory))
            dataset = build_dataset(data)
            self.assertEqual(len(dataset), 12)
            first = dataset[0]
            self.assertEqual(first['action_valid'].tolist(), [True]*4 + [False]*4)
            self.assertEqual(first['action_chunk'][:, 0].tolist(), [1., 2., 3., 4., 4., 4., 4., 4.])
            self.assertEqual(int(dataset[5]['action_valid'].sum()), 2)
            self.assertEqual(int(dataset[11]['action_valid'].sum()), 0)
            self.assertEqual(float(dataset[11]['action_chunk'].sum()), 0.)
            dataset.close()

    def test_reference_pose_is_past_skill_start_and_only_current_text_enters_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = create_data(Path(directory))
            dataset = build_dataset(data)
            sample = dataset[6]
            self.assertEqual(sample['skill_spec'].kind, 'move_gripper')
            self.assertEqual(sample['skill_spec'].interaction, 'maintain_grasp')
            self.assertAlmostEqual(float(sample['skill_start_tcp'][6]), .45, places=6)
            self.assertAlmostEqual(float(sample['tcp_pose'][6]), .46, places=6)
            observation = observation_from_batch(collate([sample]))
            observation.validate()
            self.assertNotIn('phase_index', vars(observation))
            self.assertNotIn('goal_poses', vars(observation))
            self.assertNotIn('part_mask', vars(observation))
            dataset.close()

    def test_whole_demo_split_and_auxiliary_terminal_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            data, manifest = create_data(Path(directory))
            train = build_dataset(data)
            validation = build_dataset({**data, 'split': 'validation'})
            self.assertTrue(set(map(tuple, train.split_info['demonstrations'])).isdisjoint(
                map(tuple, validation.split_info['demonstrations'])))
            progress = ProgressDataset(manifest, split='train')
            i = next(i for i, r in enumerate(progress._samples) if r['t'] == 11 and r['skill'] == 'release_obj')
            terminal = progress[i]
            self.assertEqual(terminal['skill_spec'].kind, 'release_obj')
            self.assertEqual(float(terminal['progress_target']), 1.)
            self.assertEqual(terminal['meta']['source'], 'complete_chain_endpoint_proxy')
            self.assertNotIn('action_chunk', terminal)
            self.assertFalse(terminal['meta']['physical_success_verified'])
            prior = next(i for i, r in enumerate(progress._samples) if r['t'] == 6 and r['query_phase'] == 0)
            self.assertEqual(progress[prior]['skill_spec'].kind, 'grasp_obj')
            self.assertEqual(float(progress[prior]['progress_target']), 1.)
            for dataset in (train, validation, progress):
                dataset.close()

    def test_no_gt_part_cloud_can_be_configured_as_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = create_data(Path(directory))
            with self.assertRaises(ValueError):
                build_dataset({**data, 'use_part_pcd': True})

    def test_touch_opening_is_supervised_even_when_contact_geometry_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = create_data(Path(directory))
            with h5py.File(data['train_hdf5'][0], 'r+') as handle:
                demo = handle['data/demo_1']
                demo['skill_instructions'][:5] = [b'Touch the mug at its handle']*5
            dataset = build_dataset(data)
            sample = dataset[0]
            self.assertEqual(sample['skill_spec'].interaction, 'closed_contact')
            self.assertTrue(bool(sample['goal_supervision_valid'][0, 2]))
            self.assertEqual(float(sample['goal_opening'][0]), 0.)
            self.assertFalse(bool(sample['goal_supervision_valid'][0, 1]))
            dataset.close()


if __name__ == '__main__':
    unittest.main()
