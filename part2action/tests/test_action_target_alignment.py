"""Causal action targets, legacy compatibility and observation-label boundaries."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from data.partinstruct_loader import PartInstructDataset, collate_part2action


class ActionAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/'demo.hdf5'
        self.actions = np.zeros((32, 7), dtype=np.float32)
        self.actions[16:, 0] = .7
        self.actions[:, 3] = np.arange(32)*.01
        self.actions[:10, 6] = .04
        with h5py.File(self.path, 'w') as handle:
            demo = handle.create_group('data/demo_0')
            demo.create_dataset('actions', data=self.actions)
            text_type = h5py.string_dtype()
            instructions = ['Touch the mug at its handle']*18 + ['Grasp the mug at its handle']*14
            demo.create_dataset('skill_instructions', data=instructions, dtype=text_type)
            demo.create_dataset('task_instructions', data=['touch then grasp']*32, dtype=text_type)
            obs = demo.create_group('obs')
            rgb = np.broadcast_to(np.arange(32, dtype=np.uint8)[:, None, None, None], (32, 8, 8, 3))
            obs.create_dataset('agentview_rgb', data=rgb)
            obs.create_dataset('agentview_part_mask', data=np.ones((32, 8, 8), dtype=np.uint8))
            pcd = np.zeros((32, 6, 4), dtype=np.float32)
            pcd[:, :3, 3] = 1
            obs.create_dataset('agentview_pcd', data=pcd[..., :3])
            obs.create_dataset('agentview_part_pcd', data=pcd)
            tcp = np.zeros((32, 7), dtype=np.float32)
            tcp[:, 3] = 1
            tcp[:, 4] = np.arange(32)*.001
            obs.create_dataset('tcp_pose', data=tcp)
            obs.create_dataset('gripper_state', data=np.arange(32)[:, None]*.001)
            obs.create_dataset('joint_states', data=np.ones((32, 7), dtype=np.float32))
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:
            dataset.close()
        self.directory.cleanup()

    def dataset(self, **kwargs):
        options = dict(action_chunk=8, n_obs_steps=2, sample_stride=1,
                       use_pcd=True, use_tcp_pose=True, use_contact_xyz=True,
                       use_point_part_labels=True, use_hierarchy=True, max_skill_slots=4)
        options.update(kwargs)
        dataset = PartInstructDataset([str(self.path)], **options)
        self.datasets.append(dataset)
        return dataset

    def test_next_action_initiates_from_stationary_frame_and_preserves_legacy_samples(self):
        legacy, future = self.dataset(), self.dataset(action_target_offset=1)
        self.assertEqual([s.t for s in legacy._samples], list(range(24)))
        self.assertEqual([s.t for s in legacy._samples], [s.t for s in future._samples])
        np.testing.assert_array_equal(legacy[15]['action_chunk'], self.actions[15:23])
        np.testing.assert_array_equal(future[15]['action_chunk'], self.actions[16:24])
        self.assertEqual(float(legacy[15]['action_chunk'][0, 0]), 0.)
        self.assertAlmostEqual(float(future[15]['action_chunk'][0, 0]), .7)
        history = future[15]['rgb'][:, 0, 0, 0]
        torch.testing.assert_close(history, torch.tensor([14., 15.])/255)
        stride2 = self.dataset(action_target_offset=1, sample_stride=2)
        self.assertNotIn(15, [s.t for s in stride2._samples])

    def test_final_chunk_is_complete_and_does_not_cross_demo_end(self):
        future = self.dataset(action_target_offset=1)
        last = future[len(future)-1]
        np.testing.assert_array_equal(last['action_chunk'], self.actions[24:32])
        self.assertEqual(last['meta']['action_start_t'], 24)
        self.assertEqual(last['meta']['action_last_t'], 31)
        self.assertEqual(last['meta']['action_target_offset'], 1)

    def test_gripper_onset_uses_command_immediately_before_future_chunk(self):
        future = self.dataset(action_target_offset=1)
        self.assertEqual(float(future[9]['gripper_closed'][0]), 1.)
        self.assertEqual(float(future[9]['gripper_loss_weight'][0]), 4.)
        self.assertEqual(float(future[10]['gripper_loss_weight'][0]), 1.)

    def test_skill_boundary_does_not_shift_observation_or_auxiliary_targets(self):
        for mode in ('legacy', 'skill_specific'):
            with self.subTest(mode=mode):
                legacy = self.dataset(contact_label_mode=mode)
                future = self.dataset(contact_label_mode=mode, action_target_offset=1)
                before, after = legacy[17], future[17]
                for key in ('rgb', 'part_mask', 'agentview_pcd', 'tcp_pose', 'gripper_state',
                            'joint_states', 'contact_xy', 'approach_dir', 'contact_xyz',
                            'contact_xyz_valid', 'point_part_labels', 'point_part_valid',
                            'phase_index', 'phase_termination', 'skill_valid_mask'):
                    torch.testing.assert_close(before[key], after[key], msg=key)
                self.assertEqual(after['instruction'], before['instruction'])
                self.assertEqual(after['instruction'], 'Touch the mug at its handle')
                self.assertEqual(int(after['phase_index']), 0)
                self.assertEqual(float(after['phase_termination']), 1.)
                np.testing.assert_array_equal(after['action_chunk'], self.actions[18:26])
                self.assertEqual(int(future[18]['phase_index']), 1)
                batch = collate_part2action([after, future[18]])
                self.assertEqual(batch['action_chunk'].shape, (2, 8, 7))

    def test_invalid_offsets_fail_instead_of_silently_truncating(self):
        for offset in (-1, 2, .5, True, '1'):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.dataset(action_target_offset=offset)

    def test_training_and_offline_evaluation_use_configured_alignment(self):
        from train import build_dataset
        from evaluate import build_eval_loader
        data = dict(train_hdf5=[str(self.path)], action_chunk=8,
                    action_target_offset=1, sample_stride=1, n_obs_steps=2,
                    batch_size=2, eval_batch_size=2, eval_num_workers=0,
                    use_pcd=True, use_tcp_pose=True, use_contact_xyz=True,
                    use_hierarchy=True, max_skill_slots=4)
        dataset = build_dataset(data)
        loader = build_eval_loader({'data': data})
        self.datasets.extend([dataset, loader.dataset])
        self.assertEqual(dataset.action_target_offset, 1)
        self.assertEqual(loader.dataset.action_target_offset, 1)
        np.testing.assert_array_equal(dataset[15]['action_chunk'], self.actions[16:24])
        self.assertEqual(next(iter(loader))['meta'][0]['action_start_t'], 1)

    def test_timing_configs_match_except_target_alignment(self):
        from validate_configs import validate
        folder = ROOT/'configs/architecture_update'
        paths = [folder/f'hierarchical_world_geometry_10epoch_timing_{v}.yaml'
                 for v in ('same_index', 'next_action')]
        configs = [yaml.safe_load(p.read_text()) for p in paths]
        for path, cfg in zip(paths, configs):
            self.assertEqual(validate(path, check_files=True), [])
            self.assertEqual(cfg['data']['sample_stride'], 1)
            self.assertFalse(cfg['model']['contact_conditioning_enabled'])
            self.assertEqual(cfg['data']['val_hdf5'], [])
        self.assertEqual([cfg['data']['action_target_offset'] for cfg in configs], [0, 1])
        for cfg in configs:
            cfg.pop('name'); cfg.pop('output_dir'); cfg['data'].pop('action_target_offset')
        self.assertEqual(configs[0], configs[1])
        invalid = copy.deepcopy(configs[0])
        invalid['data']['action_target_offset'] = True
        path = Path(self.directory.name)/'bad.yaml'
        path.write_text(yaml.safe_dump(invalid))
        self.assertTrue(any('action_target_offset' in error for error in validate(path)))


if __name__ == '__main__':
    unittest.main()
