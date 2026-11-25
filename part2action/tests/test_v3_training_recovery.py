import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from diagnostics.run_health import RunHealth
from models.pose_token_diffusion_v3.checkpoint import checkpoint_state, save_checkpoint
from models.pose_token_diffusion_v3.training import train_step
from models.pose_token_diffusion_v3.training_state import (
    archive_incomplete_epochs, atomic_torch_save, load_training_state, save_training_state)
from models.pose_token_diffusion_v3.tests.test_model import fixture, progress_batch


class RecoveryTests(unittest.TestCase):
    def test_resume_reproduces_next_optimizer_update_and_random_streams(self):
        random.seed(17)
        np.random.seed(17)
        model, batch = fixture()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        train_step(model, batch, optimizer, progress_batch=progress_batch(batch))
        generators = {'shuffle': torch.Generator().manual_seed(99)}
        torch.rand(5, generator=generators['shuffle'])
        sampler = SimpleNamespace(epoch=1)
        identity = {'recipe': 'fixed', 'sources': 'fixed'}
        initial = {'checkpoint_loaded': False}
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)/'latest.pt'
            save_training_state(model, optimizer, state,
                history=[{'epoch': 1, 'optimizer_steps': 1}], generators=generators,
                progress_sampler=sampler, identity=identity, initialization=initial)
            expected_random = (random.random(), np.random.random(), torch.rand(3),
                               torch.rand(3, generator=generators['shuffle']))
            train_step(model, batch, optimizer, progress_batch=progress_batch(batch))
            expected_weights = {k: v.clone() for k, v in checkpoint_state(model).items()}
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            restored, restored_batch = fixture()
            restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.9)
            restored_generators = {'shuffle': torch.Generator().manual_seed(1)}
            restored_sampler = SimpleNamespace(epoch=0)
            payload = load_training_state(restored, restored_optimizer, state,
                generators=restored_generators, progress_sampler=restored_sampler, identity=identity)
            self.assertEqual(payload['history'][-1]['epoch'], 1)
            self.assertEqual(restored_sampler.epoch, 1)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.random(), expected_random[1])
            torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
            torch.testing.assert_close(torch.rand(3, generator=restored_generators['shuffle']), expected_random[3], rtol=0, atol=0)
            train_step(restored, restored_batch, restored_optimizer, progress_batch=progress_batch(restored_batch))
            for name, value in checkpoint_state(restored).items():
                torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
            actual_optimizer = restored_optimizer.state_dict()
            self.assertEqual(actual_optimizer['param_groups'], expected_optimizer['param_groups'])
            for parameter, values in actual_optimizer['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, expected_optimizer['state'][parameter][name], rtol=0, atol=0)

    def test_loader_restart_does_not_consume_shuffle_randomness(self):
        data = TensorDataset(torch.arange(35))
        shuffle, workers = torch.Generator().manual_seed(42), torch.Generator().manual_seed(4042)
        def loader(order, worker):
            return DataLoader(data, batch_size=8, sampler=RandomSampler(data, generator=order), generator=worker)
        first = loader(shuffle, workers)
        list(first)
        saved_order, saved_workers = shuffle.get_state(), workers.get_state()
        expected = torch.cat([row[0] for row in first])
        resumed_order, resumed_workers = torch.Generator(), torch.Generator()
        resumed_order.set_state(saved_order)
        resumed_workers.set_state(saved_workers)
        actual = torch.cat([row[0] for row in loader(resumed_order, resumed_workers)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_atomic_failure_preserves_last_good_snapshot_and_previous_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'latest.pt'
            atomic_torch_save(path, {'value': 1})
            atomic_torch_save(path, {'value': 2})
            with patch('models.pose_token_diffusion_v3.training_state.torch.save', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    atomic_torch_save(path, {'value': 3})
            self.assertEqual(torch.load(path, weights_only=True)['value'], 2)
            self.assertEqual(torch.load(path.with_name('latest_previous.pt'), weights_only=True)['value'], 1)
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_old_model_only_checkpoints_cannot_silently_resume_training(self):
        model, _ = fixture()
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model_only.pt'
            save_checkpoint(model, path)
            before = {k: v.clone() for k, v in checkpoint_state(model).items()}
            with self.assertRaises(ValueError):
                load_training_state(model, optimizer, path, generators={}, progress_sampler=SimpleNamespace(epoch=0), identity={})
            for key, value in checkpoint_state(model).items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_partial_epoch_artifacts_are_retained_when_replaying_an_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'epoch_001.pt').write_text('complete')
            (root/'epoch_002.pt').write_text('partial')
            (root/'validation/epoch_002').mkdir(parents=True)
            (root/'validation/epoch_002/samples.csv').write_text('partial')
            archive_incomplete_epochs(root, 1)
            self.assertTrue((root/'epoch_001.pt').exists())
            self.assertFalse((root/'epoch_002.pt').exists())
            self.assertEqual(len(list((root/'interrupted_artifacts').glob('*/epoch_002.pt'))), 1)

    def test_failure_report_records_batch_context_and_full_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'health'
            with self.assertRaisesRegex(ValueError, 'bad sample'):
                with RunHealth(path, stack_dump_seconds=0) as health:
                    health.context.update(epoch=3, step=1751, action_samples=[{'demo': 'demo_2', 't': 8}])
                    raise ValueError('bad sample')
            report = json.loads((path/'failure.json').read_text())
            self.assertEqual(report['exception_type'], 'ValueError')
            self.assertEqual(report['context']['epoch'], 3)
            self.assertEqual(report['context']['action_samples'][0]['t'], 8)
            self.assertIn('ValueError: bad sample', (path/'traceback.log').read_text())


if __name__ == '__main__':
    unittest.main()
