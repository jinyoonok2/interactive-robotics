"""Protect the shared untrained initialization used by matched scratch runs."""
import copy
import unittest

import torch
from torch import nn

from train import policy_state_digest, verify_untrained_initialization


class InitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        self.state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        self.initial = {'model': self.state, 'epoch': -1,
                        'initialization_kind': 'fresh_random_policy',
                        'policy_state_sha256': policy_state_digest(self.state)}

    def test_same_shared_state_overrides_different_random_initializations(self):
        expected = verify_untrained_initialization(self.initial)
        for seed in [7, 99]:
            torch.manual_seed(seed)
            other = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
            self.assertNotEqual(policy_state_digest(other.state_dict()), expected)
            other.load_state_dict(self.state, strict=True)
            self.assertEqual(policy_state_digest(other.state_dict()), expected)

    def test_reject_trained_checkpoint_even_if_tensor_fingerprint_matches(self):
        for field, value in [('epoch', 29), ('initialization_kind', 'trained_policy')]:
            checkpoint = copy.deepcopy(self.initial)
            checkpoint[field] = value
            with self.assertRaises(ValueError):
                verify_untrained_initialization(checkpoint)

    def test_reject_modified_initial_weights(self):
        checkpoint = copy.deepcopy(self.initial)
        checkpoint['model']['0.weight'][0, 0] += .1
        with self.assertRaises(ValueError):
            verify_untrained_initialization(checkpoint)

    def test_fingerprint_includes_tensor_shape_and_handles_scalars(self):
        scalar = {'scale': torch.tensor(.01), 'steps': torch.tensor(100)}
        self.assertEqual(policy_state_digest(scalar), policy_state_digest(copy.deepcopy(scalar)))
        self.assertNotEqual(policy_state_digest({'x':torch.arange(4)}),
                            policy_state_digest({'x':torch.arange(4).reshape(2,2)}))


if __name__ == '__main__':
    unittest.main()
