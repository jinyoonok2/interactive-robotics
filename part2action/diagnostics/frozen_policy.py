"""Opt-in frozen-policy gate/target diagnostics; no training-source changes."""
import torch
from data.geometry import tcp_position_torch

MODES = ('gate_oracle', 'gate_none', 'target_identity', 'target_zero', 'target_shift_x')


class FrozenPolicyDiagnostic:
    def __init__(self, mode, oracle_mask_provider=None):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.oracle_mask_provider = oracle_mask_provider
        self.env = None
        self.begin_trial(42)

    def begin_trial(self, seed):
        self.seed = int(seed)
        self.index = 0
        self.phase = 0
        self.last = {}

    def attach(self, model):
        if getattr(model, 'contact_conditioning_stage', 'late') != 'late':
            raise ValueError('Diagnostics currently require the late path')
        if self.mode.startswith('gate_'):
            if not model.use_part_gate or model.use_skill_gate:
                raise ValueError('Requires the active unconditioned visual part gate')
            model.part_gate.register_forward_hook(self._gate_hook(model))
        else:
            if (not model.use_unified_contact_conditioning or not model.contact_conditioning_enabled
                    or model.action_head_type != 'diffusion' or model.split_action_heads):
                raise ValueError('Requires enabled monolithic diffusion target conditioning')
            model.contact_geometry_conditioner.register_forward_pre_hook(self._target_hook)
        original = model.forward
        device = next(model.parameters()).device

        def forward(*args, **kwargs):
            current = kwargs.get('current_phase')
            self.phase = int(current.item()) if current is not None else 0
            self.last = {'mode': self.mode, 'inference_index': self.index,
                         'target_phase_index': self.phase}
            index = self.index
            self.index += 1
            if self.mode.startswith('gate_'):
                return original(*args, **kwargs)
            # Each condition gets the same DDPM random stream at the same
            # trial/inference index, regardless of earlier branch consumption.
            noise_seed = (self.seed * 1000003 + index) % (2**63 - 1)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            self.last['diffusion_noise_seed'] = noise_seed
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(noise_seed)
                return original(*args, **kwargs)
        model.forward = forward
        return model

    def _gate_hook(self, model):
        def hook(module, inputs, output):
            tokens = inputs[0]
            if self.mode == 'gate_none':
                self.last['gate_applied'] = False
                return tokens, torch.full_like(output[1], -80.)
            if self.oracle_mask_provider is None:
                raise ValueError('Missing oracle-mask provider')
            mask, metadata = self.oracle_mask_provider(self.env, self.phase)
            self.last.update(metadata)
            if mask is None:
                self.last['gate_applied'] = False
                return tokens, torch.full_like(output[1], -80.)
            mask = torch.as_tensor(mask, device=tokens.device, dtype=torch.float32)[None]
            logits = model.mask_to_gate_logits(mask, tokens.shape[1])
            self.last['gate_applied'] = True
            self.last['oracle_mask_fraction'] = float(mask.mean())
            return model.apply_part_gate(tokens, logits.to(output[1].dtype))
        return hook

    def _target_hook(self, module, inputs):
        features, contact, tcp, *tail = inputs
        if self.mode == 'target_identity':
            changed = contact
        elif self.mode == 'target_zero':
            # Relative input zero matches the conditioner's masked-input bias.
            changed = tcp_position_torch(tcp).to(contact)
        else:
            changed = contact + contact.new_tensor([.05, 0., 0.])
        self.last['predicted_target_xyz'] = contact.detach().cpu().tolist()
        self.last['conditioned_target_xyz'] = changed.detach().cpu().tolist()
        return (features, changed, tcp, *tail)
