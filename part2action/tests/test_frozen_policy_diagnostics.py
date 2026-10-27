import copy
import torch
from torch import nn
from models.heads import PartMaskGate, ContactGeometryConditioner
from models.part2action_model import Part2ActionModel
from diagnostics.frozen_policy import FrozenPolicyDiagnostic


class Policy(nn.Module):
    contact_conditioning_stage='late'
    use_part_gate=True
    use_skill_gate=False
    use_unified_contact_conditioning=True
    contact_conditioning_enabled=True
    action_head_type='diffusion'
    split_action_heads=False
    mask_to_gate_logits=staticmethod(Part2ActionModel.mask_to_gate_logits)
    apply_part_gate=staticmethod(Part2ActionModel.apply_part_gate)

    def __init__(self):
        super().__init__()
        self.part_gate=PartMaskGate(8)
        self.contact_geometry_conditioner=ContactGeometryConditioner(8, mode='normalized_global')

    def forward(self, x, contact=None, tcp=None):
        if contact is None:
            return self.part_gate(x)[0]
        y=self.contact_geometry_conditioner(x,contact,tcp,enabled=True)[0]
        self.noise=torch.randn_like(y)
        return y+self.noise


def test_gate_identity_oracle_and_unsupported_preserve_parameters():
    torch.manual_seed(9)
    original=Policy().eval();x=torch.ones(1,4,8)
    state=copy.deepcopy(original.state_dict())
    none=copy.deepcopy(original)
    FrozenPolicyDiagnostic('gate_none').attach(none)
    assert torch.equal(none(x),x)
    context=FrozenPolicyDiagnostic('gate_oracle',lambda e,p:(torch.tensor([[1.,0.],[0.,1.]]),{'supported':True}))
    oracle=copy.deepcopy(original);context.attach(oracle)
    y=oracle(x)
    assert y[0,0].mean()>1.99 and y[0,1].mean()<1.01
    assert context.last['gate_applied']
    unsupported=copy.deepcopy(original)
    c=FrozenPolicyDiagnostic('gate_oracle',lambda e,p:(None,{'supported':False,'reason':'move'}));c.attach(unsupported)
    assert torch.equal(unsupported(x),x) and not c.last['gate_applied']
    for model in (none,oracle,unsupported):
        for key,value in state.items():assert torch.equal(model.state_dict()[key],value)


def test_target_conditions_share_noise_and_restore_rng():
    torch.manual_seed(17)
    original=Policy().eval();x=torch.randn(1,4,8)
    contact=torch.tensor([[.2,.1,.3]])
    tcp=torch.tensor([[0.,0.,0.,1.,.1,.1,.1]])
    outputs={};noise=[]
    for mode in ['target_identity','target_zero','target_shift_x']:
        model=copy.deepcopy(original)
        c=FrozenPolicyDiagnostic(mode);c.attach(model);c.begin_trial(43)
        before=torch.random.get_rng_state().clone()
        y=model(x,contact,tcp)
        assert torch.equal(before,torch.random.get_rng_state())
        changed=torch.tensor(c.last['conditioned_target_xyz'])
        expected=original.contact_geometry_conditioner(x,changed,tcp,enabled=True)[0]
        torch.testing.assert_close(y,expected+model.noise,rtol=0,atol=0)
        noise.append(model.noise.clone());outputs[mode]=y
        c.begin_trial(43)
        torch.testing.assert_close(model(x,contact,tcp),y,rtol=0,atol=0)
        for k,v in original.state_dict().items():assert torch.equal(model.state_dict()[k],v)
    for n in noise[1:]:assert torch.equal(n,noise[0])
    assert not torch.equal(outputs['target_identity'],outputs['target_zero'])
    assert not torch.equal(outputs['target_identity'],outputs['target_shift_x'])


def test_reject_disabled_target_input():
    model=Policy();model.contact_conditioning_enabled=False
    try:FrozenPolicyDiagnostic('target_zero').attach(model)
    except ValueError:pass
    else:raise AssertionError('A disabled conditioning path must not be tested as an active intervention')
