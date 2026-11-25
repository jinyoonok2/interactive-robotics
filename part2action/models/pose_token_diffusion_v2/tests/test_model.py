from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.tests.test_model import Vision,Language
from models.pose_token_diffusion.contracts import Observation
from models.pose_token_diffusion.checkpoint import save_checkpoint as save_v1
from models.pose_token_diffusion.tests.test_model import fixture as fixture_v1
from models.pose_token_diffusion_v2.config import PoseTokenDiffusionV2Config
from models.pose_token_diffusion_v2.policy import build_policy
from models.pose_token_diffusion_v2.dataset import observation_from_batch,provided_goals_from_batch
from models.pose_token_diffusion_v2.losses import compute_losses
from models.pose_token_diffusion_v2.training import train_step
from models.pose_token_diffusion_v2.checkpoint import checkpoint_state,save_checkpoint,load_checkpoint
from models.pose_token_diffusion_v2.runtime import PoseTokenV2RolloutPolicy


def fixture(source='predicted'):
    torch.manual_seed(71)
    config=PoseTokenDiffusionV2Config(hidden_dim=16,action_horizon=3,img_size=28,attention_heads=4,
        decoder_layers=1,hierarchy_layers=1,diffusion_steps=4,visual_token_grid=2,
        scene_token_budget=4,local_token_budget=2,pose_source=source)
    model=build_policy(config,backbones=FrozenBackbones(Vision(),Language()))
    points=torch.tensor([[[-.2,-.1,1.],[-.1,.15,1.],[0.,0.,.9],[.1,-.2,1.],[.2,.2,1.]]]).repeat(2,1,1)
    tcp=torch.tensor([[0.,0.,0.,1.,-.5,.05,.4]]).repeat(2,1)
    batch=dict(rgb=torch.rand(2,2,3,28,28),agentview_pcd=points,
        task_instruction=['grasp body then lift','touch cap then move'],tcp_pose=tcp,
        gripper_state=torch.full((2,1),.04),action_chunk=torch.rand(2,3,7),
        goal_poses=tcp[:,None].repeat(1,4,1),goal_opening=torch.zeros(2,4),
        goal_component_valid=torch.zeros(2,4,3,dtype=torch.bool),
        goal_supervision_valid=torch.zeros(2,4,3,dtype=torch.bool),
        goal_skill_kind=torch.tensor([[0,2,5,5],[1,2,5,5]]),
        skill_plan=[['grasp body','move up'],['touch cap','move left']],
        skill_valid_mask=torch.tensor([[1.,1.,0.,0.]]).repeat(2,1),
        phase_index=torch.zeros(2,dtype=torch.long),phase_termination=torch.zeros(2),
        point_part_labels=torch.tensor([[1.,1.,0.,0.,0.]]).repeat(2,1),
        point_part_valid=torch.ones(2),part_mask=torch.ones(2,28,28))
    batch['goal_poses'][:,:,4:7]+=torch.arange(4)[None,:,None]*.02
    batch['goal_component_valid'][:,0,:]=True
    batch['goal_component_valid'][:,1,:]=True
    batch['goal_supervision_valid'][:,0,:]=True
    batch['goal_component_valid'][1,0,1:]=False
    batch['goal_supervision_valid'][1,0,1:]=False
    return model,batch


class ModelTests(unittest.TestCase):
    def test_actual_cross_attention_and_action_loss_reach_predicted_goals(self):
        model,batch=fixture()
        out=model(observation_from_batch(batch),target_actions=batch['action_chunk'])
        F.mse_loss(out.actions['action_noise_pred'].float(),out.actions['action_noise']).backward()
        for name in ('position','rotation','opening','skill','applicability'):
            gradient=getattr(model.blocks['goal_head'],name).weight.grad
            self.assertIsNotNone(gradient,name)
            self.assertGreater(float(gradient.abs().sum()),0,name)
        self.assertGreater(float(model.blocks['grounding'].key.weight.grad.abs().sum()),0)
        self.assertFalse(any(p.grad is not None for p in model.blocks['encoder'].backbones.parameters()))
        self.assertEqual(out.predicted_goals.position_world.shape,(2,4,3))
        self.assertEqual(out.pose_memory.tokens.shape,(2,3,16))

    def test_all_supervision_is_finite_and_updates_real_modules(self):
        model,batch=fixture()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        before=model.blocks['goal_head'].position.weight.detach().clone()
        metrics,gradients=train_step(model,batch,optimizer,collect_gradients=True)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        self.assertFalse(torch.equal(before,model.blocks['goal_head'].position.weight))
        self.assertGreater(gradients['blocks.goal_head.position.weight'],0)

    def test_phase_change_selects_different_provided_goal_without_changing_v1(self):
        model,batch=fixture('provided');model.eval()
        observation=observation_from_batch(batch)
        # Explicit cached learned-plan-shaped values isolate the selector here.
        first=model(observation,provided_goals=provided_goals_from_batch(batch),target_actions=batch['action_chunk'])
        hierarchy=first.encoded.hierarchy
        cached=replace(observation,instructions=None,plan_slots=hierarchy['plan_slots'],
            slot_valid_logits=torch.full_like(hierarchy['slot_valid_logits'],10.),force_current_phase=True)
        zero=model(replace(cached,current_phase=torch.zeros(2,dtype=torch.long)),
            provided_goals=provided_goals_from_batch(batch),target_actions=batch['action_chunk'])
        one=model(replace(cached,current_phase=torch.ones(2,dtype=torch.long)),
            provided_goals=provided_goals_from_batch(batch),target_actions=batch['action_chunk'])
        torch.testing.assert_close(zero.selected_goal.position_world[:,0],batch['goal_poses'][:,0,4:7])
        torch.testing.assert_close(one.selected_goal.position_world[:,0],batch['goal_poses'][:,1,4:7])
        self.assertFalse(torch.equal(zero.pose_memory.relative_error,one.pose_memory.relative_error))

    def test_touch_orientation_mask_has_no_effect_on_memory(self):
        model,batch=fixture('provided');observation=observation_from_batch(batch)
        goal=provided_goals_from_batch(batch).select(torch.zeros(2,dtype=torch.long))
        a=model.blocks['pose_tokenizer'](observation_encoded(model,observation),goal)
        rotation=goal.rotation_world.clone();rotation[1,0]*=-1
        b=model.blocks['pose_tokenizer'](observation_encoded(model,observation),replace(goal,rotation_world=rotation))
        torch.testing.assert_close(a.tokens[1],b.tokens[1])
        self.assertFalse(a.valid_mask[1,1])

    def test_no_goal_labels_or_expert_phase_enter_observation_or_deployment(self):
        model,batch=fixture();model.eval()
        observation=observation_from_batch(batch)
        self.assertNotIn('goal_poses',vars(observation))
        self.assertIsNone(observation.current_phase)
        torch.manual_seed(5);first=model(observation,target_actions=batch['action_chunk'])
        altered=dict(batch,goal_poses=torch.rand_like(batch['goal_poses']),phase_index=torch.ones(2,dtype=torch.long))
        torch.manual_seed(5);second=model(observation_from_batch(altered),target_actions=batch['action_chunk'])
        torch.testing.assert_close(first.actions['action_noise_pred'],second.actions['action_noise_pred'],rtol=0,atol=0)
        with self.assertRaises(ValueError):model(observation,provided_goals=provided_goals_from_batch(batch))

    def test_disabled_memory_retains_vision_and_amp_is_finite(self):
        model,batch=fixture('disabled')
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=model(observation_from_batch(batch),target_actions=batch['action_chunk'])
            loss,_=compute_losses(out,batch,model)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(int(out.pose_memory.tokens.count_nonzero()),0)
        self.assertFalse(out.pose_memory.valid_mask.any())
        self.assertGreater(float(model.blocks['encoder'].visual_projection.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_roundtrip_identity_and_no_overwrite(self):
        model,batch=fixture();model.eval()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'v2.pt';save_checkpoint(model,path)
            payload=torch.load(path,weights_only=True)
            self.assertEqual(payload['architecture'],'pose_token_diffusion_v2')
            self.assertFalse(any(k.startswith('blocks.encoder.backbones.') for k in payload['model']))
            restored,_=fixture();load_checkpoint(restored,path);restored.eval()
            torch.manual_seed(9);a=model(observation_from_batch(batch)).actions['action_chunk']
            torch.manual_seed(9);b=restored(observation_from_batch(batch)).actions['action_chunk']
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaises(FileExistsError):save_checkpoint(model,path)
            old,_,_=fixture_v1();oldpath=Path(folder)/'v1.pt';save_v1(old,oldpath)
            before={k:v.clone() for k,v in checkpoint_state(model).items()}
            with self.assertRaises(ValueError):load_checkpoint(model,oldpath)
            self.assertTrue(all(torch.equal(v,checkpoint_state(model)[k]) for k,v in before.items()))

    def test_native_adapter_preserves_plan_cache_and_produces_commands(self):
        model,batch=fixture();model.eval();adapter=PoseTokenV2RolloutPolicy(model)
        first=adapter(batch['rgb'],batch['task_instruction'],agentview_pcd=batch['agentview_pcd'],
            tcp_pose=batch['tcp_pose'],gripper_state=batch['gripper_state'])
        self.assertEqual(first['action_chunk'].shape,(2,3,7))
        second=adapter(batch['rgb'],None,agentview_pcd=batch['agentview_pcd'],tcp_pose=batch['tcp_pose'],
            gripper_state=batch['gripper_state'],plan_slots_override=first['plan_slots'],
            slot_valid_logits_override=first['slot_valid_logits'],current_phase=first['predicted_phase'],force_current_phase=True)
        torch.testing.assert_close(first['predicted_phase'],second['predicted_phase'])
        provided,_=fixture('provided')
        with self.assertRaises(ValueError):PoseTokenV2RolloutPolicy(provided)

    def test_v1_config_cannot_silently_select_v2(self):
        with self.assertRaises(ValueError):PoseTokenDiffusionV2Config.from_mapping({'architecture':'pose_token_diffusion_v1'})
        with self.assertRaises(ValueError):PoseTokenDiffusionV2Config(progress_enabled=True)


def observation_encoded(model,observation):
    model.eval()
    return model.blocks['encoder'](observation)


if __name__=='__main__':unittest.main()
