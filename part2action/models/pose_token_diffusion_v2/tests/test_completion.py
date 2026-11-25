"""Completion boundaries, late observations, query supervision and loss isolation."""
from pathlib import Path
import copy
import json
import tempfile
import unittest

import h5py
import numpy as np
import torch

from models.pose_token_diffusion_v2.completion import (
    completion_records, CompletionDataset, BalancedCompletionSampler,
    completion_loss, validate_completion)
from models.pose_token_diffusion_v2.tests.test_model import fixture
from models.pose_token_diffusion_v2.dataset import observation_from_batch
from models.pose_token_diffusion_v2.training import train_step
from models.pose_token_diffusion.blocks.encoder import SharedObservationEncoder


class CompletionTests(unittest.TestCase):
    def phases(self):
        return [{'start':0,'end':9,'kind':'grasp_obj'},
                {'start':10,'end':19,'kind':'release_obj'}]

    def test_no_anticipation_and_previous_query_survives_transition(self):
        records, masked = completion_records(self.phases(),20,complete_chain=True)
        grasp = [r for r in records if r['query_phase']==0]
        self.assertTrue(all(r['target']==0 for r in grasp if r['t']<9))
        self.assertEqual([r['t'] for r in grasp if r['target']],list(range(9,18)))
        self.assertFalse(any(r['physical_success_verified'] for r in records))
        self.assertEqual(masked,[])

    def test_incomplete_or_untrusted_final_is_masked(self):
        for complete,allow in ((False,True),(True,False)):
            records,masked=completion_records(self.phases(),20,complete_chain=complete,
                                              allow_endpoint_proxy=allow)
            self.assertFalse(any(r['skill']=='release_obj' for r in records))
            self.assertEqual(masked[0]['phase'],1)

    def test_explicit_complete_endpoint_is_a_proxy_not_verified_success(self):
        records,_=completion_records(self.phases(),20,complete_chain=True)
        final=next(r for r in records if r['t']==19)
        self.assertEqual(final['source'],'complete_chain_endpoint_proxy')
        self.assertEqual(final['target'],1)
        self.assertFalse(final['physical_success_verified'])

    def test_balanced_batches_and_reproducible_sampling(self):
        class Data:
            _samples=[{'skill':k,'target':y} for k in ('grasp_obj','release_obj') for y in (0,1)]
        a=BalancedCompletionSampler(Data(),5,42)
        first=list(a)
        self.assertEqual(first,list(BalancedCompletionSampler(Data(),5,42)))
        for start in range(0,len(first),4):
            self.assertEqual(sum(Data._samples[i]['target'] for i in first[start:start+4]),2)
        self.assertNotEqual(first,list(a))

    def test_terminal_observation_loads_without_action_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'demos').mkdir()
            path=root/'demos/mug.hdf5'
            with h5py.File(path,'w') as f:
                demo=f.create_group('data/demo_1');obs=demo.create_group('obs')
                demo.create_dataset('actions',data=np.zeros((20,7)))
                demo.create_dataset('skill_instructions',data=[b'Grasp the mug at its body']*10+[b'Release']*10)
                demo.create_dataset('task_instructions',data=[b'Grasp and release']*20)
                demo.create_dataset('metadata/ep_id',data=b'1')
                obs.create_dataset('agentview_rgb',data=np.stack([np.full((2,2,3),t,dtype=np.uint8) for t in range(20)]))
                obs.create_dataset('agentview_pcd',data=np.ones((20,5,3),dtype=np.float32))
                obs.create_dataset('tcp_pose',data=np.tile([0.,0.,0.,1.,0.,0.,0.],(20,1)))
                obs.create_dataset('gripper_state',data=np.zeros((20,1)))
            (root/'episodes_meta_train.json').write_text(json.dumps({'mug':{'train':{'5':[{
                'ep_id':'1','skill_instructions':['Grasp the mug at its body','Release']}]}}}))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'demonstrations':[{'file':str(path),'object':'mug','demo':'demo_1','split':'train'}]}))
            ds=CompletionDataset(manifest,split='train')
            i=next(i for i,r in enumerate(ds._samples) if r['t']==19)
            sample=ds[i]
            self.assertAlmostEqual(float(sample['rgb'][0,0,0,0]),18/255.)
            self.assertAlmostEqual(float(sample['rgb'][-1,0,0,0]),19/255.)
            self.assertEqual(float(sample['phase_termination']),1.)
            self.assertNotIn('action_chunk',sample)
            ds.close()

    def completion_batch(self,batch):
        return {k:v for k,v in batch.items() if k in
                ('rgb','agentview_pcd','task_instruction','tcp_pose','gripper_state')} | {
            'completion_query_phase':torch.tensor([1,0]),'phase_termination':torch.tensor([1.,0.]),
            'meta':[{'skill':'move_gripper','source':'annotated_transition','target':1},
                    {'skill':'touch_obj','source':'annotated_transition','target':0}]}

    def test_query_labels_select_loss_without_forcing_actor_phase(self):
        model,batch=fixture();model.eval()
        encoded=model.blocks['encoder'](observation_from_batch(batch))
        logits=encoded.hierarchy['completion_logits_by_phase'];logits.retain_grad()
        chosen=encoded.hierarchy['predicted_phase'].clone()
        completion_loss(encoded.hierarchy,self.completion_batch(batch)).backward()
        self.assertEqual(int(logits.grad.count_nonzero()),2)
        self.assertNotEqual(float(logits.grad[0,1]),0.)
        self.assertNotEqual(float(logits.grad[1,0]),0.)
        self.assertTrue(torch.equal(chosen,encoded.hierarchy['predicted_phase']))

    def test_query_bank_preserves_original_encoder_parameters_and_selected_output(self):
        model,batch=fixture();model.eval()
        current=model.blocks['encoder']
        original=SharedObservationEncoder(model.config,copy.deepcopy(current.backbones)).eval()
        original.load_state_dict(current.state_dict(),strict=True)
        obs=observation_from_batch(batch)
        with torch.no_grad():
            a,b=current(obs),original(obs)
        for name in ('visual_tokens','scene_tokens','language_tokens','robot_tokens'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name))
        for name in ('termination_logits','phase_logits','predicted_phase','plan_slots'):
            torch.testing.assert_close(a.hierarchy[name],b.hierarchy[name])

    def test_auxiliary_update_and_validation_are_isolated_from_action_targets(self):
        model,batch=fixture()
        batch['meta']=[{},{}]
        auxiliary=self.completion_batch(batch)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        metrics,grads=train_step(model,batch,optimizer,completion_batch=auxiliary,collect_gradients=True)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertGreater(grads['blocks.encoder.termination_head.net.0.weight'],0.)
        state={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            result=validate_completion(model,[auxiliary],Path(directory)/'report')
            self.assertEqual(result['overall']['count'],2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))


if __name__=='__main__':unittest.main()
