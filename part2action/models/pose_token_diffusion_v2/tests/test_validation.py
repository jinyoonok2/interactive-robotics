import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.pose_token_diffusion_v2.tests.test_model import fixture
from models.pose_token_diffusion_v2.training import forward_batch, validate
from models.pose_token_diffusion_v2.dataset import provided_goals_from_batch
from models.pose_token_diffusion_v2.metrics import component_rows, summarize, write_report
from models.pose_token_diffusion_v2.pilot import stratified_selection
from models.pose_token_diffusion_v2.reporting import project_world
from data.geometry import camera_to_world_numpy


def batch_fixture():
    model,batch=fixture()
    batch['meta']=[{'file':'mug.hdf5','demo':'demo_1','t':0},
                   {'file':'bottle.hdf5','demo':'demo_2','t':0}]
    return model,batch


class ValidationTests(unittest.TestCase):
    def test_physical_units_masks_and_selected_phase_are_reported_separately(self):
        model,batch=batch_fixture();model.eval()
        output=forward_batch(model,batch)
        target=provided_goals_from_batch(batch)
        position=target.position_world.clone();position[0,1]=position[0,0]+torch.tensor([.5,0,0])
        rotation=target.rotation_world.clone();rotation[0,0]=torch.tensor([[0.,-1,0],[1.,0,0],[0.,0,1]])
        goal=replace(output.predicted_goals,position_world=position,rotation_world=rotation,
            opening=target.opening,skill_logits=target.skill_logits,
            applicability_logits=batch['goal_supervision_valid'].float()*20-10)
        encoded=replace(output.encoded,hierarchy={**output.encoded.hierarchy,'predicted_phase':torch.tensor([1,0])})
        output=replace(output,predicted_goals=goal,encoded=encoded)
        rows=component_rows(output,batch)
        self.assertAlmostEqual(rows[0]['goal_position_cm'],0.)
        self.assertAlmostEqual(rows[0]['selected_goal_position_cm'],50.)
        self.assertAlmostEqual(rows[0]['goal_rotation_degrees'],90.,places=4)
        self.assertIsNone(rows[1]['goal_rotation_degrees'])
        self.assertEqual(rows[0]['phase_correct'],0)
        summary=summarize(rows)
        self.assertEqual(summary['goal_rotation_degrees']['count'],1)
        self.assertEqual(summary['applicability']['position']['f1'],1.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report'
            write_report(path,rows,epoch=1,losses={'total':1.},scope='test')
            self.assertTrue((path/'README.md').is_file())
            with (path/'samples.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),2)
            self.assertEqual(json.loads((path/'metrics.json').read_text())['by_skill']['touch_obj']['goal_rotation_degrees']['count'],0)

    def test_validation_writes_human_reports_without_updating_weights_or_rng(self):
        model,batch=batch_fixture();model.train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'validation'
            result=validate(model,[batch],None,path,epoch=0,device=torch.device('cpu'),examples=2)
            self.assertEqual(result['overall']['samples'],2)
            self.assertTrue((path/'index.html').is_file())
            self.assertEqual(len(list((path/'examples').glob('*.png'))),2)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_missing_labels_are_null_instead_of_zero_error(self):
        model,batch=batch_fixture()
        batch['goal_supervision_valid'].zero_()
        rows=component_rows(forward_batch(model,batch),batch)
        result=summarize(rows)
        self.assertEqual(result['goal_position_cm']['count'],0)
        self.assertIsNone(result['goal_position_cm']['mean'])
        self.assertIsNone(result['applicability']['rotation']['recall'])

    def test_pilot_selection_is_deterministic_balanced_and_demo_disjoint(self):
        candidates=[]
        for obj in ('mug','bottle'):
            for kind in ('grasp_obj','touch_obj'):
                for i in range(6):
                    candidates.append({'object':obj,'file':obj+'.hdf5','demo':kind+str(i),'skills':[kind]})
        first,summary=stratified_selection(candidates,per_object=8,validation_fraction=.1,seed=42)
        second,_=stratified_selection(list(reversed(candidates)),per_object=8,validation_fraction=.1,seed=42)
        self.assertEqual(first,second)
        for obj,item in summary.items():
            self.assertEqual(item['demonstrations'],8)
            heldout=[r for r in first if r['object']==obj and r['split']=='validation']
            self.assertEqual({r['skills'][0] for r in heldout},{'grasp_obj','touch_obj'})
        keys=[(r['file'],r['demo']) for r in first]
        self.assertEqual(len(keys),len(set(keys)))

    def test_plot_projection_matches_camera_coordinates(self):
        points=camera_to_world_numpy(np.array([[0.,0.,1.],[0.,0.,-1.]],dtype=np.float32))
        pixels,valid=project_world(points)
        np.testing.assert_allclose(pixels[0],[150.,150.],atol=1e-3)
        self.assertEqual(valid.tolist(),[True,False])


if __name__=='__main__':unittest.main()
