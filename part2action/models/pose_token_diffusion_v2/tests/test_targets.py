import unittest
import numpy as np

from data.geometry import camera_to_world_numpy
from models.pose_token_diffusion_v2.targets import phase_targets,labels_at,TargetSettings,skill_kind,fingertip_proxy_poses


def demonstration():
    poses=np.zeros((25,7),dtype=np.float32);poses[:,3]=1
    poses[:,4]=.4;poses[:,5]=.1;poses[:,6]=np.linspace(.2,.4,25)
    actions=np.zeros((25,7),dtype=np.float32);actions[:3,6]=.04;actions[5:10,6]=.04;actions[20:,6]=.04
    instructions=sum(([name]*5 for name in ['Grasp body','Touch cap','Move up','Reorient cap forward','Release object']),[])
    poses[19,:4]=[0.,0.,2**-.5,2**-.5]
    # Invert the existing static affine camera transform, rather than treating
    # training-only part clouds as world coordinates.
    origin=camera_to_world_numpy(np.zeros((1,3),dtype=np.float32))[0]
    rotation=camera_to_world_numpy(np.eye(3,dtype=np.float32))-origin
    world=fingertip_proxy_poses(poses,TargetSettings().touch_probe_offset_m)[:,4:7]
    world[:,0]+=.1;world[7,0]-=.09
    camera=(world-origin)@np.linalg.inv(rotation)
    cloud=np.concatenate([camera[:,None],np.ones((25,1,1),dtype=np.float32)],-1)
    return actions,poses,instructions,cloud


class TargetTests(unittest.TestCase):
    def test_grasp_uses_measured_preclosure_pose_not_close_command(self):
        actions,poses,text,cloud=demonstration()
        goals=phase_targets(actions,poses,text,part_pcd=cloud,max_slots=5)
        self.assertEqual(goals[0]['target_t'],2)
        np.testing.assert_array_equal(goals[0]['pose'],poses[2])
        self.assertTrue(goals[0]['component_valid'].all())
        after=labels_at(goals,4,5)
        self.assertEqual(after['goal_supervision_valid'][0].tolist(),[False,False,True])

    def test_touch_is_geometry_proxy_with_unconstrained_orientation(self):
        actions,poses,text,cloud=demonstration()
        goals=phase_targets(actions,poses,text,part_pcd=cloud,max_slots=5)
        self.assertEqual(goals[1]['target_t'],7)
        self.assertEqual(goals[1]['quality'],'closest_fingertip_to_part_proxy')
        self.assertEqual(goals[1]['component_valid'].tolist(),[True,False,False])
        self.assertAlmostEqual(goals[1]['touch_proxy_to_part_distance_m'],.01,places=5)
        # The target remains measured TCP, not the offset fingertip position.
        np.testing.assert_array_equal(goals[1]['pose'],poses[7])
        self.assertFalse(labels_at(goals,8,5)['goal_supervision_valid'].any())

    def test_missing_or_far_touch_geometry_is_not_a_valid_pose_label(self):
        actions,poses,text,cloud=demonstration()
        self.assertFalse(phase_targets(actions,poses,text,max_slots=5)[1]['component_valid'].any())
        goals=phase_targets(actions,poses,text,part_pcd=cloud,settings=TargetSettings(.005),max_slots=5)
        self.assertFalse(goals[1]['component_valid'].any())

    def test_fingertip_offset_uses_tcp_rotation_and_invalid_poses_are_skipped(self):
        poses=np.array([[0.,2**-.5,0.,2**-.5,1.,2.,3.],
                        [0.,0.,0.,0.,1.,2.,3.]],dtype=np.float32)
        probe=fingertip_proxy_poses(poses,.05)
        np.testing.assert_allclose(probe[0,4:7],[1.05,2.,3.],atol=1e-6)
        self.assertTrue(np.isnan(probe[1,4:7]).all())

    def test_later_skill_goals_change_and_keep_real_reference_frames(self):
        actions,poses,text,cloud=demonstration()
        goals=phase_targets(actions,poses,text,part_pcd=cloud,max_slots=5)
        self.assertEqual([g['kind'] for g in goals],['grasp_obj','touch_obj','move_gripper','rotate_obj','release_obj'])
        self.assertEqual([g['target_t'] for g in goals],[2,7,14,19,24])
        self.assertGreater(goals[2]['pose'][6],goals[0]['pose'][6])
        np.testing.assert_allclose(goals[3]['pose'],poses[19],rtol=1e-6,atol=1e-7)
        for t,active in [(1,0),(6,1),(12,2),(16,3),(22,4)]:
            labels=labels_at(goals,t,5)
            self.assertEqual(labels['goal_active_phase'],active)
            self.assertEqual(np.flatnonzero(labels['goal_supervision_valid'].any(-1)).tolist(),[active])

    def test_invalid_pose_and_unknown_skill_are_masked(self):
        actions,poses,text,cloud=demonstration();poses[2]=np.nan
        goals=phase_targets(actions,poses,text,part_pcd=cloud,max_slots=5)
        self.assertFalse(goals[0]['component_valid'].any())
        self.assertTrue(np.isfinite(goals[0]['pose']).all())
        text=['Unrecognised operation']*25
        goal=phase_targets(actions,poses,text,max_slots=1)[0]
        self.assertEqual(goal['reason'],'unsupported_skill')
        with self.assertRaises(ValueError):phase_targets(actions,poses,text[:-1])
        self.assertEqual(skill_kind('Reorient the cap'),'rotate_obj')


if __name__=='__main__':unittest.main()
