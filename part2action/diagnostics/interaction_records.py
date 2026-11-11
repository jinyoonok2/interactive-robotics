"""Pre-intervention observations and explicit outcomes for grasp replay records."""
import numpy as np
from scipy.spatial.transform import Rotation

from diagnostics.interaction_tolerances import perturb_actions

OUTCOME_LIMITS = {'lift_min_m':.03, 'bilateral_min':.8, 'target_contact_min':.8, 'drift_max_m':.01}


def decision_frame(actions, window, candidates):
    """First changed command across the candidate set; observe BEFORE this action."""
    first=[]
    for candidate in candidates:
        changed=perturb_actions(actions,window,candidate)
        indices=np.flatnonzero(np.any(np.abs(changed-actions[:len(changed)])>1e-8,axis=1))
        if len(indices):first.append(int(indices[0]))
    return min(first) if first else max(0,window['close']-11)


def candidate_vector(condition, hz=20.):
    return np.r_[condition['translation_m'],np.deg2rad(condition['rotation_deg']),condition['close_offset_steps']/hz].astype(np.float32)


def outcome_components(row):
    margins={'lift':row['lift_m']-OUTCOME_LIMITS['lift_min_m'],
             'bilateral':row['hold']['bilateral_fraction']-OUTCOME_LIMITS['bilateral_min'],
             'target_contact':row['hold']['target_contact_fraction']-OUTCOME_LIMITS['target_contact_min'],
             'drift':OUTCOME_LIMITS['drift_max_m']-row['hold_relative_position_drift_m']}
    passed={key:bool(value>=0) for key,value in margins.items()}
    return {'thresholds':OUTCOME_LIMITS,'margins':margins,'passes':passed,
            'failure_causes':[key for key,good in passed.items() if not good]}


def motion_delta(tcp, reference, frame):
    """Achieved displacement vs identity at the SAME nominal closing frame."""
    tcp,reference=np.asarray(tcp),np.asarray(reference)
    rotation=Rotation.from_quat(reference[frame,:4]).inv()*Rotation.from_quat(tcp[frame,:4])
    return {'frame':frame,'world_translation_m':(tcp[frame,4:7]-reference[frame,4:7]).tolist(),
            'reference_tcp_local_rotation_vector_rad':rotation.as_rotvec().tolist()}


def depth_points(depth, intrinsic, view, count=1024):
    """Full visible scene, deterministic pixel sampling; no GT semantic filtering."""
    y,x=np.nonzero(np.isfinite(depth)&(depth>0)&(depth<3.))
    if not len(x):raise ValueError('No valid RGB-D scene points')
    keep=np.linspace(0,len(x)-1,count,dtype=int);x,y=x[keep],y[keep]
    z=depth[y,x]
    camera=np.column_stack(((x-intrinsic[0,2])*z/intrinsic[0,0],
                            (y-intrinsic[1,2])*z/intrinsic[1,1],z)).astype(np.float32)
    gl=camera.copy();gl[:,1:3]*=-1
    world=(np.column_stack((gl,np.ones(count)))@np.linalg.inv(view).T)[:,:3]
    return camera,world.astype(np.float32)


def observe(scene, config):
    """Read-only native camera optics with an explicit CPU TinyRenderer backend."""
    p=scene.world.p;w,h=int(config.render_width),int(config.render_height)
    view=np.asarray(p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(config.cam_static_target),distance=config.cam_static_dist,
        yaw=config.cam_static_yaw,pitch=config.cam_static_pitch,roll=0,upAxisIndex=config.up_axis_index)).reshape(4,4,order='F')
    projection=np.asarray(p.computeProjectionMatrixFOV(fov=60,aspect=w/h,
        nearVal=config.cam_near,farVal=config.cam_far)).reshape(4,4,order='F')
    result=p.getCameraImage(width=w,height=h,viewMatrix=view.flatten(order='F'),
        projectionMatrix=projection.flatten(order='F'),renderer=p.ER_TINY_RENDERER)
    rgb=np.asarray(result[2],dtype=np.uint8).reshape(h,w,-1)[:,:,:3]
    buffer=np.asarray(result[3]).reshape(h,w)
    depth=(config.cam_far*config.cam_near/(config.cam_far-(config.cam_far-config.cam_near)*buffer)).astype(np.float32)
    intrinsic=np.array([[projection[0,0]*w/2,0,w/2],[0,projection[1,1]*h/2,h/2],[0,0,1]],dtype=np.float32)
    camera,world=depth_points(depth,intrinsic,view)
    return {'rgb':rgb.transpose(2,0,1),'depth_m':depth,'pcd_camera_m':camera,'pcd_world_m':world,
            'tcp_pose_xyzw_xyz':scene.robot.get_tcp_pose().to_list().astype(np.float32),
            'finger_position_m':np.array([scene.robot.get_gripper_state()],dtype=np.float32),
            'joint_states':np.asarray(scene.robot.get_joint_states(),dtype=np.float32),
            'camera_intrinsic':intrinsic,'world_to_camera_opengl':view.astype(np.float32)}
