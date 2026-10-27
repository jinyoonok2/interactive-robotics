"""Read-only physics-substep collision and grasp-state logging for evaluation."""
import numpy as np


class PhysicsCollisionLogger:
    def __init__(self, env):
        self.env = env
        self.client = env.world.p
        self.original_step = self.client.stepSimulation
        self.rows = []
        self.action_index = -1
        self.command = None
        self.link_names = {-1: "robot_base"}
        for i in range(self.client.getNumJoints(env.robot.panda)):
            self.link_names[i] = self.client.getJointInfo(env.robot.panda, i)[12].decode()
        self.capture()
        self.client.stepSimulation = self.step

    def step(self, *args, **kwargs):
        result = self.original_step(*args, **kwargs)
        self.capture()
        return result

    def capture(self):
        env, p = self.env, self.client
        position, quaternion = p.getBasePositionAndOrientation(env.obj.uid)
        tcp = env.robot.get_tcp_pose()
        contacts = p.getContactPoints(bodyA=env.robot.panda, bodyB=env.obj.uid)
        self.rows.append({
            "physics_step": len(self.rows), "elapsed_s": len(self.rows)*env.world.dt,
            "action_index": self.action_index, "executed_action": self.command,
            "object_xyz": list(position), "object_quat_xyzw": list(quaternion),
            "tcp_xyz": np.asarray(tcp.translation).tolist(),
            "tcp_quat_xyzw": tcp.rotation.as_quat().tolist(),
            "finger_positions_m": [float(p.getJointState(env.robot.panda, i)[0])
                                   for i in [env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX]],
            "contacts": [{"robot_link": int(c[3]), "robot_link_name": self.link_names[int(c[3])],
                          "object_link": int(c[4]), "position_on_object": list(c[6]),
                          "normal_force_n": float(c[9]), "distance_m": float(c[8]),
                          "is_finger": int(c[3]) in [env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX]}
                         for c in contacts],
        })

    def close(self):
        self.client.stepSimulation = self.original_step
