from types import SimpleNamespace

import numpy as np

from diagnostics.physics_collision import PhysicsCollisionLogger


def test_logger_reads_each_substep_and_restores_simulation_method():
    class Client:
        count = 0
        def stepSimulation(self):
            self.count += 1
            return 42
        def getNumJoints(self, body): return 11
        def getJointInfo(self, body, index): return [None]*12+[f"link_{index}".encode()]
        def getBasePositionAndOrientation(self, body): return [0., 0., 1.], [0., 0., 0., 1.]
        def getJointState(self, body, index): return [.03, 0., 0., 0.]
        def getContactPoints(self, **kwargs):
            if not self.count: return []
            link = 8 if self.count == 1 else 9
            return [(0, 1, 2, link, -1, [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], -.001, 2.)]
    client = Client()
    pose = SimpleNamespace(translation=np.zeros(3), rotation=SimpleNamespace(as_quat=lambda: np.array([0.,0.,0.,1.])))
    env = SimpleNamespace(world=SimpleNamespace(p=client,dt=1/240),obj=SimpleNamespace(uid=2),
                          robot=SimpleNamespace(panda=1,LEFTFINGERIDX=9,RIGHTFINGERIDX=10,get_tcp_pose=lambda:pose))
    original = client.stepSimulation
    logger = PhysicsCollisionLogger(env)
    logger.action_index = 0; logger.command = [0.]*7
    assert client.stepSimulation() == 42
    assert client.stepSimulation() == 42
    assert client.count == 2 and len(logger.rows) == 3
    assert not logger.rows[1]["contacts"][0]["is_finger"]
    assert logger.rows[2]["contacts"][0]["is_finger"]
    assert logger.rows[2]["elapsed_s"] == 2/240
    logger.close()
    assert client.stepSimulation == original
