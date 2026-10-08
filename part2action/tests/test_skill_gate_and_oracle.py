"""Tests for skill-conditioned gating and oracle phase termination."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from diagnostics.partgym_skill_oracle import PartGymSkillOracle
from diagnostics.phase_controller import should_advance_phase
from models.heads import PartMaskGate, SkillConditionedPartGate


class _FakePlanner:
    @staticmethod
    def effects_grasp_obj(part_grasp="", **kwargs):
        return {"predicate_grasping": {"params": {"part_name": part_grasp}, "value": True}}

    @staticmethod
    def effects_touch_obj(part_touch="", **kwargs):
        return {"predicate_touching": {"params": {"part_name": part_touch}, "value": True}}

    @staticmethod
    def effects_move_gripper(
        grasping=False,
        touching=False,
        dir_move=(1, 0, 0),
        distance=0.1,
        put_down=False,
        last_gripper_position=(0, 0, 0),
        **kwargs,
    ):
        return {
            "predicate_at_position": {
                "params": {
                    "position": [
                        last_gripper_position[0] + distance,
                        last_gripper_position[1],
                        last_gripper_position[2],
                    ],
                    "position_thred": 0.05,
                },
                "value": True,
            }
        }

    @staticmethod
    def predicate_grasping(env, part_name=""):
        return bool(getattr(env, "grasping", False))

    @staticmethod
    def predicate_touching(env, part_name=""):
        return bool(getattr(env, "touching", False))

    @staticmethod
    def predicate_at_position(env, is_obj=True, position=(0, 0, 0), position_thred=0.02):
        current = getattr(env, "obj_position", (0, 0, 0))
        dist = sum((a - b) ** 2 for a, b in zip(current, position)) ** 0.5
        return dist < position_thred


class _FakeBullet:
    def __init__(self):
        self.saved = 0
        self.restored = []
        self.removed = []

    def saveState(self):
        self.saved += 1
        return 17

    def restoreState(self, *, stateId):
        self.restored.append(stateId)

    def removeState(self, *, stateUniqueId):
        self.removed.append(stateUniqueId)


def test_legacy_part_gate_does_not_consume_skill():
    gate = PartMaskGate()
    visual = torch.randn(2, 8, 384)
    gated, logits = gate(visual)
    assert gated.shape == visual.shape
    assert logits.shape == (2, 8)


def test_skill_gate_consumes_selected_skill_and_changes_with_skill():
    torch.manual_seed(0)
    gate = SkillConditionedPartGate()
    visual = torch.randn(1, 12, 384)
    skill_a = torch.randn(1, 512)
    skill_b = torch.randn(1, 512)
    _, logits_a = gate(visual, skill_a)
    _, logits_b = gate(visual, skill_b)
    assert logits_a.shape == (1, 12)
    assert not torch.allclose(logits_a, logits_b)


def test_phase_advance_requires_completion_cycles_and_next_slot():
    assert should_advance_phase(
        skill_complete=True,
        phase_steps=2,
        min_phase_cycles=2,
        current_phase=0,
        valid_skill_count=2,
    )
    assert not should_advance_phase(
        skill_complete=False,
        phase_steps=10,
        min_phase_cycles=2,
        current_phase=0,
        valid_skill_count=2,
    )
    assert not should_advance_phase(
        skill_complete=True,
        phase_steps=1,
        min_phase_cycles=2,
        current_phase=0,
        valid_skill_count=2,
    )
    assert not should_advance_phase(
        skill_complete=True,
        phase_steps=5,
        min_phase_cycles=2,
        current_phase=1,
        valid_skill_count=2,
    )


def test_oracle_queries_grasp_predicate_and_blocks_on_failure():
    env = SimpleNamespace(
        chain_params=[{"skill_name": "grasp_obj", "params": {"part_grasp": "handle"}}],
        grasping=False,
        initial_obj_pose=SimpleNamespace(translation=(0.0, 0.0, 0.1)),
        config=SimpleNamespace(translate_distance=0.1),
    )
    oracle = PartGymSkillOracle(bullet_planner=_FakePlanner)
    decision = oracle.is_skill_complete(env, 0)
    assert decision.supported
    assert decision.skill_name == "grasp_obj"
    assert decision.complete is False
    assert decision.predicate_results["predicate_grasping"] is False

    env.grasping = True
    decision_ok = oracle.is_skill_complete(env, 0)
    assert decision_ok.complete is True


def test_oracle_unsupported_skill_and_out_of_range_slots():
    env = SimpleNamespace(
        chain_params=[{"skill_name": "unknown_skill", "params": {}}],
        initial_obj_pose=SimpleNamespace(translation=(0.0, 0.0, 0.1)),
        config=SimpleNamespace(translate_distance=0.1),
    )
    oracle = PartGymSkillOracle(bullet_planner=_FakePlanner)
    bad = oracle.is_skill_complete(env, 0)
    assert bad.supported is False
    assert bad.complete is False

    missing = oracle.is_skill_complete(env, 3)
    assert missing.supported is False
    assert "out of range" in missing.reason


def test_oracle_move_gripper_uses_at_position():
    env = SimpleNamespace(
        chain_params=[
            {
                "skill_name": "move_gripper",
                "params": {
                    "grasping": True,
                    "touching": False,
                    "dir_move": (1.0, 0.0, 0.0),
                    "put_down": False,
                },
            }
        ],
        obj_position=(0.1, 0.0, 0.0),
        initial_obj_pose=SimpleNamespace(translation=(0.0, 0.0, 0.0)),
        config=SimpleNamespace(translate_distance=0.1),
    )
    oracle = PartGymSkillOracle(bullet_planner=_FakePlanner)
    decision = oracle.is_skill_complete(env, 0)
    assert decision.supported
    assert decision.complete is True
    assert "predicate_at_position" in decision.predicate_results


def test_exhaustive_predicate_diagnostic_records_all_expected_and_observed():
    env = SimpleNamespace(grasping=False, touching=True)
    effects = {
        "predicate_grasping": {
            "params": {"part_name": "handle"},
            "value": True,
        },
        "predicate_touching": {
            "params": {"part_name": "handle"},
            "value": False,
        },
    }
    oracle = PartGymSkillOracle(bullet_planner=_FakePlanner)
    complete, observed, expected, reason = oracle.evaluate_effects(
        env, effects, exhaustive=True
    )
    assert complete is False
    assert observed == {
        "predicate_grasping": False,
        "predicate_touching": True,
    }
    assert expected == {
        "predicate_grasping": True,
        "predicate_touching": False,
    }
    assert "predicate_grasping expected True got False" in reason
    assert "predicate_touching expected False got True" in reason


def test_predicate_diagnostic_restores_bullet_state():
    bullet = _FakeBullet()
    env = SimpleNamespace(
        chain_params=[{"skill_name": "grasp_obj", "params": {}}],
        grasping=True,
        initial_obj_pose=SimpleNamespace(translation=(0.0, 0.0, 0.0)),
        config=SimpleNamespace(translate_distance=0.1),
        world=SimpleNamespace(p=bullet),
    )
    decision = PartGymSkillOracle(
        bullet_planner=_FakePlanner
    ).is_skill_complete(
        env, 0, exhaustive=True, preserve_sim_state=True
    )
    assert decision.complete
    assert bullet.saved == 1
    assert bullet.restored == [17]
    assert bullet.removed == [17]


def test_training_config_cannot_enable_oracle_termination(tmp_path):
    import importlib.util
    from pathlib import Path

    path = tmp_path / "bad.yaml"
    path.write_text(
        """
heads: [action]
data:
  use_hierarchy: true
  max_skill_slots: 4
  use_pcd: true
  use_contact_xyz: true
  use_part_pcd: false
model:
  use_hierarchy: true
  max_skill_slots: 4
  use_part_gate: true
  use_skill_gate: false
  use_pcd: true
  use_contact_xyz: true
rollout:
  termination_mode: oracle_partgym
losses:
  skill_embedding_weight: 1.0
  slot_validity_weight: 0.25
  phase_weight: 0.5
  termination_weight: 0.5
"""
    )
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_configs.py"
    spec = importlib.util.spec_from_file_location("validate_configs", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    errors = mod.validate(path)
    assert any("oracle_partgym" in e for e in errors)


def test_tcp_xyz_uses_pose_tail():
    from data.geometry import tcp_position_torch

    pose = torch.tensor([[0.1, 0.2, 0.3, 1.0, 4.0, 5.0, 6.0]])
    torch.testing.assert_close(tcp_position_torch(pose), torch.tensor([[4.0, 5.0, 6.0]]))


if __name__ == "__main__":
    test_legacy_part_gate_does_not_consume_skill()
    test_skill_gate_consumes_selected_skill_and_changes_with_skill()
    test_phase_advance_requires_completion_cycles_and_next_slot()
    test_oracle_queries_grasp_predicate_and_blocks_on_failure()
    test_oracle_unsupported_skill_and_out_of_range_slots()
    test_oracle_move_gripper_uses_at_position()
    test_exhaustive_predicate_diagnostic_records_all_expected_and_observed()
    test_predicate_diagnostic_restores_bullet_state()
    test_tcp_xyz_uses_pose_tail()
    print("skill gate / oracle tests passed")
