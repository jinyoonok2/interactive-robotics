"""Oracle skill completion from live PartGym predicates.

Evaluation diagnostic only. Predicates must never enter policy tensors,
training observations, or deployable inference paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


SUPPORTED_SKILLS = frozenset(
    {"grasp_obj", "touch_obj", "move_gripper", "rotate_obj", "release_obj"}
)


@dataclass(frozen=True)
class OracleDecision:
    """Privileged completion decision for one policy phase."""

    complete: bool
    supported: bool
    skill_name: str | None
    skill_index: int
    predicate_results: dict[str, bool] = field(default_factory=dict)
    predicate_expected: dict[str, bool] = field(default_factory=dict)
    reason: str = ""


class PartGymSkillOracle:
    """Map ``env.chain_params[phase]`` effects onto live BulletPlanner predicates.

    Assumes the policy phase index aligns with PartGym ``chain_params`` order.
    If alignment is impossible (missing chain, unknown skill, missing args),
    returns ``supported=False`` and never claims completion.
    """

    def __init__(
        self,
        *,
        bullet_planner: Any | None = None,
        translate_distance_fallback: float = 0.1,
    ) -> None:
        self._planner = bullet_planner
        self._translate_distance_fallback = float(translate_distance_fallback)

    def _planner_cls(self, env: Any) -> Any:
        if self._planner is not None:
            return self._planner
        from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner

        return BulletPlanner

    @staticmethod
    def skill_entry(env: Any, skill_index: int) -> tuple[Optional[dict], str]:
        chain = getattr(env, "chain_params", None)
        if not isinstance(chain, Sequence) or not chain:
            return None, "env.chain_params missing or empty"
        if skill_index < 0 or skill_index >= len(chain):
            return None, (
                f"skill_index {skill_index} out of range for chain length {len(chain)}"
            )
        entry = chain[skill_index]
        if not isinstance(entry, Mapping) or "skill_name" not in entry:
            return None, f"chain_params[{skill_index}] lacks skill_name"
        return dict(entry), ""

    def effects_for_skill(
        self, env: Any, skill_name: str, skill_params: Mapping[str, Any]
    ) -> tuple[Optional[dict], str]:
        planner = self._planner_cls(env)
        effects_fn = getattr(planner, f"effects_{skill_name}", None)
        if effects_fn is None:
            return None, f"no effects_{skill_name} on BulletPlanner"
        kwargs = dict(skill_params)
        initial_pose = getattr(env, "initial_obj_pose", None)
        if initial_pose is not None and hasattr(initial_pose, "translation"):
            kwargs.setdefault("last_gripper_position", initial_pose.translation)
        cfg = getattr(env, "config", None)
        distance = getattr(cfg, "translate_distance", self._translate_distance_fallback)
        kwargs.setdefault("distance", distance)
        try:
            effects = effects_fn(**kwargs)
        except TypeError as exc:
            return None, f"effects_{skill_name} arg error: {exc}"
        if not isinstance(effects, dict):
            return None, f"effects_{skill_name} returned non-dict ({type(effects)})"
        return effects, ""

    def evaluate_effects(
        self,
        env: Any,
        effects: Mapping[str, Mapping[str, Any]],
        *,
        exhaustive: bool = False,
    ) -> tuple[bool, dict[str, bool], dict[str, bool], str]:
        planner = self._planner_cls(env)
        results: dict[str, bool] = {}
        expected_results: dict[str, bool] = {}
        failures: list[str] = []
        for predicate, items in effects.items():
            expected = bool(items.get("value", True))
            expected_results[predicate] = expected
            fn: Callable[..., bool] | None = getattr(planner, predicate, None)
            if fn is None:
                failures.append(f"missing predicate {predicate}")
                if not exhaustive:
                    return False, results, expected_results, failures[-1]
                continue
            params = dict(items.get("params", {}))
            try:
                observed = bool(fn(env, **params))
            except TypeError as exc:
                failures.append(f"{predicate} arg error: {exc}")
                if not exhaustive:
                    return False, results, expected_results, failures[-1]
                continue
            results[predicate] = observed
            if observed != expected:
                failures.append(
                    f"{predicate} expected {expected} got {observed}"
                )
                if not exhaustive:
                    return False, results, expected_results, failures[-1]
        if failures:
            return False, results, expected_results, "; ".join(failures)
        return True, results, expected_results, "all predicates matched"

    @staticmethod
    def _save_sim_state(env: Any) -> tuple[Any | None, Any | None]:
        """Snapshot Bullet state because some predicates call ``world.step()``."""
        bullet = getattr(getattr(env, "world", None), "p", None)
        save_state = getattr(bullet, "saveState", None)
        if not callable(save_state):
            return None, None
        return bullet, save_state()

    @staticmethod
    def _restore_sim_state(bullet: Any | None, state_id: Any | None) -> None:
        if bullet is None or state_id is None:
            return
        bullet.restoreState(stateId=state_id)
        remove_state = getattr(bullet, "removeState", None)
        if callable(remove_state):
            remove_state(stateUniqueId=state_id)

    def is_skill_complete(
        self,
        env: Any,
        skill_index: int,
        *,
        exhaustive: bool = False,
        preserve_sim_state: bool = False,
    ) -> OracleDecision:
        entry, reason = self.skill_entry(env, skill_index)
        if entry is None:
            return OracleDecision(
                complete=False,
                supported=False,
                skill_name=None,
                skill_index=skill_index,
                reason=reason,
            )
        skill_name = str(entry["skill_name"])
        if skill_name not in SUPPORTED_SKILLS:
            return OracleDecision(
                complete=False,
                supported=False,
                skill_name=skill_name,
                skill_index=skill_index,
                reason=f"unsupported skill_name={skill_name}",
            )
        params = entry.get("params", {}) or {}
        if not isinstance(params, Mapping):
            return OracleDecision(
                complete=False,
                supported=False,
                skill_name=skill_name,
                skill_index=skill_index,
                reason="skill params must be a mapping",
            )
        effects, reason = self.effects_for_skill(env, skill_name, params)
        if effects is None:
            return OracleDecision(
                complete=False,
                supported=False,
                skill_name=skill_name,
                skill_index=skill_index,
                reason=reason,
            )
        bullet = state_id = None
        if preserve_sim_state:
            bullet, state_id = self._save_sim_state(env)
        try:
            complete, results, expected, detail = self.evaluate_effects(
                env, effects, exhaustive=exhaustive
            )
        finally:
            self._restore_sim_state(bullet, state_id)
        return OracleDecision(
            complete=complete,
            supported=True,
            skill_name=skill_name,
            skill_index=skill_index,
            predicate_results=results,
            predicate_expected=expected,
            reason=detail,
        )
