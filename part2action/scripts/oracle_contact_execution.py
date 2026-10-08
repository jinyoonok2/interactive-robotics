"""Privileged PartGym contact-execution diagnostic.

This uses PartGym's geometry-based touch servo for the first touch/grasp skill
in each episode. It does not evaluate or provide inputs to the learned policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from _common import ensure_dir, set_seed
from diagnostics.partgym_reach import measure_tcp_to_target_part
from diagnostics.partgym_skill_oracle import PartGymSkillOracle
from rollout_partgym import (
    DEFAULT_PARTINSTRUCT_ROOT,
    _load_episode_keys,
    _load_partgym_env,
    configure_partgym_agentview_pcd,
)


def _lightweight_touch_planner(env: Any):
    """Construct BulletPlanner without the unavailable VGN grasp checkpoint."""
    from PartInstruct.PartGym.env.backend.planner.bullet_planner import (
        BulletPlanner,
        OracleChecker,
    )

    planner = BulletPlanner.__new__(BulletPlanner)
    planner.generation_mode = False
    planner.env = env
    planner.keep_all = True
    planner.checker = OracleChecker(env)
    planner.world = env.world
    planner.robot = env.robot
    planner.parser = copy.deepcopy(env.parser)
    planner.parser.world = planner.world
    return planner


def _first_contact_skill(env: Any) -> tuple[int, dict[str, Any]] | None:
    for index, raw in enumerate(getattr(env, "chain_params", [])):
        entry = dict(raw)
        if str(entry.get("skill_name")) in {"touch_obj", "grasp_obj"}:
            return index, entry
    return None


def run_trial(
    *,
    env_cls,
    partgym_cfg: Any,
    obj_class: str,
    task_type: str,
    split: str,
    trial_index: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    env = env_cls(
        config=partgym_cfg,
        gui=False,
        record=False,
        evaluation=True,
        skill_mode=False,
        obj_class=obj_class,
        split=split,
        task_type=task_type,
        track_samples=False,
    )
    result: dict[str, Any] = {
        "obj_class": obj_class,
        "task_type": task_type,
        "split": split,
        "trial_index": trial_index,
        "seed": seed,
        "supported": False,
        "controller": "partgym_geometry_touch_servo",
    }
    try:
        obs = env.reset()
        result["instruction"] = str(env.task_instruction)
        resolved = _first_contact_skill(env)
        if resolved is None:
            result["reason"] = "episode has no touch/grasp skill"
            return result
        skill_index, entry = resolved
        skill_name = str(entry["skill_name"])
        params = entry.get("params", {}) or {}
        part_name = str(
            params.get("part_touch", params.get("part_grasp", ""))
        )
        region = str(params.get("region_on_part", ""))
        result.update(
            supported=True,
            skill_index=skill_index,
            skill_name=skill_name,
            part_name=part_name,
            region_on_part=region,
            before=measure_tcp_to_target_part(
                env, obs.get("tcp_pose", []), skill_index
            ),
        )

        planner = _lightweight_touch_planner(env)
        planner_return = planner.touch_obj(
            obj=str(getattr(env.parser, "obj_name", obj_class)),
            part_touch=part_name,
            region_on_part=region,
        )
        final_pose = env.robot.get_tcp_pose()
        final_tcp_pose = [
            *final_pose.rotation.as_quat().astype(float).tolist(),
            *final_pose.translation.astype(float).tolist(),
        ]
        decision = PartGymSkillOracle().is_skill_complete(
            env,
            skill_index,
            exhaustive=True,
            preserve_sim_state=True,
        )
        result.update(
            planner_return=bool(planner_return),
            predicate_complete=bool(decision.complete),
            predicate_results=dict(decision.predicate_results),
            predicate_expected=dict(decision.predicate_expected),
            after=measure_tcp_to_target_part(
                env, final_tcp_pose, skill_index
            ),
            reason=decision.reason,
        )
        return result
    except Exception as exc:
        result["reason"] = f"{exc.__class__.__name__}: {exc}"
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partinstruct-root", type=Path, default=DEFAULT_PARTINSTRUCT_ROOT)
    parser.add_argument("--partgym-config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="test1")
    parser.add_argument("--obj-classes", nargs="+", default=["bottle", "mug", "pliers", "scissors"])
    parser.add_argument("--task-types", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--trials-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    partinstruct_root = args.partinstruct_root.expanduser().resolve()
    config_path = (
        args.partgym_config.expanduser().resolve()
        if args.partgym_config
        else partinstruct_root / "PartInstruct" / "PartGym" / "config" / "config_oracle.yaml"
    )
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else partinstruct_root / "data"
    )
    partgym_cfg = OmegaConf.load(config_path)
    partgym_cfg.data_root = str(data_root)
    full_cfg = OmegaConf.load(
        partinstruct_root / "PartInstruct" / "PartGym" / "config" / "config.yaml"
    )
    configure_partgym_agentview_pcd(partgym_cfg, full_cfg)

    meta_path = data_root / partgym_cfg.meta_path
    env_cls = _load_partgym_env(partinstruct_root)
    episodes = _load_episode_keys(
        meta_path,
        args.split,
        args.num_episodes,
        obj_classes=args.obj_classes,
        task_types=args.task_types,
    )
    results = []
    combination_index = 0
    for obj_class, task_type in episodes:
        for trial_index in range(args.trials_per_task):
            seed = int(args.seed + combination_index * 100 + trial_index)
            result = run_trial(
                env_cls=env_cls,
                partgym_cfg=partgym_cfg,
                obj_class=obj_class,
                task_type=task_type,
                split=args.split,
                trial_index=trial_index,
                seed=seed,
            )
            results.append(result)
            print(
                f"[oracle-contact] {obj_class}/{task_type} trial={trial_index} "
                f"skill={result.get('skill_name')} "
                f"complete={result.get('predicate_complete')} "
                f"reason={result.get('reason', '')}"
            )
        combination_index += 1

    by_skill: dict[str, Counter] = {}
    for result in results:
        skill = str(result.get("skill_name", "unsupported"))
        counts = by_skill.setdefault(skill, Counter())
        counts["trials"] += 1
        counts["supported"] += int(bool(result.get("supported")))
        counts["planner_return"] += int(bool(result.get("planner_return")))
        counts["predicate_complete"] += int(
            bool(result.get("predicate_complete"))
        )
    summary = {
        skill: {
            **dict(counts),
            "predicate_complete_rate": (
                counts["predicate_complete"] / counts["trials"]
                if counts["trials"]
                else 0.0
            ),
        }
        for skill, counts in sorted(by_skill.items())
    }
    report = {
        "diagnostic": "privileged geometry contact execution",
        "config": str(config_path),
        "num_trials": len(results),
        "by_skill": summary,
        "results": results,
    }
    out = args.out.expanduser().resolve()
    ensure_dir(out.parent)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"num_trials": len(results), "by_skill": summary}, indent=2))
    print(f"[oracle-contact] wrote {out}")


if __name__ == "__main__":
    main()
