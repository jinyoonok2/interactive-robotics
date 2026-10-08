"""Validate the maintained hierarchical Part2Action configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "architecture_update" / "hierarchical_world_geometry_30epoch_no_residual.yaml"


def validate(path: Path, check_files: bool = False) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    errors = []
    data, model, losses = cfg.get("data", {}), cfg.get("model", {}), cfg.get("losses", {})
    frozen = set(cfg.get("train", {}).get("freeze_prefixes", []))
    if not model.get("use_hierarchy"):
        errors.append("model.use_hierarchy must be true")
    if not data.get("use_hierarchy"):
        errors.append("data.use_hierarchy must be true")
    if model.get("max_skill_slots", 4) != data.get("max_skill_slots", 4):
        errors.append("model/data max_skill_slots must match")
    if model.get("use_pcd") and not data.get("use_pcd"):
        errors.append("3D policy requires deployable full-scene PCD")
    if model.get("use_skill_gate") and not model.get("use_part_gate"):
        errors.append("skill-conditioned gate requires model.use_part_gate")
    rollout = cfg.get("rollout", {}) or {}
    termination_mode = str(rollout.get("termination_mode", "legacy"))
    if termination_mode not in {"legacy", "oracle_partgym"}:
        errors.append("rollout.termination_mode must be legacy or oracle_partgym")
    # Oracle termination is an eval-only diagnostic and must not be the training default.
    if termination_mode == "oracle_partgym" and float(losses.get("termination_weight", 0)) > 0:
        errors.append(
            "oracle_partgym termination cannot be enabled in a training config "
            "that supervises termination_weight; keep rollout.termination_mode=legacy"
        )
    if model.get("use_contact_action_token") and not (
        model.get("use_pcd") and model.get("use_contact_xyz") and data.get("use_contact_xyz")
    ):
        errors.append("contact action token requires model PCD/contact and contact supervision")
    if model.get("use_point_part_head") and not (
        model.get("use_pcd")
        and data.get("use_pcd")
        and data.get("use_point_part_labels")
        and data.get("use_hierarchy")
    ):
        errors.append(
            "point-part head requires model/data PCD, hierarchy, and point-part labels"
        )
    if float(losses.get("point_part_weight", 0)) > 0 and not model.get(
        "use_point_part_head"
    ):
        errors.append("point_part_weight requires model.use_point_part_head")
    if float(losses.get("contact_attn_weight", 0)) > 0:
        if not model.get("use_contact_action_token"):
            errors.append("contact_attn_weight requires model.use_contact_action_token")
        if not data.get("use_point_part_labels"):
            errors.append("contact_attn_weight requires data.use_point_part_labels")
        if not data.get("use_contact_xyz"):
            errors.append("contact_attn_weight requires data.use_contact_xyz")
        if model.get("use_point_part_head"):
            errors.append(
                "contact-attention part supervision should not enable use_point_part_head"
            )
    contact_label_mode = str(data.get("contact_label_mode", "legacy"))
    if contact_label_mode not in {"legacy", "skill_specific"}:
        errors.append("data.contact_label_mode must be legacy or skill_specific")
    if contact_label_mode == "skill_specific" and not data.get("use_hierarchy"):
        errors.append("skill-specific contact labels require data.use_hierarchy")
    if model.get("split_action_heads") and not model.get("use_contact_action_token"):
        errors.append("split action heads require contact action token")
    if float(model.get("gripper_open_width", 0.04)) <= float(
        model.get("gripper_closed_width", 0.0)
    ):
        errors.append("gripper_open_width must exceed gripper_closed_width")
    if data.get("use_part_pcd"):
        errors.append("target-part PCD must not be a policy input")
    hierarchy_loss_modules = {
        "skill_embedding_weight": "skill_planner",
        "slot_validity_weight": "skill_planner",
        "phase_weight": "phase_selector",
        "termination_weight": "termination_head",
    }
    for key, module in hierarchy_loss_modules.items():
        if float(losses.get(key, 0)) <= 0 and module not in frozen:
            errors.append(f"{key} must be positive")
    if model.get("split_action_heads"):
        for key in ("arm_action_weight", "gripper_weight"):
            if float(losses.get(key, 0)) <= 0:
                errors.append(f"{key} must be positive for split action heads")
        if float(data.get("gripper_transition_weight", 0)) < 1:
            errors.append("gripper_transition_weight must be >= 1")
    if model.get("use_contact_residual"):
        if model.get("split_action_heads"):
            errors.append("contact residual must preserve the monolithic baseline action head")
        if float(model.get("contact_residual_max_translation", 0)) <= 0:
            errors.append("contact_residual_max_translation must be positive")
    if model.get("use_contact_relative_actions"):
        if model.get("use_contact_residual"):
            errors.append("contact-relative actions replace the residual; disable use_contact_residual")
        if model.get("use_action_deltas"):
            errors.append("contact-relative actions and use_action_deltas are mutually exclusive")
        if not model.get("use_contact_action_token"):
            errors.append("contact-relative actions require use_contact_action_token")
        if not data.get("use_tcp_pose"):
            errors.append("contact-relative actions require data.use_tcp_pose")
        if not data.get("use_contact_xyz"):
            errors.append("contact-relative actions require data.use_contact_xyz")
        if model.get("action_head_type", "mlp") != "mlp":
            errors.append("contact-relative actions currently require action_head_type=mlp")
        if model.get("split_action_heads"):
            errors.append("contact-relative actions currently require split_action_heads=false")
        if float(model.get("contact_relative_max_translation", 0)) <= 0:
            errors.append("contact_relative_max_translation must be positive")
    if model.get("use_unified_contact_conditioning"):
        if not model.get("use_contact_action_token") or not data.get("use_tcp_pose"):
            errors.append("unified conditioning requires contact action token and data.use_tcp_pose")
        if model.get("action_head_type", "mlp") != "mlp" or model.get("split_action_heads"):
            errors.append("unified conditioning requires monolithic MLP action head")
        if any(model.get(flag) for flag in ["use_near_contact_actions", "use_contact_relative_actions", "use_contact_residual", "use_action_deltas"]):
            errors.append("unified conditioning is incompatible with contact switching/residuals/deltas")
        if float(model.get("contact_geometry_scale_m", 0.1)) <= 0:
            errors.append("contact_geometry_scale_m must be positive")
    n_obs_steps = int(data.get("n_obs_steps", 1))
    temporal = str(model.get("temporal_encoder_type", "none")).lower()
    if n_obs_steps > 1 and temporal in {"none", "identity", ""}:
        errors.append("n_obs_steps>1 requires temporal_encoder_type=transformer")
    if temporal in {"transformer", "patch_transformer"} and n_obs_steps < 2:
        errors.append("temporal_encoder_type=transformer requires n_obs_steps>=2")
    if model.get("use_near_contact_actions"):
        if model.get("use_contact_relative_actions") or model.get("use_contact_residual"):
            errors.append("near-contact actions replace the full-path offset and the residual")
        if model.get("use_action_deltas"):
            errors.append("near-contact actions and use_action_deltas are mutually exclusive")
        if not model.get("use_contact_action_token"):
            errors.append("near-contact actions require use_contact_action_token")
        if not data.get("use_tcp_pose"):
            errors.append("near-contact actions require data.use_tcp_pose")
        if not data.get("use_contact_xyz"):
            errors.append("near-contact actions require data.use_contact_xyz")
        if model.get("action_head_type", "mlp") != "mlp":
            errors.append("near-contact actions currently require action_head_type=mlp")
        if model.get("split_action_heads"):
            errors.append("near-contact actions currently require split_action_heads=false")
        distance = float(model.get("near_contact_distance", 0))
        offset_cap = float(model.get("near_contact_max_translation", 0))
        if distance <= 0 or distance > 0.05:
            errors.append("near_contact_distance must be within (0, 0.05] meters")
        if offset_cap <= 0 or offset_cap > 0.05:
            errors.append("near_contact_max_translation must be within (0, 0.05] meters")
    if model.get("use_action_deltas"):
        if not data.get("use_tcp_pose"):
            errors.append("use_action_deltas requires data.use_tcp_pose")
        if model.get("action_head_type", "mlp") != "mlp":
            errors.append("use_action_deltas currently requires action_head_type=mlp")
        if model.get("split_action_heads"):
            errors.append("use_action_deltas currently requires split_action_heads=false")
    if check_files:
        for item in data.get("train_hdf5", []):
            candidate = Path(item) if Path(item).is_absolute() else ROOT / item
            if not candidate.resolve().exists():
                errors.append(f"missing dataset: {candidate.resolve()}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, default=[DEFAULT])
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    failed = False
    for path in args.configs:
        errors = validate(path.expanduser().resolve(), args.check_files)
        print(f"[{'ERROR' if errors else 'OK'}] {path}")
        for error in errors:
            print(f"  {error}")
        failed |= bool(errors)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
