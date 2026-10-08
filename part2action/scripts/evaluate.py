"""Evaluate part2action tracks with offline metrics + optional PartGym rollouts.

Offline metrics (always run):
  - Part heatmap IoU (predicted thresholded vs GT part_mask).
  - Contact-point error in normalized image coordinates (only for B-style models).
  - Approach-direction cosine similarity (only for B-style models).
  - Action-chunk smooth-L1 (only for B-style models).

PartGym rollouts (optional, requires a separate upstream PartInstruct env):
  - Loaded only if `--use_partgym` is passed AND the package is importable.
  - We avoid hard-importing PartGym so this script runs in `part2action`
    env even without it.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from _common import (
    ROOT,
    ensure_dir,
    load_yaml,
    resolve_paths,
    save_json,
    select_device,
    set_seed,
)
from data.geometry import (
    absolute_actions_to_contact_relative,
    absolute_actions_to_tcp_deltas,
    tcp_deltas_to_absolute_actions,
)
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel

DEFAULT_CONFIG = ROOT / "configs" / "architecture_update" / "hierarchical_world_geometry_30epoch_no_residual.yaml"


def build_eval_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    eval_paths = data_cfg.get("val_hdf5") or data_cfg["train_hdf5"]
    ds = PartInstructDataset(
        hdf5_paths=eval_paths,
        action_chunk=data_cfg.get("action_chunk", 8),
        sample_stride=max(1, data_cfg.get("sample_stride", 1) * 2),
        max_demos_per_file=data_cfg.get("max_demos_per_file"),
        n_obs_steps=data_cfg.get("n_obs_steps", 1),
        use_pcd=data_cfg.get("use_pcd", False),
        use_part_pcd=data_cfg.get("use_part_pcd", False),
        use_tcp_pose=data_cfg.get("use_tcp_pose", False),
        use_contact_xyz=data_cfg.get("use_contact_xyz", False),
        contact_label_mode=data_cfg.get("contact_label_mode", "legacy"),
        use_point_part_labels=data_cfg.get("use_point_part_labels", False),
        use_hierarchy=data_cfg.get("use_hierarchy", False),
        max_skill_slots=data_cfg.get("max_skill_slots", 4),
        gripper_transition_weight=float(data_cfg.get("gripper_transition_weight", 4.0)),
    )
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("eval_batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("eval_num_workers", 4)),
        collate_fn=collate_part2action,
    )


def iou_score(pred_logits: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    if pred_logits.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(gt_mask.unsqueeze(1), size=pred_logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * gt_mask).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt_mask.sum(dim=(-1, -2)) - inter
    iou = (inter / union.clamp(min=1e-6)).mean()
    return float(iou.item())


def gate_prf(
    pred_logits: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    side = int(pred_logits.shape[-1] ** 0.5)
    gate_map = pred_logits.reshape(-1, side, side)
    if gate_map.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(1), size=gate_map.shape[-2:], mode="area"
        ).squeeze(1)
    pred = (torch.sigmoid(gate_map) > threshold).float()
    target = (gt_mask > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = float((tp / (tp + fp).clamp(min=1e-6)).item())
    recall = float((tp / (tp + fn).clamp(min=1e-6)).item())
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": f1,
    }


def offline_eval(
    model: Part2ActionModel, dl: DataLoader, device: torch.device,
    hierarchy_mode: str = "predicted_hierarchy",
    gate_mode: str = "predicted",
) -> Dict[str, float]:
    model.eval()
    n = 0
    sums: Dict[str, float] = {}
    contact_skill_sums: Dict[str, float] = {}
    contact_skill_counts: Dict[str, int] = {}
    grip_tp = grip_fp = grip_fn = 0.0
    point_tp = point_fp = point_fn = 0.0
    gate_prf_sums: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"offline-eval[{gate_mode}]"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            extra = {k: batch[k] for k in ("agentview_pcd", "tcp_pose", "gripper_state") if k in batch}
            extra["gate_mode"] = gate_mode
            if gate_mode == "oracle":
                extra["gt_part_mask"] = batch["part_mask"]
            if model.use_hierarchy:
                instructions = batch["task_instruction"]
                if hierarchy_mode == "direct_task":
                    extra["hierarchy_mode"] = "direct"
                else:
                    extra.update(
                        skill_plan=batch["skill_plan"],
                        skill_valid_mask=batch["skill_valid_mask"],
                        phase_index=batch["phase_index"],
                        hierarchy_mode="oracle" if hierarchy_mode == "oracle_skill" else "predicted",
                    )
                    if hierarchy_mode == "oracle_skill":
                        extra["oracle_skill_instructions"] = batch["instruction"]
            else:
                instructions = batch["instruction"]
            out = model(batch["rgb"], instructions, **extra)

            if "heatmap_logits" in out:
                sums["iou"] = sums.get("iou", 0.0) + iou_score(out["heatmap_logits"], batch["part_mask"])
            if "gate_logits" in out:
                side = int(out["gate_logits"].shape[-1] ** 0.5)
                gate_map = out["gate_logits"].reshape(-1, side, side)
                sums["gate_iou"] = sums.get("gate_iou", 0.0) + iou_score(
                    gate_map, batch["part_mask"]
                )
                for key, value in gate_prf(out["gate_logits"], batch["part_mask"]).items():
                    gate_prf_sums[key] = gate_prf_sums.get(key, 0.0) + value
            if "point_part_logits" in out:
                point_valid = batch["point_part_valid"].bool()[:, None]
                point_target = batch["point_part_labels"].bool()
                point_pred = out["point_part_logits"] >= 0
                point_tp += float(
                    (point_pred & point_target & point_valid).sum().item()
                )
                point_fp += float(
                    (point_pred & ~point_target & point_valid).sum().item()
                )
                point_fn += float(
                    (~point_pred & point_target & point_valid).sum().item()
                )
            if "contact_xy" in out:
                sums["contact_l1"] = sums.get("contact_l1", 0.0) + float(F.l1_loss(out["contact_xy"], batch["contact_xy"]).item())
            if "approach_dir" in out:
                cos = (out["approach_dir"] * batch["approach_dir"]).sum(dim=-1).clamp(-1.0, 1.0).mean()
                sums["approach_cos"] = sums.get("approach_cos", 0.0) + float(cos.item())
            if "contact_xyz" in out:
                distance = torch.linalg.vector_norm(
                    out["contact_xyz"] - batch["contact_xyz"], dim=-1
                )
                valid = batch["contact_xyz_valid"]
                contact_distance = (distance * valid).sum() / valid.sum().clamp(min=1)
                sums["contact_xyz_distance"] = sums.get("contact_xyz_distance", 0.0) + float(
                    contact_distance.item()
                )
                for index, metadata in enumerate(batch.get("meta", [])):
                    if not bool(valid[index].item()):
                        continue
                    skill_kind = str(metadata.get("skill_kind", "unclassified"))
                    contact_skill_sums[skill_kind] = (
                        contact_skill_sums.get(skill_kind, 0.0)
                        + float(distance[index].item())
                    )
                    contact_skill_counts[skill_kind] = (
                        contact_skill_counts.get(skill_kind, 0) + 1
                    )
            if (
                "contact_point_logits" in out
                and "point_part_labels" in batch
                and "point_part_valid" in batch
            ):
                pred = torch.softmax(out["contact_point_logits"].float(), dim=-1)
                part_labels = batch["point_part_labels"].to(dtype=pred.dtype)
                sample_valid = batch["point_part_valid"].to(dtype=pred.dtype)
                if "contact_xyz_valid" in batch:
                    sample_valid = sample_valid * batch["contact_xyz_valid"].to(
                        dtype=pred.dtype
                    )
                sample_valid = sample_valid * (part_labels.sum(dim=-1) > 0).to(
                    dtype=pred.dtype
                )
                attn_on_part = (pred * part_labels).sum(dim=-1)
                denom = sample_valid.sum().clamp(min=1.0)
                sums["contact_attn_on_part"] = sums.get(
                    "contact_attn_on_part", 0.0
                ) + float((attn_on_part * sample_valid).sum().item() / float(denom))
            if "arm_action_chunk" in out:
                sums["arm_l1"] = sums.get("arm_l1", 0.0) + float(
                    F.smooth_l1_loss(
                        out["arm_action_chunk"], batch["action_chunk"][..., :6]
                    ).item()
                )
            elif "action_chunk" in out:
                pred_action = out["action_chunk"]
                target_action = batch["action_chunk"]
                if model.use_action_deltas:
                    target_delta = absolute_actions_to_tcp_deltas(
                        batch["action_chunk"], batch["tcp_pose"]
                    )
                    sums["action_delta_l1"] = sums.get("action_delta_l1", 0.0) + float(
                        F.smooth_l1_loss(pred_action, target_delta).item()
                    )
                    pred_action = tcp_deltas_to_absolute_actions(
                        pred_action, batch["tcp_pose"]
                    )
                sums["action_l1"] = sums.get("action_l1", 0.0) + float(
                    F.smooth_l1_loss(pred_action, target_action).item()
                )
                if "contact_xyz_offset" in out and "contact_xyz" in batch:
                    target_relative = absolute_actions_to_contact_relative(
                        batch["action_chunk"], batch["contact_xyz"]
                    )
                    sums["action_contact_offset_l1"] = sums.get(
                        "action_contact_offset_l1", 0.0
                    ) + float(
                        F.l1_loss(
                            out["contact_xyz_offset"], target_relative[..., :3]
                        ).item()
                    )
            if "gripper_logits" in out:
                pred_closed = out["gripper_logits"] >= 0
                target_closed = batch["gripper_closed"].bool()
                grip_tp += float((pred_closed & target_closed).sum())
                grip_fp += float((pred_closed & ~target_closed).sum())
                grip_fn += float((~pred_closed & target_closed).sum())
                target_has_close = target_closed.any(dim=-1)
                pred_index = pred_closed.float().argmax(dim=-1)
                pred_index = torch.where(
                    pred_closed.any(dim=-1), pred_index,
                    torch.full_like(pred_index, pred_closed.shape[-1]),
                )
                target_index = target_closed.float().argmax(dim=-1)
                if target_has_close.any():
                    onset_mae = (
                        pred_index[target_has_close] - target_index[target_has_close]
                    ).abs().float().mean()
                    sums["gripper_onset_mae"] = sums.get("gripper_onset_mae", 0.0) + float(
                        onset_mae.item()
                    )
            if "phase_logits" in out:
                sums["phase_accuracy"] = sums.get("phase_accuracy", 0.0) + float(
                    (out["phase_logits"].argmax(-1) == batch["phase_index"]).float().mean()
                )
                sums["termination_accuracy"] = sums.get("termination_accuracy", 0.0) + float(
                    ((out["termination_logits"] >= 0) == batch["phase_termination"].bool()).float().mean()
                )
                valid = batch["skill_valid_mask"]
                similarity = F.cosine_similarity(out["plan_slots"], out["skill_target_slots"], dim=-1)
                sums["plan_slot_similarity"] = sums.get("plan_slot_similarity", 0.0) + float(
                    (similarity * valid).sum() / valid.sum().clamp(min=1)
                )
            n += 1
    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["contact_xyz_distance_by_skill"] = {
        skill_kind: contact_skill_sums[skill_kind] / count
        for skill_kind, count in sorted(contact_skill_counts.items())
    }
    metrics.update({k: v / max(1, n) for k, v in gate_prf_sums.items()})
    point_precision = point_tp / max(1.0, point_tp + point_fp)
    point_recall = point_tp / max(1.0, point_tp + point_fn)
    if point_tp + point_fp + point_fn > 0:
        metrics.update(
            point_part_precision=point_precision,
            point_part_recall=point_recall,
            point_part_f1=(
                2
                * point_precision
                * point_recall
                / max(1e-8, point_precision + point_recall)
            ),
        )
    metrics["gate_mode"] = gate_mode
    precision = grip_tp / max(1.0, grip_tp + grip_fp)
    recall = grip_tp / max(1.0, grip_tp + grip_fn)
    if grip_tp + grip_fp + grip_fn > 0:
        metrics.update(
            gripper_precision=precision,
            gripper_recall=recall,
            gripper_f1=2 * precision * recall / max(1e-8, precision + recall),
        )
    return metrics


def maybe_partgym_rollout(model: Part2ActionModel, device: torch.device, n_episodes: int, splits: List[str]) -> Optional[Dict[str, Any]]:
    """Run PartGym rollouts if the upstream package is importable.

    This is intentionally lazy: missing imports return None so this script
    works in the lightweight `part2action` env. Install upstream PartInstruct
    separately to enable rollouts.
    """
    try:
        importlib.import_module("PartInstruct.PartGym.env.bullet_env")
    except Exception as e:
        print(f"[partgym] not available ({e.__class__.__name__}: {e}); skipping rollouts.")
        return None

    print("[partgym] PartGym available, but the rollout adapter is intentionally")
    print("[partgym] left as a TODO scaffold: hooking model -> action requires")
    print("[partgym] aligning the EE control space with PartInstruct's runner.")
    print("[partgym] See docs/SETUP.md for the integration recipe.")
    return {"status": "scaffold", "n_episodes": int(n_episodes), "splits": list(splits)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--hierarchy-modes", nargs="+",
        choices=["direct_task", "oracle_skill", "predicted_hierarchy"],
        default=["oracle_skill", "predicted_hierarchy"],
    )
    p.add_argument(
        "--mask-modes", nargs="+",
        choices=["predicted", "oracle"],
        default=["predicted"],
        help=(
            "Grounding diagnostic: predicted uses the learned gate; "
            "oracle forces GT part-mask gating while still reporting predicted gate_iou."
        ),
    )
    p.add_argument("--use_partgym", action="store_true")
    p.add_argument("--rollout_episodes", type=int, default=10)
    p.add_argument("--rollout_splits", nargs="+", default=["test1"], help="PartInstruct test splits.")
    args = p.parse_args()

    cfg = resolve_paths(load_yaml(args.config), ROOT)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = select_device(cfg["train"].get("device", "cuda"))
    if device.type == "cpu":
        cfg["model"]["text_device"] = "cpu"

    model = Part2ActionModel(
        heads=cfg["heads"],
        img_size=int(cfg["model"].get("img_size", 252)),
        out_size=int(cfg["model"].get("out_size", 96)),
        action_chunk=int(cfg["data"].get("action_chunk", 8)),
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_fusion_layers=int(cfg["model"].get("num_fusion_layers", 2)),
        text_device=cfg["model"].get("text_device", "cpu"),
        action_head_type=cfg["model"].get("action_head_type", "mlp"),
        diffusion_steps=int(cfg["model"].get("diffusion_steps", 50)),
        temporal_encoder_type=cfg["model"].get("temporal_encoder_type", "none"),
        n_obs_steps=int(cfg["data"].get("n_obs_steps", 1)),
        temporal_layers=int(cfg["model"].get("temporal_layers", 1)),
        temporal_heads=int(cfg["model"].get("temporal_heads", 4)),
        use_part_gate=bool(cfg["model"].get("use_part_gate", False)),
        use_pcd=bool(cfg["model"].get("use_pcd", False)),
        use_contact_xyz=bool(cfg["model"].get("use_contact_xyz", False)),
        use_hierarchy=bool(cfg["model"].get("use_hierarchy", False)),
        max_skill_slots=int(cfg["model"].get("max_skill_slots", 4)),
        hierarchy_layers=int(cfg["model"].get("hierarchy_layers", 2)),
        phase_selector_use_3d=bool(cfg["model"].get("phase_selector_use_3d", True)),
        use_skill_gate=bool(cfg["model"].get("use_skill_gate", False)),
        use_contact_action_token=bool(cfg["model"].get("use_contact_action_token", False)),
        split_action_heads=bool(cfg["model"].get("split_action_heads", False)),
        gripper_open_width=float(cfg["model"].get("gripper_open_width", 0.04)),
        gripper_closed_width=float(cfg["model"].get("gripper_closed_width", 0.0)),
        use_contact_residual=bool(cfg["model"].get("use_contact_residual", False)),
        contact_residual_max_translation=float(
            cfg["model"].get("contact_residual_max_translation", 0.03)
        ),
        use_contact_relative_actions=bool(
            cfg["model"].get("use_contact_relative_actions", False)
        ),
        contact_relative_max_translation=float(
            cfg["model"].get("contact_relative_max_translation", 0.40)
        ),
        use_unified_contact_conditioning=bool(cfg["model"].get("use_unified_contact_conditioning", False)),
        contact_conditioning_enabled=bool(cfg["model"].get("contact_conditioning_enabled", True)),
        contact_geometry_scale_m=float(cfg["model"].get("contact_geometry_scale_m", 0.1)),
        use_near_contact_actions=bool(
            cfg["model"].get("use_near_contact_actions", False)
        ),
        near_contact_distance=float(cfg["model"].get("near_contact_distance", 0.03)),
        near_contact_max_translation=float(
            cfg["model"].get("near_contact_max_translation", 0.03)
        ),
        use_action_deltas=bool(cfg["model"].get("use_action_deltas", False)),
        use_point_part_head=bool(
            cfg["model"].get("use_point_part_head", False)
        ),
        use_gripper_state=bool(cfg["model"].get("use_gripper_state", False)),
    ).to(device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if state.get("trainable_only", False):
        unexpected_real = list(unexpected)
        if unexpected_real:
            print(f"[eval] unexpected keys (likely fine): {unexpected_real[:3]}{'...' if len(unexpected_real) > 3 else ''}")
        print(f"[eval] loaded trainable params; frozen backbones reloaded from cache.")

    dl = build_eval_loader(cfg)
    results: Dict[str, Any] = {"config": args.config, "ckpt": args.ckpt}
    results["offline_modes"] = {}
    for hierarchy_mode in args.hierarchy_modes:
        results["offline_modes"][hierarchy_mode] = {
            mask_mode: offline_eval(
                model, dl, device,
                hierarchy_mode=hierarchy_mode,
                gate_mode=mask_mode,
            )
            for mask_mode in args.mask_modes
        }
    print(f"[eval] offline modes: {results['offline_modes']}")

    if args.use_partgym:
        results["partgym"] = maybe_partgym_rollout(
            model, device, n_episodes=args.rollout_episodes, splits=args.rollout_splits
        )

    out_dir = ensure_dir(Path(cfg["output_dir"]))
    save_json(out_dir / "eval.json", results)
    print(f"[eval] wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
