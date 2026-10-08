"""Unified Part2Action and hierarchical policy trainer."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from _common import ROOT, ensure_dir, load_yaml, resolve_paths, save_json, select_device, set_seed
from data.geometry import (
    absolute_actions_to_contact_relative,
    absolute_actions_to_tcp_deltas,
    camera_to_world_torch,
)
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel

DEFAULT_CONFIG = ROOT / "configs" / "architecture_update" / "hierarchical_world_geometry_30epoch_no_residual.yaml"


def build_dataset(data: dict) -> PartInstructDataset:
    return PartInstructDataset(
        hdf5_paths=data["train_hdf5"], action_chunk=data.get("action_chunk", 8),
        sample_stride=data.get("sample_stride", 1),
        max_demos_per_file=data.get("max_demos_per_file"),
        n_obs_steps=data.get("n_obs_steps", 1), use_pcd=data.get("use_pcd", False),
        use_part_pcd=data.get("use_part_pcd", False),
        use_tcp_pose=data.get("use_tcp_pose", False),
        use_contact_xyz=data.get("use_contact_xyz", False),
        contact_label_mode=data.get("contact_label_mode", "legacy"),
        use_point_part_labels=data.get("use_point_part_labels", False),
        use_hierarchy=data.get("use_hierarchy", False),
        max_skill_slots=data.get("max_skill_slots", 4),
        gripper_transition_weight=float(data.get("gripper_transition_weight", 4.0)),
    )


def build_dataloader(ds, data):
    return DataLoader(
        ds, batch_size=int(data.get("batch_size", 8)), shuffle=True,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", True)),
        collate_fn=collate_part2action, drop_last=True,
    )


def build_model(cfg: dict) -> Part2ActionModel:
    m, d = cfg["model"], cfg["data"]
    return Part2ActionModel(
        heads=cfg["heads"], img_size=int(m.get("img_size", 252)),
        out_size=int(m.get("out_size", 96)), action_chunk=int(d.get("action_chunk", 8)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_fusion_layers=int(m.get("num_fusion_layers", 2)),
        text_device=m.get("text_device", "cpu"),
        action_head_type=m.get("action_head_type", "mlp"),
        diffusion_steps=int(m.get("diffusion_steps", 50)),
        temporal_encoder_type=m.get("temporal_encoder_type", "none"),
        n_obs_steps=int(d.get("n_obs_steps", 1)),
        temporal_layers=int(m.get("temporal_layers", 1)),
        temporal_heads=int(m.get("temporal_heads", 4)),
        use_part_gate=bool(m.get("use_part_gate", False)),
        use_pcd=bool(m.get("use_pcd", False)),
        use_contact_xyz=bool(m.get("use_contact_xyz", False)),
        use_hierarchy=bool(m.get("use_hierarchy", False)),
        max_skill_slots=int(m.get("max_skill_slots", 4)),
        hierarchy_layers=int(m.get("hierarchy_layers", 2)),
        phase_selector_use_3d=bool(m.get("phase_selector_use_3d", True)),
        use_skill_gate=bool(m.get("use_skill_gate", False)),
        use_contact_action_token=bool(m.get("use_contact_action_token", False)),
        split_action_heads=bool(m.get("split_action_heads", False)),
        gripper_open_width=float(m.get("gripper_open_width", 0.04)),
        gripper_closed_width=float(m.get("gripper_closed_width", 0.0)),
        use_contact_residual=bool(m.get("use_contact_residual", False)),
        contact_residual_max_translation=float(
            m.get("contact_residual_max_translation", 0.03)
        ),
        use_contact_relative_actions=bool(m.get("use_contact_relative_actions", False)),
        contact_relative_max_translation=float(
            m.get("contact_relative_max_translation", 0.40)
        ),
        use_unified_contact_conditioning=bool(m.get("use_unified_contact_conditioning", False)),
        contact_conditioning_enabled=bool(m.get("contact_conditioning_enabled", True)),
        contact_geometry_scale_m=float(m.get("contact_geometry_scale_m", 0.1)),
        use_near_contact_actions=bool(m.get("use_near_contact_actions", False)),
        near_contact_distance=float(m.get("near_contact_distance", 0.03)),
        near_contact_max_translation=float(m.get("near_contact_max_translation", 0.03)),
        use_action_deltas=bool(m.get("use_action_deltas", False)),
        use_point_part_head=bool(m.get("use_point_part_head", False)),
        use_gripper_state=bool(m.get("use_gripper_state", False)),
    )


def freeze_prefixes(model: Part2ActionModel, prefixes: list[str]) -> list[str]:
    frozen = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def unfreeze_prefixes(model: Part2ActionModel, prefixes: list[str]) -> list[str]:
    unfrozen = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            parameter.requires_grad_(True)
            unfrozen.append(name)
    return unfrozen


def reset_prefixes(model: Part2ActionModel, prefixes: list[str]) -> list[str]:
    """Reset incompatible initialized modules before fine-tuning."""
    reset = []
    for prefix in prefixes:
        try:
            module = model.get_submodule(prefix)
        except AttributeError as exc:
            raise ValueError(f"Cannot reset unknown model prefix: {prefix}") from exc
        for child in module.modules():
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
        reset.extend(
            name
            for name, _ in model.named_parameters()
            if name == prefix or name.startswith(f"{prefix}.")
        )
    return reset


def optimizer_groups(model: Part2ActionModel, base_lr: float, lr_scales: dict) -> list[dict]:
    groups: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith(("visual.model.", "text.model.")):
            continue
        scale = 1.0
        for prefix, candidate in lr_scales.items():
            if name == prefix or name.startswith(f"{prefix}."):
                scale = float(candidate)
                break
        groups.setdefault(scale, []).append(parameter)
    return [{"params": params, "lr": base_lr * scale} for scale, params in groups.items()]


def trainable_state_dict(model):
    # Frozen policy modules may be initialized from a baseline checkpoint and
    # must remain in the saved artifact. Only omit the reloadable foundation
    # model weights to keep checkpoints compact.
    excluded = ("visual.model.", "text.model.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(excluded)
    }


def gate_loss(logits, mask, weights):
    side = int(logits.shape[-1] ** 0.5)
    target = F.interpolate(mask[:, None], size=(side, side), mode="area").flatten(1)
    pos_weight = torch.tensor(
        float(weights.get("gate_pos_weight", 10.0)),
        device=logits.device, dtype=logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    gamma = float(weights.get("gate_focal_gamma", 2.0))
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - pt).pow(gamma) * bce).mean()
    intersection = (probability * target).sum(dim=-1)
    dice = 1.0 - (
        (2.0 * intersection + 1.0)
        / (probability.sum(dim=-1) + target.sum(dim=-1) + 1.0)
    ).mean()
    return focal + float(weights.get("gate_dice_weight", 1.0)) * dice


def point_part_loss(logits, target, sample_valid, weights):
    target = target.to(dtype=logits.dtype)
    sample_valid = sample_valid.to(dtype=logits.dtype)
    pos_weight = torch.tensor(
        float(weights.get("point_part_pos_weight", 50.0)),
        device=logits.device,
        dtype=logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    gamma = float(weights.get("point_part_focal_gamma", 2.0))
    focal_per_sample = ((1.0 - pt).pow(gamma) * bce).mean(dim=-1)
    intersection = (probability * target).sum(dim=-1)
    dice_per_sample = 1.0 - (
        (2.0 * intersection + 1.0)
        / (probability.sum(dim=-1) + target.sum(dim=-1) + 1.0)
    )
    denominator = sample_valid.sum().clamp(min=1.0)
    focal = (focal_per_sample * sample_valid).sum() / denominator
    dice = (dice_per_sample * sample_valid).sum() / denominator
    return focal + float(weights.get("point_part_dice_weight", 1.0)) * dice


def contact_attention_part_loss(logits, scene_xyz, contact_xyz, part_labels, sample_valid, weights):
    """Supervise contact attention with a part-restricted Gaussian around contact XYZ."""
    part_labels = part_labels.to(dtype=logits.dtype)
    sample_valid = sample_valid.to(dtype=logits.dtype)
    valid_xyz = torch.isfinite(scene_xyz).all(dim=-1)
    sigma = float(weights.get("contact_attn_sigma", 0.03))
    dist2 = ((scene_xyz - contact_xyz[:, None, :]) ** 2).sum(dim=-1)
    target = torch.exp(-dist2 / (2.0 * sigma * sigma)) * part_labels
    target = torch.where(valid_xyz, target, torch.zeros_like(target))
    target_sum = target.sum(dim=-1)
    sample_valid = sample_valid * (target_sum > 0).to(dtype=logits.dtype)
    target = target / target_sum.clamp(min=1e-8)[:, None]
    log_pred = F.log_softmax(logits.float(), dim=-1)
    loss_per = -(target * log_pred).sum(dim=-1)
    denominator = sample_valid.sum().clamp(min=1.0)
    loss = (loss_per * sample_valid).sum() / denominator
    pred = torch.softmax(logits.float(), dim=-1)
    attn_on_part = (pred * part_labels * valid_xyz.to(pred.dtype)).sum(dim=-1)
    attn_on_part = (attn_on_part * sample_valid).sum() / denominator
    return loss.to(dtype=logits.dtype), attn_on_part.to(dtype=logits.dtype)


def compute_losses(
    out, batch, weights, active,
    use_action_deltas: bool = False,
    use_contact_relative_actions: bool = False,
):
    parts, total = {}, torch.zeros((), device=out["fused"].device)
    action_target = batch["action_chunk"]
    if use_action_deltas:
        if "tcp_pose" not in batch:
            raise ValueError("use_action_deltas requires batch tcp_pose")
        action_target = absolute_actions_to_tcp_deltas(batch["action_chunk"], batch["tcp_pose"])
    if "heatmap" in active and weights.get("heatmap_weight", 0) > 0:
        target = F.interpolate(
            batch["part_mask"][:, None], size=out["heatmap_logits"].shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(out["heatmap_logits"], target)
        parts["heatmap"], total = loss.detach(), total + weights["heatmap_weight"] * loss
    if "gate_logits" in out and weights.get("gate_weight", 0) > 0:
        loss = gate_loss(out["gate_logits"], batch["part_mask"], weights)
        parts["gate"], total = loss.detach(), total + weights["gate_weight"] * loss
    if "point_part_logits" in out and weights.get("point_part_weight", 0) > 0:
        loss = point_part_loss(
            out["point_part_logits"],
            batch["point_part_labels"],
            batch["point_part_valid"],
            weights,
        )
        parts["point_part"] = loss.detach()
        total = total + weights["point_part_weight"] * loss
        valid = batch["point_part_valid"].bool()[:, None]
        target = batch["point_part_labels"].bool()
        predicted = out["point_part_logits"] >= 0
        true_positive = (predicted & target & valid).sum().float()
        false_positive = (predicted & ~target & valid).sum().float()
        false_negative = (~predicted & target & valid).sum().float()
        precision = true_positive / (true_positive + false_positive).clamp(min=1)
        recall = true_positive / (true_positive + false_negative).clamp(min=1)
        parts["point_part_precision"] = precision.detach()
        parts["point_part_recall"] = recall.detach()
        parts["point_part_f1"] = (
            2.0 * precision * recall / (precision + recall).clamp(min=1e-8)
        ).detach()
    if "contact" in active and weights.get("contact_weight", 0) > 0:
        loss = F.l1_loss(out["contact_xy"], batch["contact_xy"])
        parts["contact"], total = loss.detach(), total + weights["contact_weight"] * loss
    if "contact_xyz" in out and weights.get("contact_xyz_weight", 0) > 0:
        per = F.smooth_l1_loss(out["contact_xyz"], batch["contact_xyz"], reduction="none").mean(-1)
        valid = batch["contact_xyz_valid"]
        loss = (per * valid).sum() / valid.sum().clamp(min=1)
        parts["contact_xyz"], total = loss.detach(), total + weights["contact_xyz_weight"] * loss
    if (
        "contact_point_logits" in out
        and weights.get("contact_attn_weight", 0) > 0
    ):
        if "point_part_labels" not in batch:
            raise ValueError("contact_attn_weight requires batch point_part_labels")
        sample_valid = batch["point_part_valid"] * batch.get(
            "contact_xyz_valid", torch.ones_like(batch["point_part_valid"])
        )
        loss, attn_on_part = contact_attention_part_loss(
            out["contact_point_logits"],
            camera_to_world_torch(batch["agentview_pcd"][..., :3]),
            batch["contact_xyz"],
            batch["point_part_labels"],
            sample_valid,
            weights,
        )
        parts["contact_attn"] = loss.detach()
        parts["contact_attn_on_part"] = attn_on_part.detach()
        total = total + weights["contact_attn_weight"] * loss
    if "approach" in active and weights.get("approach_weight", 0) > 0:
        loss = (1 - (out["approach_dir"] * batch["approach_dir"]).sum(-1).clamp(-1, 1)).mean()
        parts["approach"], total = loss.detach(), total + weights["approach_weight"] * loss
    if "action" in active and "gripper_logits" in out:
        arm_loss = (
            F.mse_loss(out["action_noise_pred"], out["action_noise"])
            if "action_noise_pred" in out
            else F.smooth_l1_loss(out["arm_action_chunk"], action_target[..., :6])
        )
        parts["arm_action"] = arm_loss.detach()
        total = total + weights.get("arm_action_weight", weights.get("action_weight", 1.0)) * arm_loss
        pos_weight = torch.tensor(
            float(weights.get("gripper_pos_weight", 1.0)), device=total.device
        )
        grip_per = F.binary_cross_entropy_with_logits(
            out["gripper_logits"], batch["gripper_closed"],
            pos_weight=pos_weight, reduction="none",
        )
        grip_loss = (grip_per * batch["gripper_loss_weight"]).sum() / (
            batch["gripper_loss_weight"].sum().clamp(min=1)
        )
        parts["gripper"] = grip_loss.detach()
        total = total + weights.get("gripper_weight", 1.0) * grip_loss
        grip_pred = out["gripper_logits"] >= 0
        parts["gripper_accuracy"] = (
            grip_pred == batch["gripper_closed"].bool()
        ).float().mean().detach()
    elif "action" in active and weights.get("action_weight", 0) > 0:
        if use_contact_relative_actions:
            if "contact_xyz_offset" not in out:
                raise ValueError("contact-relative actions require contact_xyz_offset")
            if "contact_xyz" not in batch:
                raise ValueError("contact-relative actions require batch contact_xyz")
            if "baseline_action_chunk" not in out:
                raise ValueError("contact-relative actions require baseline_action_chunk")
            target_relative = absolute_actions_to_contact_relative(
                batch["action_chunk"], batch["contact_xyz"]
            )
            offset_loss = F.smooth_l1_loss(
                out["contact_xyz_offset"], target_relative[..., :3]
            )
            ori_grip_loss = F.smooth_l1_loss(
                out["baseline_action_chunk"][..., 3:], batch["action_chunk"][..., 3:]
            )
            loss = offset_loss + ori_grip_loss
            parts["action"] = loss.detach()
            parts["action_contact_offset_l1"] = F.l1_loss(
                out["contact_xyz_offset"], target_relative[..., :3]
            ).detach()
            parts["action_ori_gripper"] = ori_grip_loss.detach()
            parts["action_offset_mean_norm"] = (
                out["contact_xyz_offset"].norm(dim=-1).mean().detach()
            )
            total = total + weights["action_weight"] * loss
        else:
            loss = (
                F.mse_loss(out["action_noise_pred"], out["action_noise"])
                if "action_noise_pred" in out
                else F.smooth_l1_loss(out["action_chunk"], action_target)
            )
            parts["action"], total = loss.detach(), total + weights["action_weight"] * loss
            if use_action_deltas:
                parts["action_delta_xyz_l1"] = F.l1_loss(
                    out["action_chunk"][..., :3], action_target[..., :3]
                ).detach()
    if "near_contact" in out:
        parts["near_contact_fraction"] = out["near_contact"].float().mean().detach()
    if "plan_slots" in out and weights.get("skill_embedding_weight", 0) > 0:
        sim = F.cosine_similarity(out["plan_slots"], out["skill_target_slots"], dim=-1)
        valid = batch["skill_valid_mask"]
        loss = ((1 - sim) * valid).sum() / valid.sum().clamp(min=1)
        parts["skill_embedding"], total = loss.detach(), total + weights["skill_embedding_weight"] * loss
    if "slot_valid_logits" in out and weights.get("slot_validity_weight", 0) > 0:
        loss = F.binary_cross_entropy_with_logits(out["slot_valid_logits"], batch["skill_valid_mask"])
        parts["slot_validity"], total = loss.detach(), total + weights["slot_validity_weight"] * loss
    if "phase_logits" in out and weights.get("phase_weight", 0) > 0:
        loss = F.cross_entropy(out["phase_logits"], batch["phase_index"])
        parts["phase"], total = loss.detach(), total + weights["phase_weight"] * loss
        parts["phase_accuracy"] = (out["phase_logits"].argmax(-1) == batch["phase_index"]).float().mean().detach()
    if "termination_logits" in out and weights.get("termination_weight", 0) > 0:
        pos_weight = torch.tensor(float(weights.get("termination_pos_weight", 1)), device=total.device)
        loss = F.binary_cross_entropy_with_logits(
            out["termination_logits"], batch["phase_termination"], pos_weight=pos_weight
        )
        parts["termination"], total = loss.detach(), total + weights["termination_weight"] * loss
        parts["termination_accuracy"] = (
            (out["termination_logits"] >= 0) == batch["phase_termination"].bool()
        ).float().mean().detach()
    parts["total"] = total.detach()
    return total, parts


def curriculum_state(epoch: int, cfg: dict):
    h, weights = cfg.get("hierarchy", {}), dict(cfg.get("losses", {}))
    warm, planner = int(h.get("warmup_epochs", 0)), int(h.get("planner_epochs", 0))
    if epoch < warm:
        for key in ("skill_embedding_weight", "slot_validity_weight", "phase_weight", "termination_weight"):
            weights[key] = 0.0
        return "oracle_warmup", 0.0, weights
    if epoch < warm + planner:
        return "planner", 0.0, weights
    ramp = max(1, int(h.get("scheduled_sampling_epochs", 1)))
    probability = min(1.0, (epoch - warm - planner + 1) / ramp)
    return "end_to_end", probability * float(h.get("max_predicted_probability", 1)), weights


def forward_kwargs(batch, hierarchy, probability, include_targets):
    kwargs = {k: batch[k] for k in ("agentview_pcd", "tcp_pose", "gripper_state") if k in batch}
    if hierarchy:
        kwargs.update(
            skill_valid_mask=batch["skill_valid_mask"], phase_index=batch["phase_index"],
            oracle_skill_instructions=batch["instruction"],
            scheduled_sampling_prob=probability,
        )
        if include_targets:
            kwargs["skill_plan"] = batch["skill_plan"]
    if "contact_xyz" in batch and "contact_xyz_valid" in batch:
        kwargs["contact_xyz_target"] = batch["contact_xyz"]
        kwargs["contact_xyz_valid"] = batch["contact_xyz_valid"]
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--override-out", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.config).expanduser().resolve()
    cfg = resolve_paths(load_yaml(source), ROOT)
    if args.override_out:
        cfg["output_dir"] = args.override_out
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    out_dir = ensure_dir(cfg["output_dir"])
    shutil.copy2(source, out_dir / "config_source.yaml")
    device = select_device(cfg["train"].get("device", "cuda"))
    if device.type == "cpu":
        cfg["model"]["text_device"] = "cpu"
    ds, hierarchy = build_dataset(cfg["data"]), bool(cfg["model"].get("use_hierarchy", False))
    dl = build_dataloader(ds, cfg["data"])
    model = build_model(cfg).to(device)
    config_init = cfg["train"].get("init_checkpoint")
    init_from = args.init_from or config_init
    if args.resume:
        if args.init_from:
            raise ValueError("--resume and --init-from are mutually exclusive")
        init_from = None
        print(f"[train] resume {args.resume}; ignoring train.init_checkpoint")
    if init_from and not Path(init_from).is_absolute():
        init_from = str((ROOT / init_from).resolve())
    if init_from:
        initial = torch.load(init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        print(
            f"[train] initialized from {init_from}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    reset = reset_prefixes(model, list(cfg["train"].get("reset_prefixes", [])))
    if reset:
        print(f"[train] reset {len(reset)} parameter tensors after initialization")
    frozen = freeze_prefixes(model, list(cfg["train"].get("freeze_prefixes", [])))
    if frozen:
        print(f"[train] froze {len(frozen)} parameter tensors")
    active = set(cfg["heads"])
    print(f"[train] device={device} dataset={len(ds)} batches={len(dl)}")

    if args.dry_run:
        batch = next(iter(dl))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        epoch = int(cfg.get("hierarchy", {}).get("warmup_epochs", 0))
        stage, probability, weights = curriculum_state(epoch, cfg)
        out = model(
            batch["rgb"], batch["task_instruction"] if hierarchy else batch["instruction"],
            target_action=batch["action_chunk"] if "action" in active else None,
            **forward_kwargs(batch, hierarchy, probability, weights.get("skill_embedding_weight", 0) > 0),
        )
        total, parts = compute_losses(
            out, batch, weights, active,
            use_action_deltas=model.use_action_deltas,
            use_contact_relative_actions=model.use_contact_relative_actions,
        )
        print("[train] dry_run stage", stage, "outputs", sorted(k for k in out if k != "fused"))
        print("[train] dry_run loss", float(total), {k: float(v) for k, v in parts.items()})
        return

    base_lr = float(cfg["train"].get("lr", 3e-4))
    opt = torch.optim.AdamW(
        optimizer_groups(model, base_lr, cfg["train"].get("lr_scales", {})),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, start_epoch = [], 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        history = list(state.get("history", []))
        start_epoch = int(state.get("epoch", -1)) + 1
    start = time.time()
    for epoch in range(start_epoch, int(cfg["train"].get("epochs", 5))):
        for event in cfg["train"].get("unfreeze_schedule", []):
            if int(event.get("epoch", -1)) == epoch:
                names = unfreeze_prefixes(model, list(event.get("prefixes", [])))
                print(
                    f"[train] epoch {epoch} unfroze {len(names)} parameter tensors "
                    f"for {event.get('prefixes', [])}"
                )
        model.train()
        stage, probability, weights = curriculum_state(epoch, cfg)
        running, seen = {}, 0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1} {stage} p={probability:.2f}")
        for batch in pbar:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = model(
                    batch["rgb"], batch["task_instruction"] if hierarchy else batch["instruction"],
                    target_action=batch["action_chunk"] if "action" in active else None,
                    **forward_kwargs(batch, hierarchy, probability, weights.get("skill_embedding_weight", 0) > 0),
                )
                total, parts = compute_losses(
                    out, batch, weights, active,
                    use_action_deltas=model.use_action_deltas,
                    use_contact_relative_actions=model.use_contact_relative_actions,
                )
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), float(cfg["train"].get("grad_clip", 1)))
            scaler.step(opt)
            scaler.update()
            seen += 1
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value)
            if seen % int(cfg["train"].get("log_every", 50)) == 0:
                pbar.set_postfix({k: round(v / seen, 4) for k, v in running.items()})
        summary = {k: v / max(1, seen) for k, v in running.items()}
        summary.update(epoch=epoch, hierarchy_stage=stage, predicted_conditioning=probability)
        history.append(summary)
        print("[train] epoch done", summary)
        if (epoch + 1) % int(cfg["train"].get("save_every_epoch", 1)) == 0:
            torch.save(
                {"model": trainable_state_dict(model), "cfg": cfg, "epoch": epoch,
                 "history": history, "trainable_only": True},
                out_dir / "last.pt",
            )
    save_json(out_dir / "history.json", history)
    save_json(out_dir / "config_resolved.json", cfg)
    print(f"[train] done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
