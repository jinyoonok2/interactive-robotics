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
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel

DEFAULT_CONFIG = ROOT / "configs" / "architecture_update" / "hierarchical_gated_pcd_contact3d_all_demos.yaml"


def build_dataset(data: dict) -> PartInstructDataset:
    return PartInstructDataset(
        hdf5_paths=data["train_hdf5"], action_chunk=data.get("action_chunk", 8),
        sample_stride=data.get("sample_stride", 1),
        max_demos_per_file=data.get("max_demos_per_file"),
        n_obs_steps=data.get("n_obs_steps", 1), use_pcd=data.get("use_pcd", False),
        use_part_pcd=data.get("use_part_pcd", False),
        use_tcp_pose=data.get("use_tcp_pose", False),
        use_contact_xyz=data.get("use_contact_xyz", False),
        use_hierarchy=data.get("use_hierarchy", False),
        max_skill_slots=data.get("max_skill_slots", 4),
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
    )


def trainable_state_dict(model):
    keys = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keys}


def gate_loss(logits, mask):
    side = int(logits.shape[-1] ** 0.5)
    target = F.interpolate(mask[:, None], size=(side, side), mode="area").flatten(1)
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_losses(out, batch, weights, active):
    parts, total = {}, torch.zeros((), device=out["fused"].device)
    if "heatmap" in active and weights.get("heatmap_weight", 0) > 0:
        target = F.interpolate(
            batch["part_mask"][:, None], size=out["heatmap_logits"].shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(out["heatmap_logits"], target)
        parts["heatmap"], total = loss.detach(), total + weights["heatmap_weight"] * loss
    if "gate_logits" in out and weights.get("gate_weight", 0) > 0:
        loss = gate_loss(out["gate_logits"], batch["part_mask"])
        parts["gate"], total = loss.detach(), total + weights["gate_weight"] * loss
    if "contact" in active and weights.get("contact_weight", 0) > 0:
        loss = F.l1_loss(out["contact_xy"], batch["contact_xy"])
        parts["contact"], total = loss.detach(), total + weights["contact_weight"] * loss
    if "contact_xyz" in out and weights.get("contact_xyz_weight", 0) > 0:
        per = F.smooth_l1_loss(out["contact_xyz"], batch["contact_xyz"], reduction="none").mean(-1)
        valid = batch["contact_xyz_valid"]
        loss = (per * valid).sum() / valid.sum().clamp(min=1)
        parts["contact_xyz"], total = loss.detach(), total + weights["contact_xyz_weight"] * loss
    if "approach" in active and weights.get("approach_weight", 0) > 0:
        loss = (1 - (out["approach_dir"] * batch["approach_dir"]).sum(-1).clamp(-1, 1)).mean()
        parts["approach"], total = loss.detach(), total + weights["approach_weight"] * loss
    if "action" in active and weights.get("action_weight", 0) > 0:
        loss = (
            F.mse_loss(out["action_noise_pred"], out["action_noise"])
            if "action_noise_pred" in out
            else F.smooth_l1_loss(out["action_chunk"], batch["action_chunk"])
        )
        parts["action"], total = loss.detach(), total + weights["action_weight"] * loss
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
    kwargs = {k: batch[k] for k in ("agentview_pcd", "tcp_pose") if k in batch}
    if hierarchy:
        kwargs.update(
            skill_valid_mask=batch["skill_valid_mask"], phase_index=batch["phase_index"],
            oracle_skill_instructions=batch["instruction"],
            scheduled_sampling_prob=probability,
        )
        if include_targets:
            kwargs["skill_plan"] = batch["skill_plan"]
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--override-out", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = Path(args.config).expanduser().resolve()
    cfg = resolve_paths(load_yaml(source), ROOT)
    if args.override_out:
        cfg["output_dir"] = args.override_out
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    out_dir = ensure_dir(cfg["output_dir"])
    shutil.copy2(source, out_dir / "config_source.yaml")
    device = select_device(cfg["train"].get("device", "cuda"))
    if device.type == "cpu":
        cfg["model"]["text_device"] = "cpu"
    ds, hierarchy = build_dataset(cfg["data"]), bool(cfg["model"].get("use_hierarchy", False))
    dl = build_dataloader(ds, cfg["data"])
    model = build_model(cfg).to(device)
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
        total, parts = compute_losses(out, batch, weights, active)
        print("[train] dry_run stage", stage, "outputs", sorted(k for k in out if k != "fused"))
        print("[train] dry_run loss", float(total), {k: float(v) for k, v in parts.items()})
        return

    opt = torch.optim.AdamW(
        list(model.trainable_parameters()), lr=float(cfg["train"].get("lr", 3e-4)),
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
                total, parts = compute_losses(out, batch, weights, active)
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
