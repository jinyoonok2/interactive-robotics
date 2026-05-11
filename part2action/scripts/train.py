"""Unified trainer for heatmap, part-action, diffusion, and temporal tracks.

Selects active heads + loss weights from the YAML config so the same
script trains all variants with no code duplication.

Usage:
    python scripts/train.py --config configs/heatmap_synth.yaml
    python scripts/train.py --config configs/part_action_mlp_synth.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

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
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel


def build_dataset(cfg_data: dict) -> PartInstructDataset:
    return PartInstructDataset(
        hdf5_paths=cfg_data["train_hdf5"],
        action_chunk=cfg_data.get("action_chunk", 8),
        sample_stride=cfg_data.get("sample_stride", 1),
        max_demos_per_file=cfg_data.get("max_demos_per_file"),
        n_obs_steps=cfg_data.get("n_obs_steps", 1),
    )


def build_dataloader(ds: PartInstructDataset, cfg_data: dict) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg_data.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg_data.get("num_workers", 0)),
        pin_memory=bool(cfg_data.get("pin_memory", True)),
        collate_fn=collate_part2action,
        drop_last=True,
    )


def trainable_state_dict(model: torch.nn.Module) -> dict:
    """Return state_dict slice containing ONLY parameters with requires_grad=True.

    Frozen DINOv2 + Flan-T5 are reloaded from local caches by the model
    constructor, so we save and ship only the small trainable bits
    (fusion + heads). This keeps checkpoints in the MB range instead of
    ~500 MB.
    """
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    full = model.state_dict()
    return {k: v.detach().cpu() for k, v in full.items() if k in trainable_keys}


def heatmap_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask.unsqueeze(1), size=logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    else:
        mask_resized = mask
    return F.binary_cross_entropy_with_logits(logits, mask_resized)


def compute_losses(out: dict, batch: dict, weights: dict, active_heads: set) -> tuple[torch.Tensor, dict]:
    parts: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=out["fused"].device)

    if "heatmap" in active_heads and weights.get("heatmap_weight", 0) > 0:
        l = heatmap_loss(out["heatmap_logits"], batch["part_mask"])
        parts["heatmap"] = l.detach()
        total = total + weights["heatmap_weight"] * l

    if "contact" in active_heads and weights.get("contact_weight", 0) > 0:
        l = F.l1_loss(out["contact_xy"], batch["contact_xy"])
        parts["contact"] = l.detach()
        total = total + weights["contact_weight"] * l

    if "approach" in active_heads and weights.get("approach_weight", 0) > 0:
        cos = (out["approach_dir"] * batch["approach_dir"]).sum(dim=-1).clamp(-1.0, 1.0)
        l = (1.0 - cos).mean()
        parts["approach"] = l.detach()
        total = total + weights["approach_weight"] * l

    if "action" in active_heads and weights.get("action_weight", 0) > 0:
        if "action_noise_pred" in out:
            l = F.mse_loss(out["action_noise_pred"], out["action_noise"])
        else:
            l = F.smooth_l1_loss(out["action_chunk"], batch["action_chunk"])
        parts["action"] = l.detach()
        total = total + weights["action_weight"] * l

    parts["total"] = total.detach()
    return total, parts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override-out", default=None, help="Override output_dir from CLI")
    p.add_argument("--resume", default=None, help="Resume model weights/history from a checkpoint")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    cfg = resolve_paths(cfg, ROOT)
    if args.override_out:
        cfg["output_dir"] = args.override_out

    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    out_dir = ensure_dir(cfg["output_dir"])

    device = select_device(cfg["train"].get("device", "cuda"))
    print(f"[train] device={device}  cfg={args.config}")

    ds = build_dataset(cfg["data"])
    dl = build_dataloader(ds, cfg["data"])
    print(f"[train] dataset size: {len(ds)} timesteps  ({len(dl)} batches/epoch)")

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
    ).to(device)

    opt = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
    )
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_every = int(cfg["train"].get("log_every", 25))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    # Use bfloat16 instead of float16: same compute savings but full float32
    # dynamic range, eliminating the NaN overflows seen with fp16 + DINOv2/T5.
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weights = cfg.get("losses", {})
    active = set(cfg["heads"])

    history: list[dict] = []
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected:
            print(f"[train] resume unexpected keys: {list(unexpected)[:3]}{'...' if len(unexpected) > 3 else ''}")
        if missing and not state.get("trainable_only", False):
            print(f"[train] resume missing keys: {list(missing)[:3]}{'...' if len(missing) > 3 else ''}")
        history = list(state.get("history", []))
        start_epoch = int(state.get("epoch", -1)) + 1
        print(f"[train] resumed from {args.resume}; continuing at epoch {start_epoch + 1}")

    epochs = int(cfg["train"].get("epochs", 5))
    save_every = int(cfg["train"].get("save_every_epoch", 1))
    global_step = 0
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        for p in model.visual.parameters():
            p.requires_grad = False
        running: dict[str, float] = {}
        n_seen = 0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                target_action = batch["action_chunk"] if "action" in active else None
                out = model(batch["rgb"], batch["instruction"], target_action=target_action)
                total, parts = compute_losses(out, batch, weights, active)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), grad_clip)
            scaler.step(opt)
            scaler.update()

            n_seen += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v)
            if global_step % log_every == 0:
                msg = {k: round(v / max(1, n_seen), 4) for k, v in running.items()}
                pbar.set_postfix(msg)
            global_step += 1

        epoch_summary = {k: v / max(1, n_seen) for k, v in running.items()}
        epoch_summary["epoch"] = epoch
        history.append(epoch_summary)
        print(f"[train] epoch {epoch + 1} done: {epoch_summary}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = out_dir / "last.pt"
            torch.save(
                {
                    "model": trainable_state_dict(model),
                    "cfg": cfg,
                    "epoch": epoch,
                    "history": history,
                    "trainable_only": True,
                },
                ckpt_path,
            )
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"[train] saved {ckpt_path} ({size_mb:.1f} MB, trainable params only)")

    save_json(out_dir / "history.json", history)
    save_json(out_dir / "config_resolved.json", cfg)
    print(f"[train] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
