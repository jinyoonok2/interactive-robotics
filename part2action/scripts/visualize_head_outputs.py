"""Visualize non-action head predictions on PartInstruct HDF5 samples.

This is an offline qualitative tool, not a PartGym rollout. It works for
heatmap/contact/approach checkpoints that do not produce action chunks by
overlaying predictions on demonstration RGB frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, load_yaml, resolve_paths, select_device, set_seed
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel


DEFAULT_MODELS = [
    "heatmap_real",
    "heatmap_contact_real",
    "heatmap_approach_real",
    "heatmap_contact_approach_real",
]


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> Part2ActionModel:
    if str(device) == "cpu":
        cfg["model"]["text_device"] = "cpu"
    model = Part2ActionModel(
        heads=cfg["heads"],
        img_size=int(cfg["model"].get("img_size", 252)),
        out_size=int(cfg["model"].get("out_size", 96)),
        action_chunk=int(cfg["data"].get("action_chunk", 8)),
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_fusion_layers=int(cfg["model"].get("num_fusion_layers", 2)),
        text_device=cfg["model"].get("text_device", "cpu"),
        use_unified_contact_conditioning=bool(cfg["model"].get("use_unified_contact_conditioning", False)),
        contact_conditioning_enabled=bool(cfg["model"].get("contact_conditioning_enabled", True)),
        contact_geometry_scale_m=float(cfg["model"].get("contact_geometry_scale_m", 0.1)),
        action_head_type=cfg["model"].get("action_head_type", "mlp"),
        diffusion_steps=int(cfg["model"].get("diffusion_steps", 50)),
        temporal_encoder_type=cfg["model"].get("temporal_encoder_type", "none"),
        n_obs_steps=int(cfg["data"].get("n_obs_steps", 1)),
        temporal_layers=int(cfg["model"].get("temporal_layers", 1)),
        temporal_heads=int(cfg["model"].get("temporal_heads", 4)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _rgb_from_sample(sample: dict[str, Any]) -> np.ndarray:
    rgb = sample["rgb"]
    if rgb.ndim == 4:
        rgb = rgb[-1]
    return rgb.permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _iou(logits: torch.Tensor, mask: torch.Tensor) -> float:
    if logits.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1), size=logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * mask).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + mask.sum(dim=(-1, -2)) - inter
    return float((inter / union.clamp(min=1e-6)).mean().item())


def _draw_vector(ax, origin_xy: np.ndarray, vec: np.ndarray, color: str, label: str) -> None:
    scale = 45.0
    norm = np.linalg.norm(vec[:2])
    direction = vec[:2] / max(norm, 1e-6)
    ax.arrow(
        origin_xy[0],
        origin_xy[1],
        direction[0] * scale,
        direction[1] * scale,
        color=color,
        width=1.5,
        head_width=8,
        length_includes_head=True,
        label=label,
    )


def _save_sample_figure(
    out_path: Path,
    *,
    model_name: str,
    sample: dict[str, Any],
    out: dict[str, torch.Tensor],
    metrics: dict[str, float],
) -> None:
    rgb = _rgb_from_sample(sample)
    mask = sample["part_mask"].numpy()
    h, w = rgb.shape[:2]
    instruction = sample["instruction"]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"{model_name} | {instruction}", fontsize=10)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.set_title("RGB")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(rgb)
    ax.imshow(mask, alpha=0.45, cmap="Greens")
    ax.set_title("GT part mask")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(rgb)
    if "heatmap_logits" in out:
        heatmap = torch.sigmoid(out["heatmap_logits"][0]).detach().cpu().numpy()
        ax.imshow(heatmap, alpha=0.45, cmap="magma")
    title_bits = []
    if "heatmap_iou" in metrics:
        title_bits.append(f"IoU={metrics['heatmap_iou']:.3f}")

    gt_contact = sample["contact_xy"].numpy()
    gt_px = np.array([gt_contact[0] * (w - 1), gt_contact[1] * (h - 1)])
    ax.scatter([gt_px[0]], [gt_px[1]], c="lime", marker="x", s=90, label="GT contact")

    if "contact_xy" in out:
        pred_contact = out["contact_xy"][0].detach().cpu().numpy()
        pred_px = np.array([pred_contact[0] * (w - 1), pred_contact[1] * (h - 1)])
        ax.scatter([pred_px[0]], [pred_px[1]], c="cyan", marker="o", s=65, label="Pred contact")
        title_bits.append(f"contact={metrics['contact_px_l2']:.1f}px")
        vector_origin = pred_px
    else:
        vector_origin = gt_px

    if "approach_dir" in out:
        pred_approach = out["approach_dir"][0].detach().cpu().numpy()
        gt_approach = sample["approach_dir"].numpy()
        _draw_vector(ax, vector_origin, pred_approach, "cyan", "Pred approach")
        _draw_vector(ax, gt_px, gt_approach, "lime", "GT approach")
        title_bits.append(f"approach cos={metrics['approach_cos']:.3f}")

    ax.set_title("Predictions\n" + ", ".join(title_bits))
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _metrics(out: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "heatmap_logits" in out:
        metrics["heatmap_iou"] = _iou(out["heatmap_logits"], batch["part_mask"])
    if "contact_xy" in out:
        delta = out["contact_xy"] - batch["contact_xy"]
        contact_px = torch.linalg.norm(delta * torch.tensor([300.0, 300.0], device=delta.device), dim=-1)
        metrics["contact_px_l2"] = float(contact_px.mean().item())
        metrics["contact_l1_norm"] = float(F.l1_loss(out["contact_xy"], batch["contact_xy"]).item())
    if "approach_dir" in out:
        cos = (out["approach_dir"] * batch["approach_dir"]).sum(dim=-1).clamp(-1.0, 1.0)
        metrics["approach_cos"] = float(cos.mean().item())
        metrics["approach_angle_deg"] = float(torch.rad2deg(torch.acos(cos)).mean().item())
    return metrics


def run_model(model_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model_dir = args.run_dir / model_name
    config_path = model_dir / "config_source.yaml"
    ckpt_path = model_dir / "last.pt"
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config/checkpoint for {model_name}: {config_path}, {ckpt_path}")

    cfg = resolve_paths(load_yaml(config_path), ROOT)
    ds = PartInstructDataset(
        hdf5_paths=cfg["data"].get("val_hdf5") or cfg["data"]["train_hdf5"],
        action_chunk=cfg["data"].get("action_chunk", 8),
        sample_stride=max(1, int(args.sample_stride)),
        max_demos_per_file=args.max_demos_per_file,
        n_obs_steps=cfg["data"].get("n_obs_steps", 1),
    )
    model = _load_model(cfg, ckpt_path, device)
    indices = np.linspace(0, len(ds) - 1, num=min(args.num_samples, len(ds)), dtype=int).tolist()

    out_dir = args.out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    for sample_id, idx in enumerate(indices):
        sample = ds[idx]
        batch = _to_device(collate_part2action([sample]), device)
        with torch.no_grad():
            out = model(batch["rgb"], batch["instruction"])
        metrics = _metrics(out, batch)
        row = {
            "sample_id": sample_id,
            "dataset_index": int(idx),
            "instruction": sample["instruction"],
            "meta": sample["meta"],
            **metrics,
        }
        rows.append(row)
        png_path = out_dir / f"sample_{sample_id:03d}.png"
        _save_sample_figure(png_path, model_name=model_name, sample=sample, out=out, metrics=metrics)
        frames.append(imageio.imread(png_path))

    if frames:
        imageio.mimsave(out_dir / "summary.gif", frames, duration=float(args.gif_duration), loop=0)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0].keys()
        if isinstance(rows[0][key], (int, float))
    }
    payload = {"model": model_name, "checkpoint": str(ckpt_path), "samples": rows, "summary": summary}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    return {"model": model_name, "out_dir": str(out_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "runs" / "slurm_6201976")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=300)
    parser.add_argument("--max-demos-per-file", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "visualizations" / "head_outputs")
    parser.add_argument("--gif-duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_model(model_name, args, device) for model_name in args.models]
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
