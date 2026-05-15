"""Probe action-capable part2action checkpoints on PartInstruct samples.

This is an offline sanity check, not a PartGym simulator rollout. It loads
trained checkpoints, predicts action chunks on HDF5 samples, computes action
and auxiliary metrics against demonstrations, and saves visual overlays.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, load_yaml, resolve_paths, select_device, set_seed
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel


MODEL_SPECS = {
    "part_action_mlp_real": {
        "config": ROOT / "configs" / "part_action_mlp_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "part_action_mlp_real" / "last.pt",
    },
    "temporal_part_action_mlp_real": {
        "config": ROOT / "configs" / "temporal_part_action_mlp_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "temporal_part_action_mlp_real" / "last.pt",
    },
    "part_action_diffusion_real": {
        "config": ROOT / "configs" / "part_action_diffusion_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "part_action_diffusion_real" / "last.pt",
    },
    "temporal_part_action_diffusion_real": {
        "config": ROOT / "configs" / "temporal_part_action_diffusion_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "temporal_part_action_diffusion_real" / "last.pt",
    },
}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> Part2ActionModel:
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
    )
    text_device = cfg["model"].get("text_device", "cpu")
    model.to(device)
    # Part2ActionModel.to(device) moves the frozen T5 module too. If the caller
    # wants T5 on CPU to save VRAM, move only that submodule back.
    if text_device != str(device):
        model.text.model.to(text_device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _heatmap_iou(logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    if logits.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(1), size=logits.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * mask).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + mask.sum(dim=(-1, -2)) - inter
    return float((inter / union.clamp(min=1e-6)).mean().item())


def _contact_inside_mask(contact_xy: np.ndarray, mask: np.ndarray) -> bool:
    h, w = mask.shape
    x = int(np.clip(round(float(contact_xy[0]) * (w - 1)), 0, w - 1))
    y = int(np.clip(round(float(contact_xy[1]) * (h - 1)), 0, h - 1))
    return bool(mask[y, x] > 0.5)


def _compute_metrics(out: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
    pred_action = out["action_chunk"]
    gt_action = batch["action_chunk"]
    pred_contact = out["contact_xy"]
    gt_contact = batch["contact_xy"]
    pred_approach = out["approach_dir"]
    gt_approach = batch["approach_dir"]

    contact_delta = pred_contact - gt_contact
    contact_px = torch.linalg.norm(contact_delta * torch.tensor([300.0, 300.0], device=contact_delta.device), dim=-1)
    approach_cos = (pred_approach * gt_approach).sum(dim=-1).clamp(-1.0, 1.0)
    pos_err = torch.linalg.norm(pred_action[..., :3] - gt_action[..., :3], dim=-1)
    final_pos_err = torch.linalg.norm(
        pred_action[..., :3].cumsum(dim=1)[:, -1] - gt_action[..., :3].cumsum(dim=1)[:, -1],
        dim=-1,
    )

    return {
        "heatmap_iou": _heatmap_iou(out["heatmap_logits"], batch["part_mask"]),
        "contact_l1_norm": float(F.l1_loss(pred_contact, gt_contact).item()),
        "contact_px_l2": float(contact_px.mean().item()),
        "approach_cos": float(approach_cos.mean().item()),
        "approach_angle_deg": float(torch.rad2deg(torch.acos(approach_cos)).mean().item()),
        "action_smooth_l1": float(F.smooth_l1_loss(pred_action, gt_action).item()),
        "action_l1": float(F.l1_loss(pred_action, gt_action).item()),
        "pos_delta_l2": float(pos_err.mean().item()),
        "final_cumulative_pos_l2": float(final_pos_err.mean().item()),
        "gripper_l1": float(F.l1_loss(pred_action[..., 6], gt_action[..., 6]).item()),
    }


def _rgb_from_sample(sample: dict[str, Any]) -> np.ndarray:
    rgb = sample["rgb"]
    if rgb.ndim == 4:
        rgb = rgb[-1]
    return rgb.permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _save_visualization(
    out_path: Path,
    sample: dict[str, Any],
    out: dict[str, torch.Tensor],
    metrics: dict[str, float],
    title: str,
) -> None:
    rgb = _rgb_from_sample(sample)
    mask = sample["part_mask"].numpy()
    pred_heatmap = torch.sigmoid(out["heatmap_logits"][0]).detach().cpu().numpy()
    pred_contact = out["contact_xy"][0].detach().cpu().numpy()
    gt_contact = sample["contact_xy"].numpy()
    pred_action = out["action_chunk"][0].detach().cpu().numpy()
    gt_action = sample["action_chunk"].numpy()

    h, w = rgb.shape[:2]
    pred_px = np.array([pred_contact[0] * (w - 1), pred_contact[1] * (h - 1)])
    gt_px = np.array([gt_contact[0] * (w - 1), gt_contact[1] * (h - 1)])
    pred_xyz = np.cumsum(pred_action[:, :3], axis=0)
    gt_xyz = np.cumsum(gt_action[:, :3], axis=0)
    steps = np.arange(pred_action.shape[0])

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=11)

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb)
    ax.set_title("RGB")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(rgb)
    ax.imshow(mask, alpha=0.45, cmap="Greens")
    ax.set_title("GT part mask")
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb)
    ax.imshow(pred_heatmap, alpha=0.45, cmap="magma")
    ax.scatter([gt_px[0]], [gt_px[1]], c="lime", marker="x", s=80, label="GT contact")
    ax.scatter([pred_px[0]], [pred_px[1]], c="cyan", marker="o", s=60, label="Pred contact")
    ax.set_title(
        f"Pred heatmap/contact\nIoU={metrics['heatmap_iou']:.3f}, contact={metrics['contact_px_l2']:.1f}px"
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 4)
    for dim, label in enumerate(("dx", "dy", "dz")):
        ax.plot(steps, gt_action[:, dim], linestyle="--", label=f"GT {label}")
        ax.plot(steps, pred_action[:, dim], label=f"Pred {label}")
    ax.set_title(f"Position deltas\nL2={metrics['pos_delta_l2']:.4f}")
    ax.set_xlabel("chunk step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(steps, gt_action[:, 6], linestyle="--", label="GT gripper")
    ax.plot(steps, pred_action[:, 6], label="Pred gripper")
    ax.set_title(f"Gripper action\nL1={metrics['gripper_l1']:.4f}")
    ax.set_xlabel("chunk step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], marker="x", linestyle="--", label="GT xy path")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], marker="o", label="Pred xy path")
    ax.set_title(f"Cumulative XY path\nfinal L2={metrics['final_cumulative_pos_l2']:.4f}")
    ax.set_xlabel("cum dx")
    ax.set_ylabel("cum dy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, bool))]
    summary: dict[str, float] = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in rows]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


def _collect_existing_summaries(out_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics.json")):
        with open(metrics_path, "r") as f:
            payload = json.load(f)
        summaries.append(
            {
                "model": payload["model"],
                "num_samples": len(payload["samples"]),
                "out_dir": str(metrics_path.parent),
                "summary": payload["summary"],
            }
        )
    return summaries


def run_model(model_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    cfg = resolve_paths(load_yaml(spec["config"]), ROOT)
    if args.text_device is not None:
        cfg["model"]["text_device"] = args.text_device
    if str(device) == "cpu":
        cfg["model"]["text_device"] = "cpu"

    ds = PartInstructDataset(
        hdf5_paths=cfg["data"].get("val_hdf5") or cfg["data"]["train_hdf5"],
        action_chunk=cfg["data"].get("action_chunk", 8),
        sample_stride=max(1, int(args.sample_stride)),
        max_demos_per_file=args.max_demos_per_file,
        n_obs_steps=cfg["data"].get("n_obs_steps", 1),
    )
    model = _load_model(cfg, spec["ckpt"], device)

    out_dir = args.out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(ds) - 1, num=min(args.num_samples, len(ds)), dtype=int).tolist()
    rows: list[dict[str, Any]] = []

    for sample_id, idx in enumerate(indices):
        sample = ds[idx]
        batch = _to_device(collate_part2action([sample]), device)
        with torch.no_grad():
            out = model(batch["rgb"], batch["instruction"])
        metrics = _compute_metrics(out, batch)
        mask = sample["part_mask"].numpy()
        pred_contact = out["contact_xy"][0].detach().cpu().numpy()
        gt_contact = sample["contact_xy"].numpy()
        metrics["pred_contact_inside_mask"] = _contact_inside_mask(pred_contact, mask)
        metrics["gt_contact_inside_mask"] = _contact_inside_mask(gt_contact, mask)
        row = {
            "sample_id": sample_id,
            "dataset_index": int(idx),
            "instruction": sample["instruction"],
            "meta": sample["meta"],
            **metrics,
        }
        rows.append(row)
        _save_visualization(
            out_dir / f"sample_{sample_id:03d}.png",
            sample=sample,
            out=out,
            metrics=metrics,
            title=f"{model_name} | {sample['instruction']}",
        )

    summary = _summarize(rows)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": model_name, "checkpoint": str(spec["ckpt"]), "samples": rows, "summary": summary}, f, indent=2)
    return {"model": model_name, "num_samples": len(rows), "out_dir": str(out_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["part_action_mlp_real", "temporal_part_action_mlp_real"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=200)
    parser.add_argument("--max-demos-per-file", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default=None, help="Override Flan-T5 device, e.g. cpu or cuda")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "prototype" / "action_model_probe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    [run_model(model_name, args, device) for model_name in args.models]
    summaries = _collect_existing_summaries(args.out_dir)
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
