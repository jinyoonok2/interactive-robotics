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
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from models.part2action_model import Part2ActionModel


def build_eval_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    eval_paths = data_cfg.get("val_hdf5") or data_cfg["train_hdf5"]
    ds = PartInstructDataset(
        hdf5_paths=eval_paths,
        action_chunk=data_cfg.get("action_chunk", 8),
        sample_stride=max(1, data_cfg.get("sample_stride", 1) * 2),
        max_demos_per_file=data_cfg.get("max_demos_per_file"),
        n_obs_steps=data_cfg.get("n_obs_steps", 1),
    )
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
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


def offline_eval(model: Part2ActionModel, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    n = 0
    sums: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(dl, desc="offline-eval"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch["rgb"], batch["instruction"])

            if "heatmap_logits" in out:
                sums["iou"] = sums.get("iou", 0.0) + iou_score(out["heatmap_logits"], batch["part_mask"])
            if "contact_xy" in out:
                sums["contact_l1"] = sums.get("contact_l1", 0.0) + float(F.l1_loss(out["contact_xy"], batch["contact_xy"]).item())
            if "approach_dir" in out:
                cos = (out["approach_dir"] * batch["approach_dir"]).sum(dim=-1).clamp(-1.0, 1.0).mean()
                sums["approach_cos"] = sums.get("approach_cos", 0.0) + float(cos.item())
            if "action_chunk" in out:
                sums["action_l1"] = sums.get("action_l1", 0.0) + float(F.smooth_l1_loss(out["action_chunk"], batch["action_chunk"]).item())
            n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


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
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--use_partgym", action="store_true")
    p.add_argument("--rollout_episodes", type=int, default=10)
    p.add_argument("--rollout_splits", nargs="+", default=["test1"], help="PartInstruct test splits.")
    args = p.parse_args()

    cfg = resolve_paths(load_yaml(args.config), ROOT)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = select_device(cfg["train"].get("device", "cuda"))

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

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if state.get("trainable_only", False):
        unexpected_real = list(unexpected)
        if unexpected_real:
            print(f"[eval] unexpected keys (likely fine): {unexpected_real[:3]}{'...' if len(unexpected_real) > 3 else ''}")
        print(f"[eval] loaded trainable params; frozen backbones reloaded from cache.")

    dl = build_eval_loader(cfg)
    results: Dict[str, Any] = {"config": args.config, "ckpt": args.ckpt}
    results["offline"] = offline_eval(model, dl, device)
    print(f"[eval] offline: {results['offline']}")

    if args.use_partgym:
        results["partgym"] = maybe_partgym_rollout(
            model, device, n_episodes=args.rollout_episodes, splits=args.rollout_splits
        )

    out_dir = ensure_dir(Path(cfg["output_dir"]))
    save_json(out_dir / "eval.json", results)
    print(f"[eval] wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
