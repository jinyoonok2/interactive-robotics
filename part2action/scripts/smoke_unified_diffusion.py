"""One real-data optimization step and sampling check for both diffusion variants."""
import argparse
import gc
from pathlib import Path

import torch

from _common import ROOT, load_yaml, resolve_paths, save_json, set_seed
from train import (build_dataset, build_dataloader, build_model, compute_losses,
                   curriculum_state, forward_kwargs, freeze_prefixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Check one supplied diffusion configuration instead of the original pair")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "Run through a Slurm GPU allocation"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    variants = [(args.config.stem, args.config)] if args.config else [
        (variant, ROOT / "configs" / "architecture_update" / f"hierarchical_world_geometry_30epoch_unified_diffusion_{variant}.yaml")
        for variant in ["control", "contact"]]
    for variant, source in variants:
        cfg = resolve_paths(load_yaml(source), ROOT)
        set_seed(42)
        # Smoke-only limits; full training configs remain unchanged.
        cfg["data"].update(max_demos_per_file=1, batch_size=2, num_workers=0)
        dataset = build_dataset(cfg["data"])
        batch = next(iter(build_dataloader(dataset, cfg["data"])))
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model = build_model(cfg).cuda()
        freeze_prefixes(model, cfg["train"]["freeze_prefixes"])
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["train"]["lr"])
        model.train()
        _, probability, weights = curriculum_state(0, cfg)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["rgb"], batch["task_instruction"], target_action=batch["action_chunk"],
                           **forward_kwargs(batch, True, probability, True))
            total, parts = compute_losses(output, batch, weights, set(cfg["heads"]))
        assert torch.isfinite(total), "Nonfinite real-data training loss"
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        model.eval()
        # Deployment inputs only; no action targets, part masks, or contact labels.
        inputs = {k: batch[k] for k in ["tcp_pose", "agentview_pcd", "gripper_state"] if k in batch}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = model(batch["rgb"], batch["task_instruction"], **inputs)
        action = sampled["action_chunk"]
        assert action.shape == (2, 8, 7) and torch.isfinite(action).all()
        assert model.action_head.normalize(action).abs().max() <= cfg["model"]["diffusion_clip_sample"] + 1e-4
        if not cfg["model"].get("contact_conditioning_enabled", True):
            assert torch.count_nonzero(sampled["contact_conditioning_features"]) == 0
        results[variant] = {"loss": float(total), "loss_parts": {k: float(v) for k, v in parts.items()},
                            "denoiser_type": model.action_head.denoiser_type,
                            "action_head_parameters": sum(p.numel() for p in model.action_head.parameters()),
                            "action_shape": list(action.shape), "finite_gradients": True,
                            "finite_samples": True, "device": torch.cuda.get_device_name(0),
                            "normalized_target_abs_max": float(model.action_head.normalize(batch["action_chunk"]).abs().max()),
                            "sample_abs_max": float(action.abs().max()), "config": str(source)}
        save_json(args.out_dir / "verification.json", results)
        print(f"PASSED {variant}: loss={float(total):.6f}, sample shape={list(action.shape)}", flush=True)
        del model, optimizer, output, sampled, total, batch, dataset
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
