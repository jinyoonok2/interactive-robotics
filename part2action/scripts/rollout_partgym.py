"""Closed-loop PartGym rollouts for action-capable part2action checkpoints.

Run this from the PartInstruct simulator environment, for example:

    conda run -n partinstruct python scripts/rollout_partgym.py \
        --config configs/part_action_mlp_real.yaml \
        --ckpt results/prototype/part_action_mlp_real/last.pt

The script intentionally targets the non-SAM PartGym environment
(`PartInstruct.PartGym.env.bullet_env`) so SAM2 is not required.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from _common import ROOT, ensure_dir, load_yaml, resolve_paths, select_device, set_seed
from models.part2action_model import Part2ActionModel


MODEL_DEFAULTS = {
    "mlp": {
        "config": ROOT / "configs" / "part_action_mlp_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "part_action_mlp_real" / "last.pt",
    },
    "temporal_mlp": {
        "config": ROOT / "configs" / "temporal_part_action_mlp_real.yaml",
        "ckpt": ROOT / "results" / "prototype" / "temporal_part_action_mlp_real" / "last.pt",
    },
}

DEFAULT_PARTINSTRUCT_ROOT = ROOT / "third_party" / "PartInstruct"


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
        action_head_type=cfg["model"].get("action_head_type", "mlp"),
        diffusion_steps=int(cfg["model"].get("diffusion_steps", 50)),
        temporal_encoder_type=cfg["model"].get("temporal_encoder_type", "none"),
        n_obs_steps=int(cfg["data"].get("n_obs_steps", 1)),
        temporal_layers=int(cfg["model"].get("temporal_layers", 1)),
        temporal_heads=int(cfg["model"].get("temporal_heads", 4)),
    )
    text_device = cfg["model"].get("text_device", "cpu")
    model.to(device)
    if text_device != str(device):
        model.text.model.to(text_device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _make_rgb_tensor(obs_history: deque[np.ndarray], n_obs_steps: int, device: torch.device) -> torch.Tensor:
    frames = list(obs_history)
    if n_obs_steps <= 1:
        rgb = torch.from_numpy(frames[-1]).float().unsqueeze(0) / 255.0
    else:
        while len(frames) < n_obs_steps:
            frames.insert(0, frames[0])
        rgb = torch.from_numpy(np.stack(frames[-n_obs_steps:], axis=0)).float().unsqueeze(0) / 255.0
    return rgb.to(device)


def _load_partgym_env(partinstruct_root: Path):
    if str(partinstruct_root) not in sys.path:
        sys.path.insert(0, str(partinstruct_root))
    from PartInstruct.PartGym.env.bullet_env import BulletEnv

    return BulletEnv


def _load_episode_keys(
    meta_path: Path,
    split: str,
    max_episodes: int,
    obj_classes: list[str] | None = None,
    task_types: list[str] | None = None,
) -> list[tuple[str, str]]:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    keys: list[tuple[str, str]] = []
    for obj_class, by_split in meta.items():
        if obj_classes and obj_class not in obj_classes:
            continue
        if split not in by_split:
            continue
        for task_type, episodes in by_split[split].items():
            if task_types and task_type not in task_types:
                continue
            if episodes:
                keys.append((obj_class, task_type))
            if len(keys) >= max_episodes:
                return keys
    return keys


def rollout_one(
    *,
    env_cls,
    model: Part2ActionModel,
    model_cfg: dict[str, Any],
    partgym_cfg: Any,
    config_path: Path,
    obj_class: str,
    task_type: str,
    split: str,
    device: torch.device,
    max_steps: int,
    execute_steps: int,
    record: bool,
    out_dir: Path,
) -> dict[str, Any]:
    env = env_cls(
        config=partgym_cfg,
        gui=False,
        record=record,
        evaluation=True,
        skill_mode=False,
        obj_class=obj_class,
        split=split,
        task_type=task_type,
        track_samples=False,
    )
    obs = env.reset()
    instruction = env.task_instruction
    n_obs_steps = int(model_cfg["data"].get("n_obs_steps", 1))
    history: deque[np.ndarray] = deque(maxlen=max(1, n_obs_steps))
    history.append(np.asarray(obs["agentview_rgb"], dtype=np.uint8))

    action_trace: list[list[float]] = []
    info: dict[str, Any] = {}
    done = False
    try:
        for _ in range(max_steps):
            rgb = _make_rgb_tensor(history, n_obs_steps, device)
            with torch.no_grad():
                out = model(rgb, [instruction])
            chunk = out["action_chunk"][0].detach().cpu().float().numpy()
            for action in chunk[:execute_steps]:
                obs, reward, done, info = env.step(action.astype(np.float32))
                action_trace.append(action.astype(float).tolist())
                history.append(np.asarray(obs["agentview_rgb"], dtype=np.uint8))
                if done:
                    break
            if done:
                break
    finally:
        if record and env.render_sequence_buffer:
            env.dump_buffers()
            video_path = out_dir / "videos" / f"{obj_class}_{task_type}_{split}" / "rollout.mp4"
            env.save_renders(str(video_path), video_only=True)
        env.close()

    return {
        "obj_class": obj_class,
        "task_type": task_type,
        "split": split,
        "instruction": instruction,
        "success": bool(done or info.get("Success", False)),
        "completion_rate": float(info.get("Completion Rate", 0.0)) if info else 0.0,
        "steps": int(info.get("Steps", len(action_trace))) if info else len(action_trace),
        "num_actions": len(action_trace),
        "actions": action_trace,
        "info": {k: v for k, v in info.items() if k not in {"Action"}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_DEFAULTS), default="mlp")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--partinstruct-root", type=Path, default=DEFAULT_PARTINSTRUCT_ROOT)
    parser.add_argument("--partgym-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test1")
    parser.add_argument("--obj-classes", nargs="+", default=None)
    parser.add_argument("--task-types", nargs="+", default=None)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "prototype" / "partgym_rollouts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model_key]
    config_path = (args.config or defaults["config"]).expanduser().resolve()
    ckpt_path = (args.ckpt or defaults["ckpt"]).expanduser().resolve()
    partinstruct_root = args.partinstruct_root.expanduser().resolve()
    partgym_config = (
        args.partgym_config.expanduser().resolve()
        if args.partgym_config
        else partinstruct_root / "PartInstruct" / "PartGym" / "config" / "config_oracle.yaml"
    )
    data_root = (args.data_root.expanduser().resolve() if args.data_root else partinstruct_root / "data")

    set_seed(args.seed)
    device = select_device(args.device)
    partgym_cfg = OmegaConf.load(partgym_config)
    partgym_cfg.data_root = str(data_root)
    if str(device) == "cpu":
        partgym_cfg.device = "cpu"

    meta_path = data_root / partgym_cfg.meta_path
    asset_path = data_root / partgym_cfg.urdf_robot
    if not meta_path.exists() or not asset_path.exists():
        raise FileNotFoundError(
            "PartGym data/assets are incomplete. Expected "
            f"{meta_path} and {asset_path}. Download upstream PartInstruct assets.zip "
            "into the PartInstruct data directory before running rollouts."
        )

    model_cfg = resolve_paths(load_yaml(config_path), ROOT)
    model = _load_model(model_cfg, ckpt_path, device)
    env_cls = _load_partgym_env(partinstruct_root)

    out_dir = ensure_dir(args.out_dir / args.model_key)
    episodes = _load_episode_keys(
        meta_path,
        args.split,
        args.num_episodes,
        obj_classes=args.obj_classes,
        task_types=args.task_types,
    )
    results = []
    for obj_class, task_type in episodes:
        result = rollout_one(
            env_cls=env_cls,
            model=model,
            model_cfg=model_cfg,
            partgym_cfg=partgym_cfg,
            config_path=partgym_config,
            obj_class=obj_class,
            task_type=task_type,
            split=args.split,
            device=device,
            max_steps=args.max_steps,
            execute_steps=args.execute_steps,
            record=args.record,
            out_dir=out_dir,
        )
        results.append(result)
        print(f"[rollout] {obj_class}/{task_type}: success={result['success']} completion={result['completion_rate']:.3f}")

    summary = {
        "model_key": args.model_key,
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "split": args.split,
        "num_episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "mean_completion_rate": float(np.mean([r["completion_rate"] for r in results])) if results else 0.0,
        "results": results,
    }
    suffix = "_".join(filter(None, [args.split, "-".join(args.obj_classes or []), "-".join(args.task_types or [])]))
    result_path = out_dir / f"rollout_results_{suffix or 'all'}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
