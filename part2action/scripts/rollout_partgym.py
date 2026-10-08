"""Closed-loop PartGym rollouts for action-capable part2action checkpoints.

Run this from the PartInstruct simulator environment, for example:

    conda run -n partinstruct python scripts/rollout_partgym.py \
        --config configs/architecture_update/hierarchical_world_geometry_30epoch_no_residual.yaml \
        --ckpt ../results/part2action/baseline/last.pt

The script intentionally targets the non-SAM PartGym environment
(`PartInstruct.PartGym.env.bullet_env`) so SAM2 is not required.

``--termination-mode oracle_partgym`` is an upper-bound diagnostic only.
PartGym predicates never enter policy tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from _common import ROOT, ensure_dir, load_yaml, resolve_paths, select_device, set_seed
from data.geometry import tcp_deltas_to_absolute_actions
from diagnostics.partgym_reach import (
    measure_point_part_prediction,
    measure_policy_target_alignment,
    measure_tcp_to_target_part,
)
from diagnostics.partgym_skill_oracle import PartGymSkillOracle
from diagnostics.phase_controller import should_advance_phase
from models.part2action_model import Part2ActionModel


MODEL_DEFAULTS = {
    "hierarchical": {
        "config": ROOT / "configs" / "architecture_update" / "hierarchical_world_geometry_30epoch_no_residual.yaml",
        "ckpt": ROOT.parent / "results" / "part2action" / "baseline" / "last.pt",
    },
}

DEFAULT_PARTINSTRUCT_ROOT = ROOT.parent / "PartInstruct"
TERMINATION_MODES = ("legacy", "oracle_partgym")


def configure_partgym_agentview_pcd(partgym_cfg: Any, full_cfg: Any) -> None:
    """Expose point-first PCD metadata expected by BulletEnv and this model."""
    if "agentview_pcd" not in partgym_cfg.shape_meta.obs:
        partgym_cfg.shape_meta.obs["agentview_pcd"] = full_cfg.shape_meta.obs[
            "agentview_pcd"
        ]
    shape = list(partgym_cfg.shape_meta.obs["agentview_pcd"].shape)
    if len(shape) == 2 and shape[0] == 3 and shape[1] != 3:
        partgym_cfg.shape_meta.obs["agentview_pcd"].shape = [
            int(shape[1]),
            int(shape[0]),
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
    )
    text_device = cfg["model"].get("text_device", "cpu")
    model.to(device)
    if text_device != str(device):
        model.text.model.to(text_device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def _chw_frame(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[-1] == 3 and image.shape[0] != 3:
        image = np.transpose(image, (2, 0, 1))
    return image


def _make_rgb_tensor(obs_history: deque[np.ndarray], n_obs_steps: int, device: torch.device) -> torch.Tensor:
    frames = [_chw_frame(frame) for frame in obs_history]
    if n_obs_steps <= 1:
        rgb = torch.from_numpy(frames[-1]).float().unsqueeze(0) / 255.0
    else:
        while len(frames) < n_obs_steps:
            frames.insert(0, frames[0])
        rgb = torch.from_numpy(np.stack(frames[-n_obs_steps:], axis=0)).float().unsqueeze(0) / 255.0
    return rgb.to(device)


def _model_observations(obs: dict[str, Any], device: torch.device) -> dict:
    kwargs = {}
    if "agentview_pcd" in obs:
        pcd = torch.from_numpy(np.asarray(obs["agentview_pcd"])).float()
        if pcd.ndim == 2:
            pcd = pcd.unsqueeze(0)
        if pcd.ndim == 3 and pcd.shape[1] in {3, 4} and pcd.shape[-1] not in {3, 4}:
            pcd = pcd.transpose(1, 2)
        kwargs["agentview_pcd"] = pcd.to(device)
    if "tcp_pose" in obs:
        kwargs["tcp_pose"] = torch.from_numpy(np.asarray(obs["tcp_pose"])).float().reshape(1, -1).to(device)
    if "gripper_state" in obs:
        kwargs["gripper_state"] = (
            torch.from_numpy(np.asarray(obs["gripper_state"])).float().reshape(1, -1).to(device)
        )
    return kwargs


def _as_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().reshape(-1)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _rollout_diagnostics(
    *,
    inference_step: int,
    obs: dict[str, Any],
    out: dict[str, Any],
    current_phase: torch.Tensor | None,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {"inference_step": int(inference_step)}
    tcp_pose = np.asarray(obs.get("tcp_pose", []), dtype=np.float32).reshape(-1)
    if tcp_pose.size >= 7:
        diagnostic["tcp_xyz"] = tcp_pose[4:7].astype(float).tolist()
        diagnostic["tcp_quat"] = tcp_pose[:4].astype(float).tolist()
    if "agentview_pcd" in obs:
        pcd = np.asarray(obs["agentview_pcd"], dtype=np.float32)
        points = pcd[..., :3].reshape(-1, 3) if pcd.size else np.empty((0, 3), dtype=np.float32)
        valid = np.isfinite(points).all(axis=-1) if len(points) else np.array([], dtype=bool)
        diagnostic["pcd_num_points"] = int(len(points))
        diagnostic["pcd_num_valid"] = int(valid.sum()) if valid.size else 0
        if valid.any():
            diagnostic["pcd_xyz_min"] = points[valid].min(axis=0).astype(float).tolist()
            diagnostic["pcd_xyz_max"] = points[valid].max(axis=0).astype(float).tolist()
    if current_phase is not None:
        diagnostic["current_phase"] = int(current_phase.item())
    if "predicted_phase" in out:
        diagnostic["predicted_phase"] = int(out["predicted_phase"][0].item())
    if "phase_logits" in out:
        diagnostic["phase_probs"] = _as_float_list(torch.softmax(out["phase_logits"][0], dim=-1))
    if "slot_valid_logits" in out:
        diagnostic["slot_valid_probs"] = _as_float_list(torch.sigmoid(out["slot_valid_logits"][0]))
    if "termination_logits" in out:
        diagnostic["termination_prob"] = float(torch.sigmoid(out["termination_logits"][0]).item())
    if "gate_logits" in out:
        gate = torch.sigmoid(out["gate_logits"][0].detach().float())
        diagnostic["gate_mean"] = float(gate.mean().item())
        diagnostic["gate_max"] = float(gate.max().item())
        diagnostic["gate_near_zero_frac"] = float((gate < 0.05).float().mean().item())
    if "contact_xyz" in out:
        diagnostic["contact_xyz"] = _as_float_list(out["contact_xyz"][0])
    if "contact_point_logits" in out:
        weights = torch.softmax(out["contact_point_logits"][0].detach().float(), dim=-1)
        diagnostic["contact_entropy"] = float((-(weights * (weights.clamp_min(1e-8).log())).sum()).item())
    if "baseline_action_chunk" in out:
        diagnostic["baseline_action0"] = _as_float_list(out["baseline_action_chunk"][0, 0])
    if "contact_xyz_residual" in out:
        residual = out["contact_xyz_residual"][0].detach().float()
        diagnostic["residual_max_abs"] = float(residual.abs().max().item())
        diagnostic["residual_action0"] = _as_float_list(residual[0])
    if "action_chunk" in out:
        diagnostic["action0"] = _as_float_list(out["action_chunk"][0, 0])
    return diagnostic


def _print_rollout_diagnostic(diagnostic: dict[str, Any]) -> None:
    fields = [
        f"step={diagnostic.get('inference_step')}",
        f"phase={diagnostic.get('current_phase')}",
        f"term={diagnostic.get('termination_prob')}",
        f"gate_max={diagnostic.get('gate_max')}",
        f"contact={diagnostic.get('contact_xyz')}",
        f"tcp={diagnostic.get('tcp_xyz')}",
        f"action0={diagnostic.get('action0')}",
    ]
    print("[diagnostic] " + " ".join(str(x) for x in fields if x.split("=", 1)[-1] not in {"None", "None"}))


def mp4_to_gif(
    mp4_path: Path,
    gif_path: Path | None = None,
    fps: float = 10.0,
    max_width: int = 480,
    every_n: int = 2,
) -> Path:
    import cv2
    import imageio.v2 as imageio

    gif_path = gif_path or mp4_path.with_suffix(".gif")
    capture = cv2.VideoCapture(str(mp4_path))
    frames = []
    frame_index = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if frame_index % max(1, int(every_n)) == 0:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=float(fps))
    return gif_path


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


def _legacy_skill_complete(out: dict[str, Any]) -> bool:
    if "termination_logits" not in out:
        return False
    return float(torch.sigmoid(out["termination_logits"][0])) >= 0.5


def _predicate_decision_dict(decision) -> dict[str, Any]:
    """Serialize privileged PartGym predicate state for diagnostics only."""
    return {
        "supported": bool(decision.supported),
        "complete": bool(decision.complete),
        "skill_name": decision.skill_name,
        "skill_index": int(decision.skill_index),
        "reason": decision.reason,
        "predicate_expected": dict(decision.predicate_expected),
        "predicate_results": dict(decision.predicate_results),
    }


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
    trial_index: int,
    episode_seed: int,
    device: torch.device,
    max_steps: int,
    execute_steps: int,
    record: bool,
    gif_fps: float,
    gif_width: int,
    gif_every_n: int,
    diagnostics: bool,
    predicate_diagnostics: bool,
    diagnostic_every: int,
    termination_mode: str,
    min_phase_cycles: int,
    out_dir: Path,
) -> dict[str, Any]:
    if termination_mode not in TERMINATION_MODES:
        raise ValueError(f"unsupported termination_mode={termination_mode}")
    set_seed(episode_seed)
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
    phase_trace: list[int] = []
    contact_trace: list[list[float]] = []
    gate_confidence_trace: list[float] = []
    gripper_command_trace: list[float] = []
    diagnostic_trace: list[dict[str, Any]] = []
    phase_event_trace: list[dict[str, Any]] = []
    initial_point_part_diagnostic = None
    info: dict[str, Any] = {}
    done = False
    current_phase = None
    cached_slots = cached_valid = None
    phase_steps = 0
    oracle = (
        PartGymSkillOracle()
        if termination_mode == "oracle_partgym" or predicate_diagnostics
        else None
    )
    video_path = None
    gif_path = None
    try:
        for inference_step in range(max_steps):
            rgb = _make_rgb_tensor(history, n_obs_steps, device)
            kwargs = _model_observations(obs, device)
            model_instruction = [instruction]
            if cached_slots is not None:
                kwargs.update(
                    plan_slots_override=cached_slots,
                    slot_valid_logits_override=cached_valid,
                    current_phase=current_phase,
                    force_current_phase=True,
                )
                model_instruction = None
            with torch.no_grad():
                out = model(rgb, model_instruction, **kwargs)
            if cached_slots is None:
                cached_slots = out["plan_slots"].detach()
                cached_valid = out["slot_valid_logits"].detach()
                current_phase = out["predicted_phase"].detach()
            phase_idx = int(current_phase.item())
            phase_trace.append(phase_idx)
            if (
                inference_step == 0
                and "point_part_logits" in out
                and "agentview_pcd" in obs
            ):
                initial_point_part_diagnostic = measure_point_part_prediction(
                    env,
                    obs["agentview_pcd"],
                    out["point_part_logits"][0].detach().cpu().float().numpy(),
                    phase_idx,
                )
            if diagnostics:
                diagnostic = _rollout_diagnostics(
                    inference_step=inference_step,
                    obs=obs,
                    out=out,
                    current_phase=current_phase,
                )
                diagnostic_trace.append(diagnostic)
                if inference_step % max(1, diagnostic_every) == 0:
                    _print_rollout_diagnostic(diagnostic)
            if "contact_xyz" in out:
                contact_trace.append(_as_float_list(out["contact_xyz"][0]))
            if "gate_logits" in out:
                gate_confidence_trace.append(
                    float(torch.sigmoid(out["gate_logits"][0]).max().item())
                )
            chunk = out["action_chunk"]
            if model.use_action_deltas:
                if "tcp_pose" not in kwargs:
                    raise RuntimeError("use_action_deltas requires tcp_pose observations")
                chunk = tcp_deltas_to_absolute_actions(chunk, kwargs["tcp_pose"])
            policy_alignment = None
            if (
                predicate_diagnostics
                and "contact_xyz" in out
                and "tcp_pose" in obs
            ):
                policy_alignment = measure_policy_target_alignment(
                    env,
                    obs["tcp_pose"],
                    phase_idx,
                    predicted_contact_xyz=(
                        out["contact_xyz"][0].detach().cpu().float().numpy()
                    ),
                    action_target_xyz=(
                        chunk[0, 0, :3].detach().cpu().float().numpy()
                    ),
                )
            chunk = chunk[0].detach().cpu().float().numpy()
            for action in chunk[:execute_steps]:
                obs, reward, done, info = env.step(action.astype(np.float32))
                action_trace.append(action.astype(float).tolist())
                gripper_command_trace.append(float(action[6]))
                history.append(np.asarray(obs["agentview_rgb"], dtype=np.uint8))
                if done:
                    break
            phase_steps += 1
            valid_count = max(1, int((cached_valid[0].sigmoid() >= 0.5).sum().item()))
            oracle_decision = None
            predicate_decision = None
            reach_diagnostic = None
            if predicate_diagnostics:
                assert oracle is not None
                reach_diagnostic = measure_tcp_to_target_part(
                    env, obs.get("tcp_pose", []), phase_idx
                )
                predicate_decision = oracle.is_skill_complete(
                    env,
                    phase_idx,
                    exhaustive=True,
                    preserve_sim_state=True,
                )
            if termination_mode == "legacy":
                skill_complete = _legacy_skill_complete(out)
                complete_source = "legacy_termination_head"
            else:
                assert oracle is not None
                oracle_decision = (
                    predicate_decision
                    if predicate_decision is not None
                    else oracle.is_skill_complete(env, phase_idx)
                )
                skill_complete = bool(oracle_decision.complete and oracle_decision.supported)
                complete_source = "oracle_partgym"
            advance = should_advance_phase(
                skill_complete=skill_complete,
                phase_steps=phase_steps,
                min_phase_cycles=min_phase_cycles,
                current_phase=phase_idx,
                valid_skill_count=valid_count,
            )
            event = {
                "inference_step": inference_step,
                "phase": phase_idx,
                "termination_mode": termination_mode,
                "complete_source": complete_source,
                "skill_complete": bool(skill_complete),
                "phase_steps": phase_steps,
                "valid_skill_count": valid_count,
                "advanced": bool(advance),
                "legacy_termination_prob": (
                    float(torch.sigmoid(out["termination_logits"][0]).item())
                    if "termination_logits" in out
                    else None
                ),
            }
            if oracle_decision is not None:
                event["oracle"] = _predicate_decision_dict(oracle_decision)
            if predicate_decision is not None:
                event["predicate_diagnostic"] = _predicate_decision_dict(
                    predicate_decision
                )
            if reach_diagnostic is not None:
                event["reach_diagnostic"] = reach_diagnostic
            if policy_alignment is not None:
                event["policy_target_alignment"] = policy_alignment
            phase_event_trace.append(event)
            if advance:
                current_phase = current_phase + 1
                phase_steps = 0
            if done:
                break
    finally:
        if record and env.render_sequence_buffer:
            env.dump_buffers()
            trial_name = f"{obj_class}_{task_type}_{split}_trial{trial_index:02d}_seed{episode_seed}"
            video_path = out_dir / "videos" / trial_name / "rollout.mp4"
            env.save_renders(str(video_path), video_only=True)
            try:
                gif_path = mp4_to_gif(
                    video_path,
                    fps=gif_fps,
                    max_width=gif_width,
                    every_n=gif_every_n,
                )
            except Exception as exc:  # pragma: no cover - optional media path
                print(f"[rollout] gif conversion failed: {exc}")
        env.close()

    return {
        "obj_class": obj_class,
        "task_type": task_type,
        "split": split,
        "trial_index": trial_index,
        "seed": episode_seed,
        "instruction": instruction,
        "success": bool(done or info.get("Success", False)),
        "completion_rate": float(info.get("Completion Rate", 0.0)) if info else 0.0,
        "steps": int(info.get("Steps", len(action_trace))) if info else len(action_trace),
        "num_actions": len(action_trace),
        "termination_mode": termination_mode,
        "predicate_diagnostics": bool(predicate_diagnostics),
        "predicted_contact_xyz": contact_trace,
        "gate_max_confidence": gate_confidence_trace,
        "gripper_commands": gripper_command_trace,
        "actions": action_trace,
        "phase_trace": phase_trace,
        "phase_events": phase_event_trace,
        "initial_point_part_diagnostic": initial_point_part_diagnostic,
        "diagnostics": diagnostic_trace if diagnostics else [],
        "rollout_video": str(video_path) if video_path else None,
        "rollout_gif": str(gif_path) if gif_path else None,
        "info": {k: v for k, v in info.items() if k not in {"Action"}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_DEFAULTS), default="hierarchical")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--partinstruct-root", type=Path, default=DEFAULT_PARTINSTRUCT_ROOT)
    parser.add_argument("--partgym-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="test1")
    parser.add_argument("--obj-classes", nargs="+", default=None)
    parser.add_argument("--task-types", nargs="+", default=None)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2,
        help="Maximum number of unique object/task combinations (legacy name).",
    )
    parser.add_argument(
        "--trials-per-task",
        type=int,
        default=1,
        help="Number of independently seeded rollouts for each object/task combination.",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--gif-every-n", type=int, default=2)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print and save detailed per-inference model and geometry traces.",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=10,
        help="Print one diagnostic line every N inference steps (all steps are saved).",
    )
    parser.add_argument(
        "--predicate-diagnostics",
        action="store_true",
        help=(
            "Log exhaustive privileged PartGym predicates after every executed "
            "action chunk. Diagnostic only: never changes policy inputs or "
            "legacy phase transitions."
        ),
    )
    parser.add_argument(
        "--termination-mode",
        choices=TERMINATION_MODES,
        default=None,
        help="legacy uses the learned termination head; oracle_partgym is eval-only.",
    )
    parser.add_argument("--min-phase-cycles", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT.parent / "results" / "part2action" / "evaluations" / "manual_partgym_rollouts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.diagnostic_every < 1:
        parser.error("--diagnostic-every must be at least 1")
    if args.trials_per_task < 1:
        parser.error("--trials-per-task must be at least 1")

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
    model_cfg = resolve_paths(load_yaml(config_path), ROOT)
    rollout_cfg = model_cfg.get("rollout", {}) or {}
    termination_mode = args.termination_mode or str(rollout_cfg.get("termination_mode", "legacy"))
    if termination_mode not in TERMINATION_MODES:
        raise ValueError(f"unsupported termination_mode={termination_mode}")
    min_phase_cycles = (
        args.min_phase_cycles
        if args.min_phase_cycles is not None
        else int(rollout_cfg.get("min_phase_cycles", 2))
    )
    full_cfg = OmegaConf.load(
        partinstruct_root / "PartInstruct" / "PartGym" / "config" / "config.yaml"
    )
    configure_partgym_agentview_pcd(partgym_cfg, full_cfg)
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
    combination_index = 0
    for obj_class, task_type in episodes:
        for trial_index in range(args.trials_per_task):
            episode_seed = int(args.seed + combination_index * 100 + trial_index)
            result = rollout_one(
                env_cls=env_cls,
                model=model,
                model_cfg=model_cfg,
                partgym_cfg=partgym_cfg,
                config_path=partgym_config,
                obj_class=obj_class,
                task_type=task_type,
                split=args.split,
                trial_index=trial_index,
                episode_seed=episode_seed,
                device=device,
                max_steps=args.max_steps,
                execute_steps=args.execute_steps,
                record=args.record,
                gif_fps=args.gif_fps,
                gif_width=args.gif_width,
                gif_every_n=args.gif_every_n,
                diagnostics=args.diagnostics,
                predicate_diagnostics=args.predicate_diagnostics,
                diagnostic_every=args.diagnostic_every,
                termination_mode=termination_mode,
                min_phase_cycles=min_phase_cycles,
                out_dir=out_dir,
            )
            results.append(result)
            print(
                f"[rollout] {obj_class}/{task_type} trial={trial_index} seed={episode_seed}: "
                f"success={result['success']} completion={result['completion_rate']:.3f}"
            )
        combination_index += 1

    per_task: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = f"{result['obj_class']}_{result['task_type']}"
        per_task.setdefault(key, []).append(result)

    summary = {
        "model_key": args.model_key,
        "point_cloud_sampling_device": os.environ.get("PARTGYM_FPS_DEVICE", "cpu"),
        "config": str(config_path),
        "ckpt": str(ckpt_path),
        "split": args.split,
        "termination_mode": termination_mode,
        "predicate_diagnostics": bool(args.predicate_diagnostics),
        "min_phase_cycles": min_phase_cycles,
        "use_skill_gate": bool(model_cfg.get("model", {}).get("use_skill_gate", False)),
        "num_episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "mean_completion_rate": float(np.mean([r["completion_rate"] for r in results])) if results else 0.0,
        "per_task": {
            key: {
                "n": len(task_results),
                "success_rate": float(np.mean([r["success"] for r in task_results])),
                "mean_completion_rate": float(np.mean([r["completion_rate"] for r in task_results])),
            }
            for key, task_results in per_task.items()
        },
        "results": results,
    }
    suffix = "_".join(
        filter(
            None,
            [
                args.split,
                "-".join(args.obj_classes or []),
                "-".join(args.task_types or []),
                termination_mode,
            ],
        )
    )
    result_path = out_dir / f"rollout_results_{suffix or 'all'}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
