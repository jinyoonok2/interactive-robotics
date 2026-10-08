#!/usr/bin/env python3
"""Export PartInstruct expert demos to MP4/GIF from stored agentview RGB."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

DEMO_ROOT = Path("/u/xna8aw/workspace/research_projects/interactive-robotics/datasets/PartInstruct/demos")
DEFAULT_OUT = Path("results/data/gt_demo_videos")

# One grasp + one touch per object, matching the PartGym objects in the 39-trial eval.
DEFAULT_DEMOS = (
    "bottle:demo_1",     # Grasp the bottle at its lid
    "bottle:demo_1000",  # Touch the bottle at its lid
    "mug:demo_1005",     # Grasp the mug at its handle
    "mug:demo_1007",     # Touch the mug at its right
    "pliers:demo_10",    # Grasp the pliers at its leg
    "pliers:demo_0",     # Touch the pliers at its top
    "scissors:demo_1",   # Grasp the scissors at its right
    "scissors:demo_0",   # Touch the scissors at its screw
)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "demo"


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(frame)
    y = 22
    for line in lines:
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return out


def _tint_mask(frame: np.ndarray, mask: np.ndarray, color=(40, 220, 80), alpha=0.35) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    on = mask > 0
    if not np.any(on):
        return frame
    out = frame.astype(np.float32)
    tint = np.zeros_like(out)
    tint[on] = color
    out[on] = (1.0 - alpha) * out[on] + alpha * tint[on]
    return np.clip(out, 0, 255).astype(np.uint8)


def export_demo(
    hdf5_path: Path,
    demo_key: str,
    out_dir: Path,
    fps: float,
    every_n: int,
    show_part_mask: bool,
) -> Path:
    import cv2
    import imageio.v2 as imageio

    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][demo_key]
        rgb = np.asarray(demo["obs"]["agentview_rgb"])
        mask = np.asarray(demo["obs"]["agentview_part_mask"]) if show_part_mask and "agentview_part_mask" in demo["obs"] else None
        instructions = [_decode(x) for x in demo["skill_instructions"]] if "skill_instructions" in demo else [""] * len(rgb)
        actions = np.asarray(demo["actions"])

    first_inst = instructions[0] if instructions else demo_key
    stem = f"{hdf5_path.stem}_{demo_key}_{_slug(first_inst)}"
    video_dir = out_dir / stem
    video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = video_dir / "rollout.mp4"
    gif_path = video_dir / "rollout.gif"

    height, width = rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    gif_frames = []
    for t, frame in enumerate(rgb):
        if frame.shape[-1] == 3:
            vis = frame
        else:
            vis = frame[..., :3]
        if mask is not None:
            vis = _tint_mask(vis, mask[t])
        gripper = float(actions[t, -1]) if actions.ndim == 2 and actions.shape[1] >= 7 else 0.0
        vis = _overlay(
            vis,
            [
                f"GT {hdf5_path.stem} {demo_key}  t={t}/{len(rgb)-1}",
                instructions[t][:60],
                f"gripper={gripper:.3f}",
            ],
        )
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if t % max(1, every_n) == 0:
            gif_frames.append(vis)
    writer.release()
    imageio.mimsave(gif_path, gif_frames, fps=max(1.0, float(fps) / max(1, every_n)))
    print(f"[gt-video] {len(rgb)} frames -> {mp4_path}")
    return mp4_path


def parse_demo_spec(spec: str) -> tuple[str, str]:
    obj, demo_key = spec.split(":", 1)
    return obj.strip(), demo_key.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", default=str(DEMO_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--demos", nargs="*", default=list(DEFAULT_DEMOS))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gif-every-n", type=int, default=2)
    parser.add_argument("--no-part-mask", action="store_true")
    args = parser.parse_args()

    root = Path(args.demo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.demos:
        obj, demo_key = parse_demo_spec(spec)
        hdf5_path = root / f"{obj}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
        export_demo(
            hdf5_path,
            demo_key,
            out_dir,
            fps=args.fps,
            every_n=args.gif_every_n,
            show_part_mask=not args.no_part_mask,
        )


if __name__ == "__main__":
    main()
