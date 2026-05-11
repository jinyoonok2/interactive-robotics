"""Create a tiny synthetic HDF5 file in PartInstruct's schema.

Useful for smoke-testing the loader, model, and training loops without
needing access to the gated SCAI-JHU/PartInstruct dataset (~83 GB).

Usage:
    python scripts/make_synthetic_demo.py \
        --out /tmp/partinstruct_synth/bottle.hdf5 \
        --n_demos 4 --steps 24 --img 96
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _make_demo(
    h5_demo: h5py.Group,
    steps: int,
    img: int,
    rng: np.random.RandomState,
    instruction: str,
) -> None:
    obs = h5_demo.create_group("obs")

    rgb = (rng.rand(steps, img, img, 3) * 255).astype(np.uint8)
    cy, cx = img // 2, img // 3
    rgb[:, cy - 6 : cy + 6, cx - 6 : cx + 6, :] = [200, 50, 50]
    obs.create_dataset("agentview_rgb", data=rgb)

    mask = np.zeros((steps, img, img, 1), dtype=np.uint8)
    mask[:, cy - 6 : cy + 6, cx - 6 : cx + 6, 0] = 1
    obs.create_dataset("agentview_part_mask", data=mask)

    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[: steps // 2, 0] = 0.01
    actions[: steps // 2, 2] = -0.005
    actions[steps // 2 :, 6] = 1.0
    obs.create_dataset("gripper_state", data=np.where(np.arange(steps)[:, None] >= steps // 2, 1.0, 0.0).astype(np.float32))
    obs.create_dataset("joint_states", data=rng.randn(steps, 7).astype(np.float32) * 0.01)

    h5_demo.create_dataset("actions", data=actions)
    text_arr = np.array([instruction.encode("utf-8")] * steps)
    h5_demo.create_dataset("skill_instructions", data=text_arr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output .hdf5 path")
    p.add_argument("--n_demos", type=int, default=4)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--img", type=int, default=96, help="Image side; 300 matches real schema")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    instructions = [
        "Grasp the handle of the bottle",
        "Lift the bottle by its lid",
        "Grasp the body of the kettle",
        "Tilt the kettle by its handle",
    ]

    with h5py.File(out, "w") as f:
        data = f.create_group("data")
        for i in range(args.n_demos):
            demo = data.create_group(f"demo_{i}")
            _make_demo(demo, steps=args.steps, img=args.img, rng=rng, instruction=instructions[i % len(instructions)])

    print(f"[synth] wrote {args.n_demos} demos x {args.steps} steps to {out}")


if __name__ == "__main__":
    main()
