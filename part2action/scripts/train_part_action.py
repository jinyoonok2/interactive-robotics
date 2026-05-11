"""Part-action MLP trainer (heatmap + contact + approach + action). Thin wrapper."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "part_action_mlp_synth.yaml"))
    p.add_argument("--override-out", default=None)
    args, extra = p.parse_known_args()

    sys.argv = ["train.py", "--config", args.config]
    if args.override_out:
        sys.argv += ["--override-out", args.override_out]
    sys.argv += extra
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")


if __name__ == "__main__":
    main()
