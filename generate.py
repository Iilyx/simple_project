#!/usr/bin/env python3
"""CLI: локальный Kandinsky, как HF-пример."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import env_settings as settings
from pipeline_utils import SUPPORTED_SIZES, load_pipe, run_pipe


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-n", "--negative-prompt", default="")
    p.add_argument("--model", default=settings.MODEL_PATH)
    p.add_argument("--width", type=int, default=settings.WIDTH)
    p.add_argument("--height", type=int, default=settings.HEIGHT)
    p.add_argument("--steps", type=int, default=settings.STEPS)
    p.add_argument("--guidance", type=float, default=settings.GUIDANCE)
    p.add_argument("--seed", type=int, default=settings.SEED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=settings.OFFLOAD)
    args = p.parse_args()

    if (args.width, args.height) not in SUPPORTED_SIZES:
        print(f"Unsupported size. Use: {SUPPORTED_SIZES}", file=sys.stderr)
        return 1

    pipe = load_pipe(args.model, offload=args.offload)
    t0 = time.perf_counter()
    image = run_pipe(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    out = args.out
    if out is None:
        Path("outputs").mkdir(exist_ok=True)
        out = Path("outputs") / f"k5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(out)
    print(f"saved {out.resolve()} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
