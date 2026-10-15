#!/usr/bin/env python3
"""Kandinsky 5.0 T2I Lite — CUDA CLI (one-shot or interactive)."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import env_settings as settings
from pipeline_utils import SUPPORTED_SIZES, load_pipe, run_pipe


def build_parser() -> argparse.ArgumentParser:
    sizes = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_SIZES)
    p = argparse.ArgumentParser(
        description="Kandinsky 5.0 T2I Lite (CUDA). Defaults from .env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"Supported sizes: {sizes}",
    )
    p.add_argument("-p", "--prompt", default=None, help="Required unless --interactive")
    p.add_argument("-n", "--negative-prompt", default="")
    p.add_argument("--model", default=settings.MODEL_PATH)
    p.add_argument("--width", type=int, default=settings.WIDTH)
    p.add_argument("--height", type=int, default=settings.HEIGHT)
    p.add_argument("--steps", type=int, default=settings.STEPS)
    p.add_argument("--guidance", type=float, default=settings.GUIDANCE)
    p.add_argument("--seed", type=int, default=settings.SEED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--offload",
        action=argparse.BooleanOptionalAction,
        default=settings.OFFLOAD,
        help="From .env OFFLOAD (default 0). Only for OOM",
    )
    p.add_argument(
        "--attention-slicing",
        action=argparse.BooleanOptionalAction,
        default=settings.ATTENTION_SLICING,
    )
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Load model once, then generate many prompts without reload",
    )
    return p


def save_image(image, out: Path | None) -> Path:
    if out is None:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"k5_{stamp}.png"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.interactive and not args.prompt:
        print("Need -p/--prompt or --interactive", file=sys.stderr)
        return 1

    print(f".env: OFFLOAD={int(args.offload)} MODEL_PATH={args.model}")

    try:
        pipe = load_pipe(
            args.model,
            offload=args.offload,
            attention_slicing=args.attention_slicing,
        )
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1

    if args.interactive:
        print("interactive mode: type prompt, empty line / quit to exit")
        seed = args.seed
        while True:
            try:
                prompt = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt or prompt.lower() in {"q", "quit", "exit"}:
                break
            t0 = time.perf_counter()
            try:
                image = run_pipe(
                    pipe,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=seed,
                )
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            path = save_image(image, None)
            print(f"saved: {path.resolve()} ({time.perf_counter() - t0:.1f}s)")
            seed += 1
        return 0

    t0 = time.perf_counter()
    try:
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
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    path = save_image(image, args.out)
    print(f"saved: {path.resolve()} ({time.perf_counter() - t0:.1f}s)")
    print("tip: use --interactive or uvicorn main:app so weights stay in VRAM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
