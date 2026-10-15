#!/usr/bin/env python3
"""Generate images with Kandinsky 5.0 T2I Lite on Mac (MPS) / CUDA / CPU."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str):
    import torch

    # MPS: float16 is more reliable than bfloat16
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kandinsky 5.0 Image Lite (T2I) local generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Download weights yourself, then point --model at the folder:\n"
            "  uv run huggingface-cli download "
            f"{DEFAULT_MODEL} --local-dir ./models/t2i-lite\n"
            "  uv run python generate.py --model ./models/t2i-lite "
            '-p "кот в красной шапке"'
        ),
    )
    p.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="Text prompt (RU/EN)",
    )
    p.add_argument(
        "-n",
        "--negative-prompt",
        default="",
        help="Negative prompt",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HF model id or local directory with Diffusers layout",
    )
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: outputs/<timestamp>.png)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    p.add_argument(
        "--offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU offload (needed on ~24GB Mac). Disable only if you have spare RAM/VRAM",
    )
    p.add_argument(
        "--attention-slicing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce peak memory during attention",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    from diffusers import Kandinsky5T2IPipeline

    device = pick_device() if args.device == "auto" else args.device
    dtype = pick_dtype(device)

    if device == "mps" and (args.width * args.height) > (1024 * 1024):
        print(
            "warn: >1MP on 24GB Mac often OOMs; try 768x768 first",
            file=sys.stderr,
        )

    print(f"device={device} dtype={dtype} model={args.model}")
    print("loading pipeline… (first run downloads/compiles, can take a while)")

    pipe = Kandinsky5T2IPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=Path(args.model).is_dir(),
    )

    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if args.offload:
        # Keep weights mostly in RAM, run layers on GPU when needed
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print("generating…")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )

    image = result.image[0]
    out = args.out
    if out is None:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"k5_{stamp}.png"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    image.save(out)
    print(f"saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
