#!/usr/bin/env python3
"""Kandinsky 5.0 T2I Lite — CUDA only (Windows + NVIDIA)."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
LOCAL_MODEL = Path("models/t2i-lite")

# Official T2I sizes from kandinskylab/kandinsky-5 test.py
SUPPORTED_SIZES = (
    (1024, 1024),
    (640, 1408),
    (1408, 640),
    (768, 1280),
    (1280, 768),
    (896, 1152),
    (1152, 896),
)


def build_parser() -> argparse.ArgumentParser:
    sizes = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_SIZES)
    p = argparse.ArgumentParser(
        description="Kandinsky 5.0 T2I Lite (CUDA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"Supported sizes: {sizes}",
    )
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-n", "--negative-prompt", default="")
    p.add_argument(
        "--model",
        default=str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else DEFAULT_MODEL,
        help="HF id or local dir (default: ./models/t2i-lite if present)",
    )
    # default = smallest supported pixel count
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=1408)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CPU offload (slower). Keep off on 16GB if VRAM fits",
    )
    p.add_argument(
        "--attention-slicing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Saves VRAM, slower. Keep off on 4070 if possible",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if (args.width, args.height) not in SUPPORTED_SIZES:
        print(
            f"Unsupported size {args.width}x{args.height}. "
            f"Use one of: {SUPPORTED_SIZES}",
            file=sys.stderr,
        )
        return 1

    import torch
    from diffusers import Kandinsky5T2IPipeline

    if not torch.cuda.is_available():
        print(
            "CUDA not available. Install torch cu124 and check nvidia-smi.",
            file=sys.stderr,
        )
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_path = Path(args.model)
    local_only = model_path.is_dir()
    if local_only and not (model_path / "model_index.json").exists():
        print(f"No model_index.json in {model_path}", file=sys.stderr)
        return 1

    print(
        f"gpu={torch.cuda.get_device_name(0)} dtype={dtype} "
        f"model={args.model} size={args.width}x{args.height} "
        f"steps={args.steps} offload={args.offload}"
    )

    pipe = Kandinsky5T2IPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=local_only,
    )

    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if args.offload:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe = pipe.to("cuda")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )

    elapsed = time.perf_counter() - t0
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
    print(f"saved: {out.resolve()} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
