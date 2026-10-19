"""Generate or edit an image with the local FLUX.2 [klein] 4B weights.

The model must already be on disk. Inference does not contact Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "FLUX.2-klein-4B"
OUTPUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="What to draw, or how to edit.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="Reference image. Repeat the flag for several references.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Keep weights in RAM and move layers to the GPU as needed.",
    )
    return parser.parse_args()


def load_pipeline(offload: bool) -> Flux2KleinPipeline:
    if not (MODEL_DIR / "model_index.json").is_file():
        raise SystemExit(f"Model is missing in {MODEL_DIR}. Download it with the commands in the README.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script expects an NVIDIA GPU.")

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def load_images(paths: list[Path]) -> Image.Image | list[Image.Image] | None:
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    return images[0] if len(images) == 1 else images


def next_output(editing: bool) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "edit" if editing else "image"
    index = 0
    while True:
        suffix = "" if index == 0 else f"-{index}"
        path = OUTPUT_DIR / f"{stem}{suffix}.png"
        if not path.exists():
            return path
        index += 1


def main() -> None:
    args = parse_args()
    for path in args.image:
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")

    pipe = load_pipeline(args.offload)
    result = pipe(
        prompt=args.prompt,
        image=load_images(args.image),
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]

    output = args.output or next_output(bool(args.image))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(output)


if __name__ == "__main__":
    main()
