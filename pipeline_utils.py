"""Kandinsky 5.0 T2I Lite — как в HF-примере, но с локальными весами."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import Kandinsky5T2IPipeline

LOCAL_MODEL = Path("models/t2i-lite")

SUPPORTED_SIZES = (
    (1024, 1024),
    (640, 1408),
    (1408, 640),
    (768, 1280),
    (1280, 768),
    (896, 1152),
    (1152, 896),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def default_model_path() -> str:
    return os.getenv("MODEL_PATH", str(LOCAL_MODEL))


def load_pipe(model: str | None = None, *, offload: bool = False, attention_slicing: bool = False):
    """
    Как в карточке модели:
      pipe = Kandinsky5T2IPipeline.from_pretrained(...)
      pipe.to(device="cuda", dtype=torch.bfloat16)
    только model = локальная папка.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model = model or default_model_path()
    path = Path(model)
    if not path.is_dir() or not (path / "model_index.json").exists():
        raise FileNotFoundError(
            f"Local model not found: {path.resolve()} (need model_index.json)"
        )

    # как в примере HF
    dtype = torch.bfloat16

    _log(f"from_pretrained({path}) …")
    pipe = Kandinsky5T2IPipeline.from_pretrained(
        str(path),
        local_files_only=True,
    )
    _log(f'to(device="cuda", dtype={dtype}) …')
    pipe = pipe.to(device="cuda", dtype=dtype)

    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if offload:
        _log("WARN: offload=True — будет медленно")
        pipe.enable_model_cpu_offload(device="cuda")

    _log(f"READY gpu={torch.cuda.get_device_name(0)}")
    return pipe


def run_pipe(
    pipe,
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    guidance: float = 3.5,
    seed: int = 42,
):
    if (width, height) not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported size {width}x{height}. Use {SUPPORTED_SIZES}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    _log(f"generate {width}x{height} steps={steps}")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=1,
            generator=generator,
        )

    images = result.image
    if not isinstance(images, (list, tuple)):
        images = [images]
    return images[0]
