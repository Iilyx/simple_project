"""Shared CUDA pipeline loader for Kandinsky 5.0 T2I Lite."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import Kandinsky5T2IPipeline

DEFAULT_MODEL = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
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


def default_model_path() -> str:
    return str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else DEFAULT_MODEL


def setup_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Install torch cu124 and check nvidia-smi."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_pipe(model: str, *, offload: bool = False, attention_slicing: bool = False):
    setup_cuda()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    path = Path(model)
    local_only = path.is_dir()
    if local_only and not (path / "model_index.json").exists():
        raise FileNotFoundError(f"No model_index.json in {path}")

    print(
        f"loading weights once → gpu={torch.cuda.get_device_name(0)} "
        f"dtype={dtype} model={model} offload={offload}"
    )
    pipe = Kandinsky5T2IPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        local_files_only=local_only,
    )
    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if offload:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe = pipe.to("cuda")
    print("model ready in VRAM (keep this process alive for fast gens)")
    return pipe


def run_pipe(
    pipe,
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 640,
    height: int = 1408,
    steps: int = 16,
    guidance: float = 3.5,
    seed: int = 42,
):
    if (width, height) not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported size {width}x{height}. Use {SUPPORTED_SIZES}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
    return result.image[0]
