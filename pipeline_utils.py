"""Shared CUDA pipeline loader for Kandinsky 5.0 T2I Lite."""

from __future__ import annotations

import os
import time
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


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _vram() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f}/{reserved:.1f}/{total:.1f} GB (alloc/reserved/total)"


def load_pipe(model: str, *, offload: bool = False, attention_slicing: bool = False):
    setup_cuda()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    path = Path(model)
    local_only = path.is_dir()
    if local_only and not (path / "model_index.json").exists():
        raise FileNotFoundError(f"No model_index.json in {path}")

    if local_only:
        # не ходить в сеть за токенайзерами — иначе может «висеть»
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _log(
        f"[1/4] load from disk → gpu={torch.cuda.get_device_name(0)} "
        f"dtype={dtype} model={model} offload={offload}"
    )
    _log("     (Qwen + DiT большие — 2–10+ мин без новых строк это нормально)")
    t0 = time.perf_counter()
    pipe = Kandinsky5T2IPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )
    _log(f"[2/4] from_pretrained done in {time.perf_counter() - t0:.1f}s  vram={_vram()}")

    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if offload:
        _log("[3/4] enable_model_cpu_offload…")
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        _log("[3/4] moving all modules to CUDA (может долго молчать)…")
        t1 = time.perf_counter()
        pipe = pipe.to("cuda")
        torch.cuda.synchronize()
        _log(f"     .to(cuda) done in {time.perf_counter() - t1:.1f}s  vram={_vram()}")

    warmup = os.getenv("WARMUP", "1").strip() not in {"0", "false", "False"}
    if warmup and not offload:
        _log("[4/4] warmup generate (первый прогон всегда самый долгий)…")
        t2 = time.perf_counter()
        with torch.inference_mode():
            _ = pipe(
                prompt="warmup",
                negative_prompt="",
                height=640,
                width=1408,
                num_inference_steps=2,
                guidance_scale=1.0,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _log(f"     warmup done in {time.perf_counter() - t2:.1f}s  vram={_vram()}")
    else:
        _log("[4/4] warmup skipped")

    _log("READY — можно генерировать, веса в памяти")
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

    _log(f"generate start {width}x{height} steps={steps} …")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def _on_step(pipe_, step_index, timestep, callback_kwargs):
        # step_index с 0; печатаем прогресс
        if step_index == 0 or (step_index + 1) % max(1, steps // 4) == 0 or (
            step_index + 1
        ) == steps:
            _log(f"  step {step_index + 1}/{steps}")
        return callback_kwargs

    t0 = time.perf_counter()
    with torch.inference_mode():
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=_on_step,
            )
        except TypeError:
            # старые diffusers без callback
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
    torch.cuda.synchronize()
    _log(f"generate done in {time.perf_counter() - t0:.1f}s")
    return result.image[0]
