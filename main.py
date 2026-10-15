"""FastAPI: load Kandinsky once, then prompt → PNG (CUDA)."""

from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import env_settings as settings
from pipeline_utils import SUPPORTED_SIZES, load_pipe, run_pipe

_pipe = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pipe
    print(
        f".env: OFFLOAD={int(settings.OFFLOAD)} "
        f"ATTENTION_SLICING={int(settings.ATTENTION_SLICING)} "
        f"MODEL_PATH={settings.MODEL_PATH}"
    )
    _pipe = load_pipe(
        settings.MODEL_PATH,
        offload=settings.OFFLOAD,
        attention_slicing=settings.ATTENTION_SLICING,
    )
    yield


app = FastAPI(title="Kandinsky T2I", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = settings.WIDTH
    height: int = settings.HEIGHT
    steps: int = Field(settings.STEPS, ge=1, le=100)
    guidance: float = settings.GUIDANCE
    seed: int = settings.SEED


@app.get("/health")
def health():
    return {
        "ok": _pipe is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "offload": settings.OFFLOAD,
        "attention_slicing": settings.ATTENTION_SLICING,
        "model": settings.MODEL_PATH,
        "sizes": [f"{w}x{h}" for w, h in SUPPORTED_SIZES],
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(503, "Model not loaded")
    if (req.width, req.height) not in SUPPORTED_SIZES:
        raise HTTPException(400, f"Unsupported size. Use: {SUPPORTED_SIZES}")

    t0 = time.perf_counter()
    with _lock:
        image = run_pipe(
            _pipe,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
        )
    elapsed = time.perf_counter() - t0
    print(f"generate {req.width}x{req.height} steps={req.steps} → {elapsed:.1f}s")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Gen-Seconds": f"{elapsed:.2f}"},
    )
