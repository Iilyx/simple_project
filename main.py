"""FastAPI: локальный Kandinsky, как HF-пример → PNG."""

from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

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
    _pipe = load_pipe(settings.MODEL_PATH, offload=settings.OFFLOAD)
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
        "model": settings.MODEL_PATH,
        "size": f"{settings.WIDTH}x{settings.HEIGHT}",
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

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"k5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(path)
    print(f"saved {path} ({elapsed:.1f}s)", flush=True)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Gen-Seconds": f"{elapsed:.2f}"},
    )
