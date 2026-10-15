"""FastAPI: load Kandinsky once, then prompt → PNG (CUDA)."""

from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

import env_settings as settings
from pipeline_utils import SUPPORTED_SIZES, load_pipe, run_pipe

_pipe = None
_lock = threading.Lock()
_req_counter = 0
_req_counter_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pipe
    print(
        f".env: OFFLOAD={int(settings.OFFLOAD)} "
        f"ATTENTION_SLICING={int(settings.ATTENTION_SLICING)} "
        f"MODEL_PATH={settings.MODEL_PATH} "
        f"SIZE={settings.WIDTH}x{settings.HEIGHT}",
        flush=True,
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/generate":
        print(f"HTTP {request.method} {request.url.path} from {request.client}", flush=True)
    return await call_next(request)


@app.get("/health")
def health():
    return {
        "ok": _pipe is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "offload": settings.OFFLOAD,
        "attention_slicing": settings.ATTENTION_SLICING,
        "model": settings.MODEL_PATH,
        "default_size": f"{settings.WIDTH}x{settings.HEIGHT}",
        "sizes": [f"{w}x{h}" for w, h in SUPPORTED_SIZES],
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    global _req_counter
    if _pipe is None:
        raise HTTPException(503, "Model not loaded")
    if (req.width, req.height) not in SUPPORTED_SIZES:
        raise HTTPException(400, f"Unsupported size. Use: {SUPPORTED_SIZES}")

    with _req_counter_lock:
        _req_counter += 1
        req_id = _req_counter

    print(
        f"[req #{req_id}] queued prompt={req.prompt[:60]!r} "
        f"{req.width}x{req.height} steps={req.steps}",
        flush=True,
    )

    t0 = time.perf_counter()
    with _lock:
        print(f"[req #{req_id}] running…", flush=True)
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"k5_req{req_id}_{stamp}_{image.size[0]}x{image.size[1]}.png"
    image.save(path)

    print(
        f"[req #{req_id}] done {elapsed:.1f}s → {path} "
        f"({image.size[0]}x{image.size[1]})",
        flush=True,
    )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Gen-Seconds": f"{elapsed:.2f}",
            "X-Req-Id": str(req_id),
            "X-Image-Size": f"{image.size[0]}x{image.size[1]}",
        },
    )
