"""FastAPI: load Kandinsky once, then prompt → PNG (CUDA)."""

from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pipeline_utils import (
    SUPPORTED_SIZES,
    default_model_path,
    load_pipe,
    run_pipe,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", default_model_path()))
OFFLOAD = os.getenv("OFFLOAD", "0") == "1"

_pipe = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pipe
    _pipe = load_pipe(str(MODEL_PATH), offload=OFFLOAD, attention_slicing=False)
    yield


app = FastAPI(title="Kandinsky T2I", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = 640
    height: int = 1408
    steps: int = Field(16, ge=1, le=100)
    guidance: float = 3.5
    seed: int = 42


@app.get("/health")
def health():
    return {
        "ok": _pipe is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "offload": OFFLOAD,
        "model": str(MODEL_PATH),
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
