"""FastAPI: prompt → PNG (Kandinsky 5.0 T2I Lite, CUDA / Windows)."""

from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from diffusers import Kandinsky5T2IPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/t2i-lite"))
OFFLOAD = os.getenv("OFFLOAD", "0") == "1"

_pipe = None
_lock = threading.Lock()


def load_pipeline():
    global _pipe
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Install torch with CUDA "
            "(cu124) and check nvidia-smi."
        )
    if not MODEL_PATH.is_dir() or not (MODEL_PATH / "model_index.json").exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            "Download to models/t2i-lite first."
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = Kandinsky5T2IPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        local_files_only=True,
    )
    if OFFLOAD:
        pipe.enable_model_cpu_offload(device="cuda")
    else:
        pipe = pipe.to("cuda")
    _pipe = pipe
    print(f"loaded model={MODEL_PATH} device=cuda offload={OFFLOAD}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_pipeline()
    yield


app = FastAPI(title="Kandinsky T2I", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = Field(768, ge=256, le=1280)
    height: int = Field(768, ge=256, le=1280)
    steps: int = Field(28, ge=1, le=100)
    guidance: float = Field(3.5, ge=0.0, le=20.0)
    seed: int = 42


@app.get("/health")
def health():
    return {
        "ok": _pipe is not None,
        "device": "cuda",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": str(MODEL_PATH),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(503, "Model not loaded")

    generator = torch.Generator(device="cpu").manual_seed(req.seed)

    with _lock:
        result = _pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=generator,
        )
        image = result.image[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
