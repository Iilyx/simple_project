import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

MODEL_DIR = Path(__file__).resolve().parent / "models" / "FLUX.2-klein-4B"
pipe: Flux2KleinPipeline | None = None
busy = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = Field(default=4, ge=1)
    guidance: float = 1.0
    seed: int | None = None


def load_pipeline() -> Flux2KleinPipeline:
    if not (MODEL_DIR / "model_index.json").is_file():
        raise RuntimeError(f"Weights are missing in {MODEL_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU is not visible, torch {torch.__version__}")
    props = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16 if props.major >= 8 else torch.float16
    logger.info(
        "GPU %s, %.1f GB, capability %s.%s, dtype %s, torch %s",
        props.name,
        props.total_memory / 1024**3,
        props.major,
        props.minor,
        dtype,
        torch.__version__,
    )
    logger.info("Loading weights from %s", MODEL_DIR)
    loaded = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        local_files_only=True,
    )
    try:
        loaded.to("cuda")
    except torch.OutOfMemoryError:
        logger.info("Not enough GPU memory, enabling CPU offload")
        loaded.to("cpu")
        torch.cuda.empty_cache()
        loaded.enable_model_cpu_offload()
    logger.info("Weights are ready on %s", next(loaded.transformer.parameters()).device)
    return loaded


def render(
    prompt: str,
    image: Image.Image | list[Image.Image] | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    image_format: str,
) -> tuple[bytes, str]:
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
    }
    if image is not None:
        kwargs["image"] = image
    if seed is not None:
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    result = pipe(**kwargs).images[0]
    logger.info("Rendered in %.1f s", time.perf_counter() - started)
    buffer = io.BytesIO()
    result.save(buffer, format=image_format)
    return buffer.getvalue(), MEDIA_TYPES[image_format]


async def draw(**kwargs) -> Response:
    async with busy:
        payload, media_type = await asyncio.to_thread(render, **kwargs)
    return Response(content=payload, media_type=media_type)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipe
    pipe = load_pipeline()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(body: GenerateRequest) -> Response:
    logger.info("Generate: %s", body.prompt)
    return await draw(
        prompt=body.prompt,
        image=None,
        width=body.width,
        height=body.height,
        steps=body.steps,
        guidance=body.guidance,
        seed=body.seed,
        image_format="PNG",
    )


@app.post("/edit")
async def edit(
    prompt: str = Form(...),
    image: list[UploadFile] = File(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(4),
    guidance: float = Form(1.0),
    seed: int | None = Form(None),
) -> Response:
    logger.info("Edit: %s", prompt)
    opened: list[Image.Image] = []
    image_format = "PNG"
    for upload in image:
        try:
            source_image = Image.open(upload.file)
            image_format = image_format if opened else (source_image.format or "PNG")
            if image_format not in MEDIA_TYPES:
                image_format = "PNG"
            opened.append(source_image.convert("RGB"))
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Cannot read {upload.filename}") from error
    source: Image.Image | list[Image.Image] = opened[0] if len(opened) == 1 else opened
    return await draw(
        prompt=prompt,
        image=source,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        image_format=image_format,
    )


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    schema["openapi"] = "3.0.2"
    schema.pop("jsonSchemaDialect", None)

    def mark_files(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "string" and "contentMediaType" in node:
                node.pop("contentMediaType")
                node["format"] = "binary"
            for value in node.values():
                mark_files(value)
        elif isinstance(node, list):
            for item in node:
                mark_files(item)

    mark_files(schema)
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
