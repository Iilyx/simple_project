import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

parser = argparse.ArgumentParser()
parser.add_argument("prompt")
prompt = parser.parse_args().prompt

device = "cuda"
dtype = torch.bfloat16
model_dir = Path(__file__).resolve().parent / "models" / "FLUX.2-klein-4B"

pipe = Flux2KleinPipeline.from_pretrained(model_dir, dtype=dtype, local_files_only=True)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]
image.save("flux-klein.png")
