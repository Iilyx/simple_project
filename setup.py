"""Create a uv environment from requirements.txt and download FLUX.2 [klein] 4B.

Weights stay in ./models and are not committed. Run this on the GPU computer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
VENV = ROOT / ".venv"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_DIR = ROOT / "models" / "FLUX.2-klein-4B"
DEFAULT_CUDA = "cu128"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def uv_bin() -> str:
    found = shutil.which("uv")
    if found is None:
        raise SystemExit(
            "uv is not on PATH. Install it from https://docs.astral.sh/uv/getting-started/installation/ "
            "and run this again."
        )
    return found


def sync(uv: str, cuda: str) -> None:
    run([uv, "venv", str(VENV), "--python", "3.12"])
    cmd = [uv, "pip", "install", "-r", str(REQUIREMENTS), "--python", str(VENV)]
    if cuda != DEFAULT_CUDA:
        cmd.extend(["--index-url", f"https://download.pytorch.org/whl/{cuda}"])
    run(cmd)


def download_model(uv: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={MODEL_ID!r},
    local_dir={str(MODEL_DIR)!r},
)
print("saved", {str(MODEL_DIR)!r})
"""
    run([uv, "run", "--no-project", "--python", str(VENV), "python", "-c", code])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        default=DEFAULT_CUDA,
        help="Override the PyTorch index in requirements.txt, for example cu126.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Install the environment and leave the weights for a later run.",
    )
    args = parser.parse_args()

    uv = uv_bin()
    sync(uv, args.cuda)
    if not args.skip_download:
        download_model(uv)

    print()
    print("Ready. Generate:")
    print('  uv run --no-project --python .venv generate.py --prompt "a red fox in an autumn forest"')
    print("Edit:")
    print(
        "  uv run --no-project --python .venv generate.py --image input.png "
        '--prompt "change the background to snow, keep the subject"'
    )


if __name__ == "__main__":
    main()
