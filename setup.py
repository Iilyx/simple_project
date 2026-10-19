"""Create a local virtualenv and download FLUX.2 [klein] 4B onto this machine.

Weights stay in ./models and are not committed. Run this on the GPU computer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_DIR = ROOT / "models" / "FLUX.2-klein-4B"
DEFAULT_CUDA = "cu128"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def create_venv() -> Path:
    if sys.version_info < (3, 10):
        raise SystemExit("Need Python 3.10 or newer.")
    if not venv_python().exists():
        print(f"Creating {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True, clear=False)
    python = venv_python()
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    return python


def install_packages(python: Path, cuda: str) -> None:
    index = f"https://download.pytorch.org/whl/{cuda}"
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            index,
        ]
    )
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements.txt"),
        ]
    )


def download_model(python: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id={MODEL_ID!r},
    local_dir={str(MODEL_DIR)!r},
)
print("saved", {str(MODEL_DIR)!r})
"""
    run([str(python), "-c", code])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        default=DEFAULT_CUDA,
        help="PyTorch CUDA wheel channel, for example cu128 or cu126. Default: cu128.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Install the environment and leave the weights for a later run.",
    )
    args = parser.parse_args()

    python = create_venv()
    install_packages(python, args.cuda)
    if not args.skip_download:
        download_model(python)

    generate = ROOT / "generate.py"
    print()
    print("Ready. Generate:")
    print(f'  "{python}" "{generate}" --prompt "a red fox in an autumn forest"')
    print("Edit:")
    print(
        f'  "{python}" "{generate}" --image input.png '
        '--prompt "change the background to snow, keep the subject"'
    )


if __name__ == "__main__":
    main()
