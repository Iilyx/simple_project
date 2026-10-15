"""Load .env once and expose typed settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


MODEL_PATH = os.getenv("MODEL_PATH", "models/t2i-lite")
OFFLOAD = _bool("OFFLOAD", "0")
ATTENTION_SLICING = _bool("ATTENTION_SLICING", "0")
WIDTH = _int("WIDTH", 1024)
HEIGHT = _int("HEIGHT", 1024)
STEPS = _int("STEPS", 50)
GUIDANCE = _float("GUIDANCE", 3.5)
SEED = _int("SEED", 42)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int("PORT", 8000)
