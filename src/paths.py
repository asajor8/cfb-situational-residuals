"""Repo-relative paths. Override the root with CFB_ROOT if needed."""
import os
from pathlib import Path

ROOT = Path(os.environ["CFB_ROOT"]).resolve() if os.environ.get("CFB_ROOT") else Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
DERIVED = ROOT / "data" / "derived"
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"


def ensure_dirs():
    for p in (RAW, PROC, DERIVED, DOCS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
