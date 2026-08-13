"""Optional helpers for the pipeline-graph take-home."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SHEETS_DIR = ROOT / "sheets"
SHEET_IDS = tuple(f"sheet_{i:02d}" for i in range(1, 6))


def sheet_path(sheet_id: str) -> Path:
    path = SHEETS_DIR / f"{sheet_id}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_sheet_bgr(sheet_id: str) -> np.ndarray:
    """Load sheet as BGR uint8 (OpenCV)."""
    im = cv2.imread(str(sheet_path(sheet_id)), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(sheet_path(sheet_id))
    return im


def save_pipeline_stats(stats: dict[str, Any], path: Path | str) -> None:
    """
    Write the required output shape:

        {
          "pipeline_1": {"length": 534.2, "nodes": 18},
          ...
        }
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    for sid in SHEET_IDS:
        p = SHEETS_DIR / f"{sid}.png"
        print(f"{sid}: exists={p.exists()}", end="")
        if p.exists():
            im = load_sheet_bgr(sid)
            h, w = im.shape[:2]
            print(f" size={w}x{h}")
        else:
            print()
