"""Helpers for loading fixtures under ``_tests/data``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent / "data"
IMAGE_SUFFIXES = {".tif", ".tiff"}
TRACK_SUFFIXES = {".npy"}


def _require_npy(image_path: Path) -> Path:
    """Return sibling ``.npy`` path; raise if missing (run generate_numpy.py)."""
    npy_path = image_path.with_suffix(".npy")
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"Missing {npy_path.name} next to {image_path.name}. "
            "Generate with: python src/mmv_h4tracks/_tests/data/generate_numpy.py"
        )
    return npy_path


def load_image_zyx(path: Path) -> np.ndarray:
    """Load a segmentation/raw volume previously exported as ZYX ``.npy``."""
    return np.load(_require_npy(path))


def load_named_volumes(
    folder: Path,
    *,
    suffixes: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Load all image volumes in ``folder`` keyed by file stem.

    Discovers ``.tif`` / ``.tiff`` (or ``suffixes``) and loads the sibling ``.npy``.
    """
    suffixes = suffixes or IMAGE_SUFFIXES
    volumes: dict[str, np.ndarray] = {}
    if not folder.is_dir():
        return volumes
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        volumes[path.stem] = load_image_zyx(path)
    return volumes


def load_named_tracks(folder: Path) -> dict[str, np.ndarray]:
    """Load all ``.npy`` track arrays in ``folder`` keyed by stem."""
    tracks: dict[str, np.ndarray] = {}
    if not folder.is_dir():
        return tracks
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TRACK_SUFFIXES:
            continue
        # Skip accidental image dumps if both .tif and .npy exist with same stem
        # under tracks/ — tracks dir should only contain track tables.
        tracks[path.stem] = np.load(path)
    return tracks
