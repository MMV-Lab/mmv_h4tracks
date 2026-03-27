"""
Export raw / segmentation pairs for Cellpose training (per-frame TIFF files).

``export_cellpose_training_pairs`` writes TIFFs under an OS temp directory.
``train_cellpose`` runs the Cellpose CLI on that directory; weights appear under
``<export_dir>/models/`` for pickup by the plugin worker.
"""

from __future__ import annotations

import importlib
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._grabber import grab_layer

# Cellpose CLI writes trained weights under ``models/`` with names like ``cellpose_<epoch>.<step>`` (optional extra extension).
_CELLPOSE_CLI_WEIGHTS_RE = re.compile(r"^cellpose_\d+\.\d+(?:\.[^.]+)?$")

# Deterministic temp export dirs (scanned on plugin start).
MMV_TRAIN_DIR_PREFIX = "mmv_h4tracks_train_"
_MASK_STEM_FRAME_RE = re.compile(r"^(.+)_frame_(\d{5})_masks$")

# --- Export layout (Cellpose CLI-style: ``image.tif`` + ``image_masks.tif``, ``--mask_filter _masks``)

CELLPOSE_TRAINING_MASK_STEM_SUFFIX = "_masks"


def _cellpose_training_mask_path(image_path: Path) -> Path:
    """Path to the mask file paired with ``image_path`` (``foo.tif`` → ``foo_masks.tif``)."""
    return image_path.parent / (
        f"{image_path.stem}{CELLPOSE_TRAINING_MASK_STEM_SUFFIX}{image_path.suffix}"
    )


def _is_cellpose_training_mask_file(path: Path) -> bool:
    return path.stem.endswith(CELLPOSE_TRAINING_MASK_STEM_SUFFIX)


def _safe_rmtree(path: Path | str) -> None:
    """Remove a directory tree if it exists; ignore errors (best-effort cleanup)."""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def parse_use_frames(text: str) -> list[int] | None:
    """
    Parse comma-separated frame indices. Whitespace around tokens is ignored.
    Returns sorted unique indices, or None if the string is empty or invalid.
    """
    stripped = text.strip()
    if not stripped:
        return None
    parts = stripped.split(",")
    indices: list[int] = []
    for part in parts:
        token = part.strip()
        if not token:
            return None
        try:
            indices.append(int(token))
        except ValueError:
            return None
    if not indices:
        return None
    return sorted(set(indices))


def _get_array_from_layer(layer) -> np.ndarray:
    """Load layer volume as squeezed numpy array (2D single frame or 3D time series)."""
    if isinstance(layer.data, (list, tuple)) and len(layer.data) > 0:
        data = layer.data[0]
    elif isinstance(layer.data, np.ndarray):
        data = layer.data
    else:
        try:
            data = layer.data[0] if hasattr(layer.data, "__getitem__") else layer.data
        except (TypeError, IndexError, AttributeError):
            data = layer.data
    data = np.asarray(data)
    data_squeezed = np.squeeze(data)
    if data_squeezed.ndim not in (2, 3):
        raise ValueError(
            "Image and segmentation must be 2D or a stack of 2D frames (3D after "
            f"removing singleton dimensions). Got shape {data_squeezed.shape}."
        )
    return data_squeezed


def _n_frames(vol: np.ndarray) -> int:
    return 1 if vol.ndim == 2 else vol.shape[0]


def _frame_2d(vol: np.ndarray, frame_idx: int) -> np.ndarray:
    if vol.ndim == 2:
        if frame_idx != 0:
            raise ValueError(
                f"Frame index {frame_idx} is out of range for a single image "
                "(valid indices: 0 only)."
            )
        return vol
    n = vol.shape[0]
    if frame_idx < 0 or frame_idx >= n:
        raise ValueError(
            f"Frame index {frame_idx} is out of range (valid: 0 to {n - 1})."
        )
    return vol[frame_idx]


def _validate_frame_pairs(raw: np.ndarray, seg: np.ndarray, frames: list[int]) -> None:
    if _n_frames(raw) != _n_frames(seg):
        raise ValueError(
            "Raw image and segmentation must have the same number of frames "
            f"({_n_frames(raw)} vs {_n_frames(seg)})."
        )
    for i in frames:
        r = _frame_2d(raw, i)
        s = _frame_2d(seg, i)
        if r.shape != s.shape:
            raise ValueError(
                f"Raw and segmentation shape mismatch at frame {i}: "
                f"{r.shape} vs {s.shape}."
            )


def _sanitize_model_name_fragment(name: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", name.strip())
    return safe[:80] if safe else "model"


def deterministic_cellpose_training_export_path(model_name: str) -> Path:
    """Fixed temp path for a model name (sanitized), under the OS temp directory."""
    fragment = _sanitize_model_name_fragment(model_name)
    return Path(tempfile.gettempdir()) / f"{MMV_TRAIN_DIR_PREFIX}{fragment}"


def prepare_cellpose_training_export_dir(model_name: str) -> Path:
    """
    Return an empty export directory for Cellpose training.

    If the deterministic path already exists, it is removed first.
    """
    root = deterministic_cellpose_training_export_path(model_name)
    _safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_mmvh4tracks_train_directories() -> list[Path]:
    """Top-level temp directories created for plugin Cellpose training."""
    tmp = Path(tempfile.gettempdir())
    if not tmp.is_dir():
        return []
    return sorted(
        p
        for p in tmp.iterdir()
        if p.is_dir() and p.name.startswith(MMV_TRAIN_DIR_PREFIX)
    )


def parse_model_fragment_from_train_dir_name(dir_name: str) -> str | None:
    if not dir_name.startswith(MMV_TRAIN_DIR_PREFIX):
        return None
    return dir_name[len(MMV_TRAIN_DIR_PREFIX) :]


def resume_model_fragment_needs_confirm(fragment: str) -> bool:
    """If True, user should confirm/edit the name (may include sanitization artifacts)."""
    return not bool(re.fullmatch(r"[A-Za-z0-9-]+", fragment))


def classify_mmvh4tracks_training_dir(path: Path) -> str:
    """
    Classify a training export directory for startup handling.

    Returns one of: ``empty``, ``masks_only``, ``interrupted``, ``incomplete``.
    """
    if not path.is_dir():
        return "empty"
    children = list(path.iterdir())
    if not children:
        return "empty"
    subdirs = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file()]
    mask_files = [f for f in files if _is_cellpose_training_mask_export_file(f)]
    if not subdirs and files and all(_is_cellpose_training_mask_export_file(f) for f in files):
        return "masks_only"
    if not mask_files:
        return "incomplete"
    for m in mask_files:
        raw = m.with_name(m.stem[: -len("_masks")] + m.suffix)
        if not raw.is_file():
            return "incomplete"
    return "interrupted"


def parse_layer_prefix_and_frames_from_masks_dir(
    path: Path,
) -> tuple[str, tuple[int, ...]] | None:
    """
    From ``*_frame_NNNNN_masks.tif`` files, return ``(layer_prefix, sorted frame indices)``.
    All masks must share the same prefix.
    """
    prefixes: set[str] = set()
    frames: list[int] = []
    for m in path.iterdir():
        if not m.is_file() or not _is_cellpose_training_mask_export_file(m):
            continue
        mch = _MASK_STEM_FRAME_RE.fullmatch(m.stem)
        if not mch:
            return None
        prefixes.add(mch.group(1))
        frames.append(int(mch.group(2)))
    if len(prefixes) != 1 or not frames:
        return None
    return (prefixes.pop(), tuple(sorted(set(frames))))


def _cellpose_training_raw_cyx(raw_2d: np.ndarray) -> np.ndarray:
    """
    Cellpose needs ndim>=3 for normalization (CYX), and the built-in **nuclei** U-Net
    uses ``nchan=2`` (grayscale + optional second channel; second is zeros here).
    Three channels would mismatch pretrained BatchNorm (e.g. running_mean size errors).
    """
    if raw_2d.ndim != 2:
        return raw_2d
    z = np.zeros_like(raw_2d)
    return np.stack((raw_2d, z), axis=0)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    try:
        import tifffile

        tifffile.imwrite(str(path), array)
    except ImportError:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), array)


@dataclass(frozen=True)
class CellposeTrainingExport:
    """Paths written by export_cellpose_training_pairs."""

    export_dir: Path
    frame_indices: tuple[int, ...]


def export_cellpose_training_pairs(
    viewer,
    raw_layer_name: str,
    segmentation_layer_name: str,
    frames: list[int],
    model_name: str,
) -> CellposeTrainingExport:
    """
    Write per-frame raw and mask TIFFs under a new directory in the OS temp folder.

    Layout matches Cellpose training defaults: one folder with
    ``<layer>_frame_XXXXX.tif`` and ``<layer>_frame_XXXXX_masks.tif``
    (``--mask_filter _masks``). ``<layer>`` is a filesystem-safe form of the raw
    layer name.
    """
    if not raw_layer_name.strip():
        raise ValueError("Select an image layer.")
    if not segmentation_layer_name.strip():
        raise ValueError("Select a segmentation layer.")

    raw_layer = grab_layer(viewer, raw_layer_name)
    seg_layer = grab_layer(viewer, segmentation_layer_name)
    raw = _get_array_from_layer(raw_layer)
    seg = _get_array_from_layer(seg_layer)

    _validate_frame_pairs(raw, seg, frames)

    export_root = prepare_cellpose_training_export_dir(model_name)
    layer_prefix = _sanitize_model_name_fragment(raw_layer_name)

    for idx in frames:
        r = _frame_2d(raw, idx)
        s = _frame_2d(seg, idx)
        image_path = export_root / f"{layer_prefix}_frame_{idx:05d}.tif"
        _write_tiff(image_path, _cellpose_training_raw_cyx(r))
        _write_tiff(_cellpose_training_mask_path(image_path), s)

    return CellposeTrainingExport(
        export_dir=export_root, frame_indices=tuple(frames)
    )


@dataclass(frozen=True)
class CellposeCliTrainingResult:
    """Output of ``train_cellpose`` (weights path under the temp export tree)."""

    export_dir: Path
    source_weights_path: Path
    diam_mean: float


def find_cellpose_cli_weights(export_dir: Path) -> Path:
    """
    Return the newest Cellpose CLI weights under ``export_dir/models``.

    Filenames match ``cellpose_<digits>.<digits>``; an optional file extension after
    that (e.g. ``.pth``) is accepted if present.
    """
    models_dir = Path(export_dir) / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"No models/ directory under training export {export_dir} — did training run?"
        )
    candidates = sorted(
        (p for p in models_dir.iterdir() if p.is_file() and _CELLPOSE_CLI_WEIGHTS_RE.match(p.name)),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No cellpose_<n>.<n> weights found in {models_dir} after Cellpose training."
        )
    return candidates[-1]


def _is_cellpose_training_mask_export_file(path: Path) -> bool:
    """True for mask TIFFs from plugin export (``*_frame_<5-digit-index>_masks.tif``)."""
    if not path.is_file() or path.suffix.lower() != ".tif":
        return False
    return bool(re.fullmatch(r".+_frame_\d{5}_masks", path.stem))


def prune_cellpose_training_export_dir(export_dir: Path) -> None:
    """
    Remove everything under ``export_dir`` except ``*_frame_<5-digit-index>_masks.tif``.

    Raw training images, flow TIFFs, and the ``models/`` directory are removed.
    Call only after weights have been copied elsewhere (e.g. ``persist_custom_model_entry``).
    """
    root = Path(export_dir)
    if not root.is_dir():
        return
    for child in list(root.iterdir()):
        if _is_cellpose_training_mask_export_file(child):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def _diameter_hint_from_cellpose_checkpoint(weights_path: Path) -> float:
    """Best-effort diameter for ``custom_models.json``; falls back to Cellpose default."""
    try:
        import torch

        load_kw: dict = {"map_location": "cpu"}
        try:
            ckpt = torch.load(weights_path, **load_kw, weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, **load_kw)
    except Exception:
        return 30.0
    if not isinstance(ckpt, dict):
        return 30.0
    for key in ("diam_labels", "diam_mean", "diam"):
        if key not in ckpt:
            continue
        v = ckpt[key]
        try:
            if hasattr(v, "mean"):
                return float(v.mean().item() if hasattr(v.mean(), "item") else v.mean())
            if hasattr(v, "item"):
                return float(v.item())
            return float(v)
        except (TypeError, ValueError):
            continue
    return 30.0


def train_cellpose(export_dir: Path) -> CellposeCliTrainingResult:
    """
    Run ``python -m cellpose`` training on ``export_dir`` (must match plugin export layout).

    After success, reads the newest weights under ``models/`` and a diameter hint from the checkpoint.
    """
    export_dir = Path(export_dir)

    sys.argv = [
        "cellpose",
        "--dir", str(export_dir),
        "--train",
        "--n_epochs", "20",
        "--pretrained_model", "nuclei",
        "--chan", "0",
        "--chan2", "0",
        "--min_train_mask", "1",
        "--verbose"
    ]

    cellpose_main = importlib.import_module("cellpose.__main__")
    cellpose_main.main()

    weights = find_cellpose_cli_weights(export_dir)
    diam = _diameter_hint_from_cellpose_checkpoint(weights)
    return CellposeCliTrainingResult(
        export_dir=export_dir.resolve(),
        source_weights_path=weights.resolve(),
        diam_mean=diam,
    )
