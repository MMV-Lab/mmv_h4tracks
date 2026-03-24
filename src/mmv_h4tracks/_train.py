"""
Export raw / segmentation pairs for Cellpose training (per-frame TIFF files).
"""

from __future__ import annotations

import inspect
import logging
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib
import numpy as np

from ._grabber import grab_layer


def _safe_rmtree(path: Path | str) -> None:
    """Remove a directory tree if it exists; ignore errors (best-effort cleanup)."""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


# Defaults for cellpose.train.train_seg — override via train_seg_overrides in run_cellpose_training.
# min_train_masks=1: Cellpose default 5 skips sparse training images (bad for small exports).
DEFAULT_TRAIN_SEG_KWARGS: dict[str, Any] = {
    "batch_size": 1,
    "learning_rate": 1e-5,
    "n_epochs": 200,
    "weight_decay": 0.1,
    "normalize": True,
    "load_files": True,
    "min_train_masks": 1,
    "save_every": 200,
    "save_each": False,
}


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
    # Prefer tifffile (common in napari / bio stacks); fall back to imageio.
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

    Structure: ``<temp>/mmv_h4tracks_train_<model>_<random>/raw/frame_XXXXX.tif``
    and ``.../masks/frame_XXXXX.tif`` (matching basenames).
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

    fragment = _sanitize_model_name_fragment(model_name)
    export_root = Path(
        tempfile.mkdtemp(prefix=f"mmv_h4tracks_train_{fragment}_")
    )
    raw_dir = export_root / "raw"
    masks_dir = export_root / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for idx in frames:
        r = _frame_2d(raw, idx)
        s = _frame_2d(seg, idx)
        stem = f"frame_{idx:05d}.tif"
        _write_tiff(raw_dir / stem, _cellpose_training_raw_cyx(r))
        _write_tiff(masks_dir / stem, s)

    return CellposeTrainingExport(
        export_dir=export_root, frame_indices=tuple(frames)
    )


def list_training_pair_paths(export_dir: Path, pattern: str = "*.tif") -> tuple[list[str], list[str]]:
    """
    Build parallel lists of absolute paths to raw images and mask images.

    Expects ``export_dir/raw/`` and ``export_dir/masks/`` with matching basenames.
    """
    raw_dir = Path(export_dir) / "raw"
    masks_dir = Path(export_dir) / "masks"
    if not raw_dir.is_dir() or not masks_dir.is_dir():
        raise ValueError("Export directory must contain 'raw' and 'masks' subfolders.")
    raw_files = sorted(raw_dir.glob(pattern))
    train_files: list[str] = []
    label_files: list[str] = []
    for rf in raw_files:
        mf = masks_dir / rf.name
        if not mf.is_file():
            raise ValueError(f"Missing matching mask for {rf.name} in masks/.")
        train_files.append(str(rf.resolve()))
        label_files.append(str(mf.resolve()))
    return train_files, label_files


def _resolve_saved_weights_path(save_path_base: Path, train_seg_filename) -> Path:
    """Resolve path returned by train_seg to an existing file on disk."""
    p = Path(train_seg_filename)
    if p.is_file():
        return p.resolve()
    for suffix in (".pth", ".pt"):
        cand = Path(str(train_seg_filename) + suffix)
        if cand.is_file():
            return cand.resolve()
    models_dir = save_path_base / "models"
    if models_dir.is_dir():
        found = sorted(models_dir.glob("*.pth"))
        if len(found) == 1:
            return found[0].resolve()
        if found:
            return max(found, key=lambda x: x.stat().st_mtime).resolve()
    raise FileNotFoundError(
        f"Could not locate saved model weights under {save_path_base} (train_seg returned {train_seg_filename!r})."
    )


def _cellpose_net(cellpose_model) -> Any:
    if hasattr(cellpose_model, "net"):
        return cellpose_model.net
    raise AttributeError(
        "CellposeModel has no 'net' attribute; cannot run train_seg with this Cellpose version."
    )


def _kwargs_for_train_seg(train_seg_fn, kwargs_for_call: dict[str, Any]) -> dict[str, Any]:
    """
    Drop unsupported keys for this Cellpose version. If ``train_seg`` exposes ``**kwargs``,
    pass through all entries (needed for tests that replace ``train_seg`` with a stub).
    """
    sig = inspect.signature(train_seg_fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs_for_call
    names = set(sig.parameters.keys())
    return {k: v for k, v in kwargs_for_call.items() if k in names}


def use_gpu_for_cellpose() -> bool:
    """True if Cellpose should use CUDA (when available)."""
    try:
        import torch
        from cellpose import core

        return bool(core.use_gpu() and torch.cuda.is_available())
    except Exception:
        return False


def _import_cellpose_train():
    """
    Load ``cellpose.train`` lazily. Some installs omit this submodule; importing it at
    module load would break the napari plugin before export-only code can run.
    """
    try:
        return importlib.import_module("cellpose.train")
    except ModuleNotFoundError as e:
        raise ImportError(
            "Cellpose training is not available: the 'cellpose.train' module is missing "
            "from this environment. Use a full Cellpose install that includes training "
            "(see Cellpose docs / upgrade cellpose)."
        ) from e


@contextmanager
def _tqdm_cli_friendly():
    """
    Cellpose uses tqdm; in Napari/GUI launches stderr is often not a TTY and tqdm
    disables itself. Force progress to stderr with a minimum update interval to limit noise.
    """
    try:
        import tqdm as tqdm_pkg
        import tqdm.std as tqdm_std
    except ImportError:
        yield
        return

    orig_std = tqdm_std.tqdm
    orig_pkg = tqdm_pkg.tqdm

    # Subclass, do not replace `tqdm` with a function: Cellpose/tqdm use class attrs
    # like `format_interval` on the tqdm name.
    class _TqdmCliFriendly(orig_std):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("file", sys.stderr)
            kwargs.setdefault("mininterval", 1.0)
            if kwargs.get("disable") is None:
                kwargs["disable"] = False
            super().__init__(*args, **kwargs)

    tqdm_std.tqdm = _TqdmCliFriendly
    tqdm_pkg.tqdm = _TqdmCliFriendly
    try:
        yield
    finally:
        tqdm_std.tqdm = orig_std
        tqdm_pkg.tqdm = orig_pkg


@contextmanager
def _cellpose_train_logging_to_stderr():
    """Emit Cellpose training log lines (epoch losses, etc.) to stderr."""
    lg = logging.getLogger("cellpose.train")
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    old_level = lg.level
    lg.addHandler(handler)
    if lg.getEffectiveLevel() > logging.INFO:
        lg.setLevel(logging.INFO)
    try:
        yield
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


@dataclass(frozen=True)
class CellposeTrainingRunResult:
    """Output of run_cellpose_training (fine-tune from exported TIFF pairs)."""

    train_save_root: Path
    weights_path: Path
    train_losses: np.ndarray
    test_losses: np.ndarray
    diam_mean: float


def run_cellpose_training(
    export_dir: Path,
    model_name: str,
    train_seg_overrides: dict[str, Any] | None = None,
) -> CellposeTrainingRunResult:
    """
    Fine-tune Cellpose's **nuclei** pretrained model on exported ``raw``/``masks`` TIFF pairs.

    Training checkpoints are written under a new OS temp directory (not ``export_dir``).
    """
    train_files, train_labels_files = list_training_pair_paths(export_dir)
    if not train_files:
        raise ValueError("No training images found in export raw/ folder.")

    fragment = _sanitize_model_name_fragment(model_name)
    train_root = Path(tempfile.mkdtemp(prefix=f"mmv_h4tracks_cellpose_train_{fragment}_"))

    try:
        kwargs: dict[str, Any] = {**DEFAULT_TRAIN_SEG_KWARGS, **(train_seg_overrides or {})}
        kwargs["train_files"] = train_files
        kwargs["train_labels_files"] = train_labels_files
        kwargs["save_path"] = str(train_root)
        kwargs["model_name"] = fragment

        from cellpose import models

        cellpose_train = _import_cellpose_train()
        gpu = use_gpu_for_cellpose()
        cp_model = models.CellposeModel(gpu=gpu, pretrained_model="nuclei")
        net = _cellpose_net(cp_model)

        train_kwargs = _kwargs_for_train_seg(cellpose_train.train_seg, kwargs)

        with _tqdm_cli_friendly():
            with _cellpose_train_logging_to_stderr():
                out = cellpose_train.train_seg(net, **train_kwargs)
        filename_returned = out[0]
        train_losses = np.asarray(out[1]) if len(out) > 1 else np.array([])
        test_losses = np.asarray(out[2]) if len(out) > 2 else np.array([])

        weights_path = _resolve_saved_weights_path(train_root, filename_returned)

        diam_mean = 30.0
        try:
            dlm = net.diam_labels
            if hasattr(dlm, "mean"):
                diam_mean = float(dlm.mean().item())
            elif hasattr(dlm, "item"):
                diam_mean = float(dlm.item())
            else:
                diam_mean = float(dlm)
        except Exception:
            pass

        return CellposeTrainingRunResult(
            train_save_root=train_root,
            weights_path=weights_path,
            train_losses=train_losses,
            test_losses=test_losses,
            diam_mean=diam_mean,
        )
    except Exception:
        _safe_rmtree(train_root)
        raise
