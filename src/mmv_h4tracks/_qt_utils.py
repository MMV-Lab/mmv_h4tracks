"""Small Qt / napari UI helpers shared across tab windows."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import napari
import numpy as np

from ._constants import STATUS_AWAITING_USER, STATUS_READY

logger = logging.getLogger(__name__)

_dock_status_host = None


def register_dock_status_host(widget) -> None:
    """Register the main dock widget used for progress/status updates."""
    global _dock_status_host
    _dock_status_host = widget


def resolve_dock_status_host(widget=None):
    """Return a widget with ``set_status_text``, if available."""
    candidates = []
    if widget is not None:
        candidates.append(widget)
        parent_attr = getattr(widget, "parent", None)
        if callable(parent_attr):
            try:
                candidates.append(parent_attr())
            except TypeError:
                pass
        elif parent_attr is not None:
            candidates.append(parent_attr)
    candidates.append(_dock_status_host)
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "set_status_text"):
            return candidate
    return None


@contextmanager
def awaiting_user_dialog(host=None):
    """
    Set the dock status label to ``STATUS_AWAITING_USER`` for a modal dialog.

    Restores the previous status text when the dialog closes.
    """
    status_host = resolve_dock_status_host(host)
    if status_host is None:
        yield
        return
    label = getattr(status_host, "status_label", None)
    previous = label.text() if label is not None else STATUS_READY
    status_host.set_status_text(STATUS_AWAITING_USER)
    if label is not None:
        label.repaint()
    try:
        yield
    finally:
        status_host.set_status_text(previous)
        if label is not None:
            label.repaint()


def apply_napari_dark_theme(widget) -> None:
    """Apply napari's dark stylesheet (API differs across napari versions)."""
    try:
        widget.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
    except TypeError:
        widget.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))


def _materialize_array(data) -> np.ndarray:
    """Convert layer level data to a concrete numpy array (handles dask/zarr)."""
    if hasattr(data, "compute"):
        try:
            data = data.compute()
        except Exception:
            pass
    return np.asarray(data)


def _spatial_pixel_count(arr: np.ndarray) -> int:
    """Score for picking the highest-resolution pyramid level (YX plane size)."""
    if arr.ndim >= 2:
        return int(np.prod(arr.shape[-2:]))
    return int(arr.size)


def _iter_multiscale_levels(data):
    """Yield level arrays from a list/tuple or napari-like multiscale sequence."""
    if isinstance(data, (list, tuple)):
        return list(data)
    # Duck-type napari MultiScaleData / similar (has len + getitem, no ndarray shape)
    if hasattr(data, "__getitem__") and hasattr(data, "__len__") and not hasattr(
        data, "shape"
    ):
        return [data[i] for i in range(len(data))]
    return None


def layer_as_numpy(layer) -> np.ndarray:
    """
    Return layer ``.data`` as a concrete numpy array.

    Multiscale / pyramid layers store several resolutions. The level with the
    largest spatial (Y×X) size is used so Cellpose never silently runs on a
    coarse pyramid level when level ordering differs from ``data[0]``.
    """
    data = layer.data
    levels = _iter_multiscale_levels(data)
    if levels is None and getattr(layer, "multiscale", False):
        # Flagged multiscale but unusual container — try indexing.
        # Skip when ``data`` already has a ``shape`` (ndarray / dask / zarr): a
        # truthy ``multiscale`` on mocks or mis-set flags must not treat T/Z as
        # pyramid levels.
        if not hasattr(data, "shape"):
            try:
                levels = [data[i] for i in range(len(data))]
            except Exception as exc:
                raise ValueError(
                    f"Could not read multiscale levels from layer "
                    f"{getattr(layer, 'name', layer)!r}"
                ) from exc

    if levels is not None:
        if len(levels) == 0:
            raise ValueError("Layer has empty multiscale data")
        materialized = [_materialize_array(level) for level in levels]
        scores = [_spatial_pixel_count(level) for level in materialized]
        chosen_idx = int(np.argmax(scores))
        chosen = materialized[chosen_idx]
        shapes = [tuple(level.shape) for level in materialized]
        logger.info(
            "layer_as_numpy multiscale: layer=%r levels=%s chose_index=%s "
            "chose_shape=%s (yx=%s)",
            getattr(layer, "name", None),
            shapes,
            chosen_idx,
            tuple(chosen.shape),
            chosen.shape[-2:] if chosen.ndim >= 2 else chosen.shape,
        )
        return np.array(chosen, copy=True)

    out = np.array(_materialize_array(data), copy=True)
    logger.debug(
        "layer_as_numpy: layer=%r shape=%s dtype=%s yx=%s",
        getattr(layer, "name", None),
        tuple(out.shape),
        out.dtype,
        out.shape[-2:] if out.ndim >= 2 else out.shape,
    )
    return out
