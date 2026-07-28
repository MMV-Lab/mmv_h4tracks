"""Small Qt / napari UI helpers shared across tab windows."""

from __future__ import annotations

import napari
import numpy as np


def apply_napari_dark_theme(widget) -> None:
    """Apply napari's dark stylesheet (API differs across napari versions)."""
    try:
        widget.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
    except TypeError:
        widget.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))


def layer_as_numpy(layer) -> np.ndarray:
    """
    Return layer ``.data`` as a numpy array.

    Multiscale / pyramid layers store a list or tuple of levels; the highest
    resolution (first) level is used.
    """
    data = layer.data
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Layer has empty multiscale data")
        data = data[0]
    return np.asarray(data)
