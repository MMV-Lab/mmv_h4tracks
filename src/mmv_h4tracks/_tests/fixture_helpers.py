"""Shared helpers for pytest fixtures (safe to import; not collected as tests)."""

from __future__ import annotations

from qtpy.QtWidgets import QApplication

from mmv_h4tracks import MMVH4TRACKS


def clear_viewer_layers(viewer) -> None:
    """Drop all napari layers as cheaply as possible."""
    viewer.layers.clear()


def reset_plugin_state(widget: MMVH4TRACKS) -> None:
    """
    Restore mutable plugin state without touching viewer layers/comboboxes.

    Use between tests that keep shared layers and only mutate layer.data /
    plugin caches.
    """
    try:
        widget.callback_handler.remove_callback_viewer()
    except Exception:
        pass

    widget.align_cache = None
    widget.eval_cache = [None, None]
    widget.is_multiscale = False
    widget.session_trained_models.clear()
    widget._session_trained_layer_ids_hooked.clear()
    widget._cellpose_ready = True
    widget.segmentation_window.apply_cellpose_ready_state()
    if hasattr(widget, "zarr"):
        delattr(widget, "zarr")

    tracking = widget.tracking_window
    tracking.cached_tracks = None
    tracking.cached_graph = None
    tracking.selected_cells = []
    tracking.reset_button_labels()

    handler = widget.callback_handler
    handler._added_callback = None
    handler._cached_layer_mode = None

    plot_window = getattr(widget, "plot_window", None)
    if plot_window is not None:
        plot_window.close()
        widget.plot_window = None

    QApplication.restoreOverrideCursor()


def reset_widget(widget: MMVH4TRACKS) -> None:
    """Full clean: clear layers and restore plugin state for the next test."""
    clear_viewer_layers(widget.viewer)
    reset_plugin_state(widget)

    # Layer removal events usually sync comboboxes; force a clean slate.
    for combobox in widget.layer_comboboxes:
        combobox.blockSignals(True)
        combobox.clear()
        combobox.blockSignals(False)
