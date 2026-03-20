"""
Utility functions for handling callbacks in napari layers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import QApplication
from napari.layers import Labels

if TYPE_CHECKING:
    from typing import Callable
    from qtpy.QtWidgets import QWidget
    from napari.viewer import Viewer
    from napari.layers import Layer, Tracks


class CallbackHandler:
    """Handles the addition and removal of custom callbacks for napari layers."""

    widget: QWidget
    viewer: Viewer
    _added_callback: Callable | None = None
    _cached_layer_mode: str | None = None

    def __init__(self, widget):
        self.widget = widget
        self.viewer = widget.viewer

    def print_callbacks(self, layer: Layer) -> None:
        """Prints the callbacks of a given layer."""
        for callback in layer.mouse_drag_callbacks:
            print(callback)

    def print_cache(self) -> None:
        """Prints the cached information.
        Includes the cached callback and the layer mode of the label layer."""
        print(f"Cached callback: {self._cached_layer_mode}")
        print(f"Layer mode: {self._cached_layer_mode}")

    def _remove_callback(self, layer: Layer) -> None:
        """Removes the custom callback from a given layer."""
        if (
            self._added_callback is not None
            and self._added_callback in layer.mouse_drag_callbacks
        ):
            layer.mouse_drag_callbacks.remove(self._added_callback)
            if isinstance(layer, Labels):
                # Restore original layer mode
                # Mode needs to be changed for callbacks to work properly
                if self._cached_layer_mode is not None:
                    layer.mode = "paint"  # Temporarily set to "paint" mode
                    layer.mode = self._cached_layer_mode
                    layer.refresh()  # Not sure if refresh is needed
                    self._cached_layer_mode = None

    def remove_callback_viewer(self, keep_tracking: bool = False) -> None:
        """Removes the custom callback from all layers in the viewer.
        This should be called by all buttons that don't add a callback
        Should also be called when exiting callback handling, unless new callback is added.
        """
        for layer in self.viewer.layers:
            self._remove_callback(layer)
        self._added_callback = None
        if not keep_tracking:
            # Reset the tracking stage
            self.reset_tracking_stage()
        QApplication.restoreOverrideCursor()

    def reset_tracking_stage(self) -> None:
        """Resets the buttons and cached cells"""
        self.widget.tracking_window.reset_button_labels()
        self.widget.tracking_window.selected_cells = []

    def _add_callback(self, layer: Layer, callback: Callable) -> None:
        """Adds a callback to a given layer."""
        layer.mouse_drag_callbacks.append(callback)
        if isinstance(layer, Labels):
            # Store the current layer mode
            self._cached_layer_mode = layer.mode
            # Set the layer mode to "pan_zoom" to allow for callback functionality
            # Here it does not matter if we are already in "pan_zoom" mode
            layer.mode = "pan_zoom"
            layer.refresh()  # Not sure if refresh is needed

    def add_callback_viewer(
        self, callback: Callable, keep_tracking: bool = False
    ) -> None:
        """Adds a callback to all layers of the viewer.
        This should be called by all buttons that add a callback"""
        self.remove_callback_viewer(keep_tracking)
        self._added_callback = callback
        for layer in self.viewer.layers:
            self._add_callback(layer, callback)


def preserve_and_filter_graph(tracks_layer, new_tracks_data: np.ndarray) -> dict:
    """
    Preserves the graph from an existing tracks layer and filters it to only include
    track IDs that exist in the new tracks data.
    
    This prevents napari validation errors when the graph contains references to
    track IDs that don't exist in the tracks data (e.g., after filtering).
    
    Parameters
    ----------
    tracks_layer : Tracks | None
        The existing tracks layer (may be None if no layer exists)
    new_tracks_data : np.ndarray
        The new tracks data array with shape (N, 4) where first column is track_id
        
    Returns
    -------
    dict
        Filtered graph dictionary mapping track_id -> list of parent_ids.
        Returns empty dict if no graph exists or no valid track IDs remain.
    """
    # Get graph from existing layer if it exists
    graph = {}
    if tracks_layer is not None:
        graph = getattr(tracks_layer, 'graph', {}) or {}
    
    if not graph:
        return {}
    
    # Get valid track IDs from new tracks data
    valid_track_ids = set(np.unique(new_tracks_data[:, 0]).astype(int))
    
    # Filter graph to only include entries where:
    # 1. The track_id (key) exists in new tracks
    # 2. All parent_ids (values) exist in new tracks
    filtered_graph = {}
    for track_id, parent_ids in graph.items():
        track_id_int = int(track_id)
        if track_id_int in valid_track_ids:
            # Filter parent_ids to only those that exist in new tracks
            filtered_parents = [
                int(p) for p in parent_ids 
                if int(p) in valid_track_ids
            ]
            # Only add entry if at least one valid parent remains (or if empty list is valid)
            # Empty list means no parents (root track)
            if filtered_parents or not parent_ids:
                filtered_graph[track_id_int] = filtered_parents if filtered_parents else []
    
    return filtered_graph
