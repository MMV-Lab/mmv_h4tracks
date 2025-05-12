"""
Utility functions for handling callbacks in napari layers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication
from napari.layers import Labels

if TYPE_CHECKING:
    from typing import Callable
    from qtpy.QtWidgets import QWidget
    from napari.viewer import Viewer
    from napari.layers import Layer


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
