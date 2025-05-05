"""
Utility functions for handling callbacks in napari layers.
"""

from PyQt5.QtWidgets import QApplication, QWidget
from napari.viewer import Viewer
from napari.layers import Layer, Labels

class CallbackHandler:
    """Handles the addition and removal of custom callbacks for napari layers."""
    widget: QWidget
    viewer: Viewer
    _added_callback: callable | None = None
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
                    layer.mode = "paint" # Temporarily set to "paint" mode
                    layer.mode = self._cached_layer_mode
                    layer.refresh() # Not sure if refresh is needed
                    self._cached_layer_mode = None

    def remove_callback_viewer(self) -> None:
        """Removes the custom callback from all layers in the viewer.
        This should be called by all buttons that don't add a callback
        Should also be called when exiting callback handling, unless new callback is added."""
        for layer in self.viewer.layers:
            self._remove_callback(layer)
        self._added_callback = None
        QApplication.restoreOverrideCursor()

    def _add_callback(self, layer: Layer, callback: callable) -> None:
        """Adds a callback to a given layer."""
        layer.mouse_drag_callbacks.append(callback)
        if isinstance(layer, Labels):
            # Store the current layer mode
            self._cached_layer_mode = layer.mode
            # Set the layer mode to "pan_zoom" to allow for callback functionality
            # Here it does not matter if we are already in "pan_zoom" mode
            layer.mode = "pan_zoom"
            layer.refresh() # Not sure if refresh is needed

    def add_callback_viewer(self, callback: callable) -> None:  
        """Adds a callback to all layers of the viewer.
        This should be called by all buttons that add a callback"""
        self.remove_callback_viewer()
        self._added_callback = callback
        for layer in self.viewer.layers:
            self._add_callback(layer, callback)
