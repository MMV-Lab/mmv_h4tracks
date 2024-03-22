import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QVBoxLayout,
    QPushButton,
    QGridLayout,
    QSizePolicy,
    QApplication,
)
from qtpy.QtCore import Qt
from scipy import ndimage
import napari
import pandas as pd

from ._logger import notify, handle_exception
from ._grabber import grab_layer
import mmv_h4tracks._processing as processing
from .add_models import ModelWindow


class SegmentationWindow(QWidget):
    """
    A (QWidget) window to correct the segmentation of the data.

    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance

    Methods
    -------
    """

    def __init__(self, parent):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.parent = parent
        self.viewer = parent.viewer
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        self.custom_models = processing.read_custom_model_dict()
        # Used to cach the callback on the label layer
        self.cached_callback = None

        ### QObjects

        # Buttons
        btn_false_positive = QPushButton("Remove cell")
        btn_false_positive.setToolTip(
            "Remove label from segmentation\n"
            "WARNING: If a tracked cell is removed the track will be cut!"
        )

        btn_free_label = QPushButton("Next free ID")
        btn_free_label.setToolTip("Load next free segmentation label \n\n" "Hotkey: E")

        btn_false_merge = QPushButton("Separate")
        btn_false_merge.setToolTip(
            "Split two separate parts of the same label into two"
        )

        btn_false_cut = QPushButton("Merge cell")
        btn_false_cut.setToolTip("Merge two separate labels into one")

        btn_grab_label = QPushButton("Select ID")
        btn_grab_label.setToolTip("Load selected segmentation label")

        self.btn_segment = QPushButton("Run Segmentation")
        btn_segment_tooltip = (
            "Run instance segmentation with selected model.\n"
            "Computation varies depending on GPU/CPU computation.\n"
            "Selecting preview segments only 5 slices."
        )
        self.btn_segment.setToolTip(btn_segment_tooltip)

        self.btn_add_custom_model = QPushButton("Add custom Cellpose model")
        btn_add_custom_model_tooltip = (
            "Add a custom trained Cellpose model.\n"
            "Note: This is not for training a Cellpose model."
        )
        self.btn_add_custom_model.setToolTip(btn_add_custom_model_tooltip)

        btn_false_positive.clicked.connect(self._add_remove_callback)
        btn_free_label.clicked.connect(self._set_label_id)
        btn_false_merge.clicked.connect(self._add_replace_callback)
        btn_false_cut.clicked.connect(self._add_merge_callback)
        self.btn_segment.clicked.connect(self.segment)
        self.btn_add_custom_model.clicked.connect(self._add_model)
        btn_grab_label.clicked.connect(self._add_select_callback)

        # QComboBoxes
        self.combobox_segmentation = QComboBox()
        self.combobox_segmentation.setToolTip("select model")
        hardcoded_models, custom_models = processing.read_models(self)
        processing.display_models(self, hardcoded_models, custom_models)
        self.combobox_segmentation.currentTextChanged.connect(
            self.toggle_segmentation_button
        )

        # QCheckBoxes
        self.checkbox_preview = QCheckBox("Preview")

        # Spacer
        v_spacer = QWidget()
        v_spacer.setFixedWidth(4)
        v_spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(0)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_2 = QWidget()
        h_spacer_2.setFixedHeight(0)
        h_spacer_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        automatic_segmentation = QGroupBox("Automatic segmentation")
        automatic_segmentation.setLayout(QGridLayout())
        automatic_segmentation.layout().addWidget(h_spacer_1, 0, 0, 1, -1)
        automatic_segmentation.layout().addWidget(
            self.combobox_segmentation, 1, 0, 1, 1
        )
        automatic_segmentation.layout().addWidget(self.btn_segment, 1, 1, 1, 1)
        automatic_segmentation.layout().addWidget(self.checkbox_preview, 1, 2, 1, 1)
        automatic_segmentation.layout().addWidget(
            self.btn_add_custom_model, 2, 0, 1, -1
        )

        segmentation_correction = QGroupBox("Segmentation correction")
        segmentation_correction.setLayout(QGridLayout())
        segmentation_correction.layout().addWidget(h_spacer_2, 0, 0, 1, -1)
        segmentation_correction.layout().addWidget(btn_false_positive, 1, 0)
        segmentation_correction.layout().addWidget(btn_free_label, 2, 0)
        segmentation_correction.layout().addWidget(btn_grab_label, 2, 1)
        segmentation_correction.layout().addWidget(btn_false_cut, 3, 0)
        segmentation_correction.layout().addWidget(btn_false_merge, 3, 1)

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(automatic_segmentation)
        content.layout().addWidget(segmentation_correction)
        content.layout().addWidget(v_spacer)

        self.layout().addWidget(content)

    def segment(self):
        """
        Runs the segmentation with the selected model.
        """
        if self.checkbox_preview.isChecked():
            processing.run_demo_segmentation(self)
        else:
            processing.run_segmentation(self)

    def _add_model(self):
        """
        Opens a [ModelWindow]
        """
        self.model_window = ModelWindow(self)
        self.model_window.show()

    def toggle_segmentation_button(self, text):
        """
        Toggles the segmentation button if a valid model is selected.

        Parameters
        ----------
        text : str
            the text of the combobox
        """
        if text == "selected model":
            self.btn_segment.setEnabled(False)
        else:
            self.btn_segment.setEnabled(True)

    def _add_remove_callback(self):
        """
        Adds the callback to remove the label at the given position from the segmentation layer
        """
        try:
            grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        except ValueError as exc:
            handle_exception(exc)
            return

        def _remove_label(_, event):
            """
            Removes the cell at the given position from the segmentation layer and updates the callbacks

            Parameters
            ----------
            layer : Layer
                the layer that triggered the callback
            event : Event
                the event that triggered the callback"""
            self._remove_label(event)
            self._update_callbacks()

        self._update_callbacks(_remove_label)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def _remove_label(self, event):
        """
        Removes the cell at the given position from the segmentation layer

        Parameters
        ----------
        event : Event
            the event that triggered the callback
        """
        self.remove_cell_from_tracks(event.position)

        # replace label with 0 to make it background
        self._replace_label(event, 0)

    def remove_cell_from_tracks(self, position):
        """
        Removes the cell at the given position from the tracks layer

        Parameters
        ----------
        position : list
            the position of the cell to remove
        """
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )

        if label_layer is None:
            QApplication.restoreOverrideCursor()
            notify("No segmentation layer found")
            return

        x = int(round(position[2]))
        y = int(round(position[1]))
        z = int(position[0])
        selected_id = label_layer.data[z, y, x]
        if selected_id == 0:
            print("no cell")
            return
        centroid = ndimage.center_of_mass(
            label_layer.data[z], labels=label_layer.data[z], index=selected_id
        )
        cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]

        try:
            track_id, displayed = self.get_track_id_of_cell(cell)
        except ValueError:
            print("cell untracked")
            return

        tracks_name = self.parent.combobox_tracks.currentText()
        if tracks_name == "":
            print("no tracks")
            return
        tracks_layer = grab_layer(self.viewer, tracks_name)
        displayed_tracks = tracks_layer.data
        all_tracks = [displayed_tracks]

        if self.parent.tracking_window.cached_tracks is not None:
            next_id = max(self.parent.tracking_window.cached_tracks[:, 0]) + 1
            all_tracks.append(self.parent.tracking_window.cached_tracks)
        else:
            next_id = max(displayed_tracks[:, 0]) + 1
        indices_to_delete = [[], []]

        for indicator, tracks in enumerate(all_tracks):
            # find index of entry in displayed tracks
            for i in range(len(tracks)):
                if np.all(tracks[i, 1:4] == cell):
                    index = i
                    # get track id of that entry
                    track_id = tracks[i][0]
                    break

            # find first and last index of that track id
            indices = np.where(tracks[:, 0] == track_id)[0]
            if len(indices) == 0:
                indices_to_delete[indicator] = []
                continue
            first = min(indices)
            last = max(indices)

            # if index != first and != last index change id of all entries after index
            if index > first + 1:
                indices = indices[indices >= index]
            if index < last - 1:
                indices = indices[indices <= index]

            if len(indices) == 1:
                for i in range(index + 1, last + 1):
                    tracks[i][0] = next_id

            indices_to_delete[indicator] = indices

        # remove entry (or entries)
        if displayed:
            displayed_tracks = np.delete(displayed_tracks, indices_to_delete[0], 0)
            tracks_layer.data = displayed_tracks
        if self.parent.tracking_window.cached_tracks is not None:
            self.parent.tracking_window.cached_tracks = np.delete(
                self.parent.tracking_window.cached_tracks, indices_to_delete[1], 0
            )
            df = pd.DataFrame(
                self.parent.tracking_window.cached_tracks, columns=["ID", "Z", "Y", "X"]
            )
            df.sort_values(["ID", "Z"], ascending=True, inplace=True)
            self.parent.tracking_window.cached_tracks = df.values

    def get_track_id_of_cell(self, cell):
        """
        Returns the track id of the cell at the given position

        Parameters
        ----------
        cell : list
            the position of the cell

        Returns
        -------
        int
            the track id of the cell
        bool
            whether the track is displayed or not
        """
        tracks_name = self.parent.combobox_tracks.currentText()
        tracks_layer = grab_layer(self.viewer, tracks_name)
        if tracks_layer is None:
            return
        tracks = tracks_layer.data

        for track in tracks:
            if np.all(track[1:4] == cell):
                return track[0], True
        if self.parent.tracking_window.cached_tracks is not None:
            for track in self.parent.tracking_window.cached_tracks:
                if np.all(track[1:4] == cell):
                    return track[0], False
        raise ValueError("No matching track found")

    def _add_select_callback(self):
        """
        Adds the callback to select the label at the given position
        """
        try:
            segmentation_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError as exc:
            handle_exception(exc)
            return

        def _select_label(_, event):
            """
            Selects the label at the given position and updates the callbacks

            Parameters
            ----------
            layer : Layer
                the layer that triggered the callback
            event : Event
                the event that triggered the callback
            """
            id = self._read_label_id(event)
            self._set_label_id(id)
            self._update_callbacks()

        self._update_callbacks(_select_label)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def _set_label_id(self, id=0):
        """
        Sets the given id as the current id in the label layer

        Parameters
        ----------
        id : int
            the ID to set as the currently selected one in the napari viewer
        """
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            notify("Please make sure the label layer exists!")
            return

        if id == 0:
            id = self._get_free_label_id(label_layer)

        # set the new id
        label_layer.selected_label = id

        # set the label layer as currently selected layer
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(label_layer)

    def _get_free_label_id(self, label_layer):
        """
        Calculates the next free id in the passed layer
        (always returns maximum value + 1 for now, could be changed later)

        Parameters
        ----------
        label_layer : layer
            label layer to calculate the free id for

        Returns
        -------
        int
            a free id
        """
        return np.amax(label_layer.data) + 1

    def _add_replace_callback(self):
        """
        Adds the callback to replace the label at the given position with the currently selected one
        """
        try:
            grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        except ValueError as exc:
            handle_exception(exc)
            return

        def _replace_label(_, event):
            """
            Replaces the label at the given position with the currently selected one and updates the callbacks

            Parameters
            ----------
            layer : Layer
                the layer that triggered the callback
            event : Event
                the event that triggered the callback
            """
            self._replace_label(event)
            self._update_callbacks()

        self._update_callbacks(_replace_label)
        QApplication.setOverrideCursor(Qt.CrossCursor)
        """for layer in self.viewer.layers:

            @layer.mouse_drag_callbacks.append
            def _replace_label(layer, event):
                self._replace_label(event)
                self._update_callbacks()"""

    def _add_merge_callback(self):
        """
        Adds the callback to merge the label at the given position with the currently selected one
        """
        try:
            grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        except ValueError as exc:
            handle_exception(exc)
            return

        def _pick_merge_label(layer, event):
            """
            Picks the label to merge with and updates the callbacks

            Parameters
            ----------
            layer : Layer
                the layer that triggered the callback
            event : Event
                the event that triggered the callback
            """
            id = self._read_label_id(event)

            def _assimilate_label(_, event):
                """
                Assimilates the label at the given position with the currently selected one and updates the callbacks

                Parameters
                ----------
                layer : Layer
                    the layer that triggered the callback
                event : Event
                    the event that triggered the callback
                """
                self._replace_label(event, id)
                self._update_callbacks()

            self._update_callbacks(_assimilate_label)
            QApplication.setOverrideCursor(Qt.CrossCursor)

        self._update_callbacks(_pick_merge_label)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def _replace_label(self, event, id=-1):
        """
        Replaces the label at the given position with the given ID

        Parameters
        ----------
        position : list
            list of float values describing the position the user clicked on the layer (z,y,x)
        id : int
            the id to set for the given position
        """
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            notify("Please make sure the label layer exists!")
            return

        x = int(round(event.position[2]))
        y = int(round(event.position[1]))
        z = int(round(event.position[0]))

        if id == -1:
            id = self._get_free_label_id(label_layer)

        # Replace the ID with the new id
        old_id = label_layer.data[z, y, x]
        """if old_id == 0:
            notify("Can't change ID of background, please make sure to select a cell!")
            return"""
        label_layer.fill((z, y, x), id)

        # set the label layer as currently selected layer
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(label_layer)

    def _read_label_id(self, event):
        """
        Reads the label id at the given position

        Parameters
        ----------
        event : Event
            the event that triggered the callback

        Returns
        -------
        int
            id at the given position in the segmentation layer
        """
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            notify("Please make sure the label layer exists!")
            return

        x = int(round(event.position[2]))
        y = int(round(event.position[1]))
        z = int(round(event.position[0]))

        return label_layer.data[z, y, x]

    def _update_callbacks(self, callback=None):
        """
        Updates the callbacks of all layers

        Parameters
        ----------
        callback : function
            the callback to add to the layers
        """
        if not len(self.viewer.layers):
            return
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if callback:
            current_callback = (
                label_layer.mouse_drag_callbacks[0]
                if label_layer.mouse_drag_callbacks
                else None
            )
            if current_callback and current_callback.__qualname__ in ["draw", "pick"]:
                self.cached_callback = current_callback
            for layer in self.viewer.layers:
                layer.mouse_drag_callbacks = [callback]
        else:
            for layer in self.viewer.layers:
                if layer == label_layer and self.cached_callback:
                    layer.mouse_drag_callbacks = [self.cached_callback]
                else:
                    layer.mouse_drag_callbacks = []
            self.cached_callback = None
        QApplication.restoreOverrideCursor()
