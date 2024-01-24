import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QSizePolicy,
    QApplication,
)
from qtpy.QtCore import Qt
from scipy import ndimage
import napari

from ._logger import notify
from ._grabber import grab_layer
import mmv_tracking_napari._processing as processing
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
            self.setStyleSheet(napari.qt.get_stylesheet(theme = "dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        ### QObjects

        # Buttons
        btn_false_positive = QPushButton("Remove")
        btn_false_positive.setToolTip("Remove label from segmentation")

        btn_free_label = QPushButton("Next free")
        btn_free_label.setToolTip("Load next free segmentation label")

        btn_false_merge = QPushButton("Separate")
        btn_false_merge.setToolTip(
            "Split two separate parts of the same label into two"
        )

        btn_false_cut = QPushButton("Merge")
        btn_false_cut.setToolTip("Merge two separate labels into one")

        btn_grab_label = QPushButton("Select")
        btn_grab_label.setToolTip("Load selected segmentation label")

        self.btn_segment = QPushButton("Run Segmentation")

        self.btn_add_custom_model = QPushButton("Add custom Cellpose model")

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
        processing.read_models(self)
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
        automatic_segmentation = QGroupBox("Automatic Segmentation")
        automatic_segmentation.setLayout(QGridLayout())
        automatic_segmentation.layout().addWidget(h_spacer_1, 0, 0, 1, -1)
        automatic_segmentation.layout().addWidget(self.combobox_segmentation, 1, 0, 1, 1)
        automatic_segmentation.layout().addWidget(self.btn_segment, 1, 1, 1, 1)
        automatic_segmentation.layout().addWidget(self.checkbox_preview, 1, 2, 1, 1)
        automatic_segmentation.layout().addWidget(self.btn_add_custom_model, 2, 0, 1, -1)

        segmentation_correction = QGroupBox("Segmentation Correction")
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
        if self.checkbox_preview.isChecked():
            processing.run_demo_segmentation(self)
        else:
            processing.run_segmentation(self)
        
    def _add_model(self):
        """
        Opens a [ModelWindow]
        """
        self.model_window = ModelWindow(self)
        print("Opening model window")
        self.model_window.show()

    def toggle_segmentation_button(self, text):
        """
        Toggles the segmentation button if a valid model is selected.
        
        Args:
            text (str): Name of the selected model

        Returns:
            None
        """
        if text == "selected model":
            self.btn_segment.setEnabled(False)
        else:
            self.btn_segment.setEnabled(True)

    def _add_remove_callback(self):
        self._remove_on_clicks()
        QApplication.setOverrideCursor(Qt.CrossCursor)
        for layer in self.viewer.layers:

            @layer.mouse_drag_callbacks.append
            def _remove_label(layer, event):
                self._remove_label(event)

        print("Added callback to remove cell")

    def _remove_label(self, event):
        """
        Removes the cell at the given position from the segmentation layer

        Parameters
        ----------
        position : list
            list of float values describing the position the user clicked on the layer (z,y,x)
        """
        self.remove_cell_from_tracks(event.position)
        
        # replace label with 0 to make it background
        self._replace_label(event, 0)
        print("Removed cell")
        self._remove_on_clicks()
        print("Remove cell callback is removed")
        
    def remove_cell_from_tracks(self, position): # TODO: doesn't work for cached tracks
        label_layer = grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        
        if label_layer is None:
            print("Segmentation layer not selected")
            QApplication.restoreOverrideCursor()
            notify("No segmentation layer found")
            return
            
        x = int(round(position[2]))
        y = int(round(position[1]))
        z = int(position[0])
        selected_id = label_layer.data[z,y,x]
        if selected_id == 0:
            return
        centroid = ndimage.center_of_mass(
            label_layer.data[z],
            labels = label_layer.data[z],
            index = selected_id
        )
        cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
        try:
            track_id = self.get_track_id_of_cell(cell)
        except ValueError:
            return

        tracks_name = self.parent.combobox_tracks.currentText()
        if tracks_name == "":
            return
        tracks = grab_layer(self.viewer, tracks_name).data
        
        next_id = max(tracks[:,0]) + 1

        # find index of entry
        for i in range(len(tracks)):
            if (
                tracks[i, 1] == cell[0]
                and tracks[i, 2] == cell[1]
                and tracks[i, 3] == cell[2]
            ):
                index = i
                # get track id of that entry
                track_id = tracks[i][0]
                break
                
        # find first and last index of that track id
        indices = np.where(tracks[:,0] == track_id)[0]
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
                
        # remove entry (or entries)
        tracks = np.delete(tracks, indices, 0)
        self.viewer.layers.remove(tracks_name)
        self.viewer.add_tracks(tracks, name=tracks_name)
        
    def get_track_id_of_cell(self, cell):
        tracks_name = self.parent.combobox_tracks.currentText()
        """if tracks_name == "":
            return"""
        tracks_layer = grab_layer(self.viewer, tracks_name)
        if tracks_layer is None:
            return
        tracks = tracks_layer.data
        new_track_id = np.amax(tracks[:,0]) + 1
        
        for i in range(len(tracks)):
            if(
                tracks[i, 1] == cell[0]
                and tracks[i, 2] == cell[1]
                and tracks[i, 3] == cell[2]
            ):
                return tracks[i,0]
        raise ValueError('No matching track found')

    def _add_select_callback(self):
        self._remove_on_clicks()
        QApplication.setOverrideCursor(Qt.CrossCursor)
        for layer in self.viewer.layers:

            @layer.mouse_drag_callbacks.append
            def _select_label(layer, event):
                id = self._read_label_id(event)
                self._set_label_id(id)
                self._remove_on_clicks()
                print("Select cell callback is removed")

        print("Added callback to select a label")

    def _set_label_id(self, id=0):
        """
        Sets the given id as the current id in the label layer

        Parameters
        ----------
        id : int
            the ID to set as the currently selected one in the napari viewer
        """
        label_layer = grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            print("Tried to set label id on missing label layer")
            notify("Please make sure the label layer exists!")
            return

        if id == 0:
            id = self._get_free_label_id(label_layer)

        # set the new id
        label_layer.selected_label = id #self._get_free_label_id(label_layer)

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
        Sets a new id on the clicked label
        """
        self._remove_on_clicks()
        QApplication.setOverrideCursor(Qt.CrossCursor)
        for layer in self.viewer.layers:

            @layer.mouse_drag_callbacks.append
            def _replace_label(layer, event):
                self._fill_label(event)
                self._remove_on_clicks()
                print("Replace cell callback is removed")

        print("Added callback to replace cell")

    def _add_merge_callback(self):
        self._remove_on_clicks()
        QApplication.setOverrideCursor(Qt.CrossCursor)
        for layer in self.viewer.layers:

            @layer.mouse_drag_callbacks.append
            def _pick_merge_label(layer, event):
                id = self._read_label_id(event)
                self._remove_on_clicks()
                print("Merge step 1 callback is removed")
                QApplication.setOverrideCursor(Qt.CrossCursor)
                for layer in self.viewer.layers:

                    @layer.mouse_drag_callbacks.append
                    def _assimilate_label(layer, event):
                        self._replace_label(event, id)
                        self._remove_on_clicks()
                        print("Merge step 2 callback is removed")

                print("Added callback for merge step 2")

        print("Added callback for merge step 1")

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
        label_layer = grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            print("Tried to replace label but no label layer found")
            notify("Please make sure the label layer exists!")
            return

        x = int(round(event.position[2]))
        y = int(round(event.position[1]))
        z = int(round(event.position[0]))

        if id == -1:
            id = self._get_free_label_id(label_layer)

        # Replace the ID with the new id
        old_id = label_layer.data[z, y, x]
        print(old_id)
        """if old_id == 0:
            notify("Can't change ID of background, please make sure to select a cell!")
            return"""
        np.place(label_layer.data[z], label_layer.data[z] == old_id, id)

        # Refresh the layer
        label_layer.refresh()

        # set the label layer as currently selected layer
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(label_layer)
    
    def _fill_label(self, event):
        """
        Replaces the label at the given position with a new ID.
        Only affects pixels directly connected to the selected one
        """
        label_layer = grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            print("Tried to replace label but no label layer found")
            notify("Please make sure the label layer exists!")
            return

        x = int(round(event.position[2]))
        y = int(round(event.position[1]))
        z = int(round(event.position[0]))

        id = self._get_free_label_id(label_layer)
        
        # Replace the ID with the new id
        label_layer.fill(coord=(z,y,x), new_label=id, refresh=True)

        # set the label layer as currently selected layer
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(label_layer)

    def _read_label_id(self, event):
        """
        Reads the label id at the given position

        Parameters
        ----------
        position : list
            position of the mouse event

        Returns
        -------
        int
            id at the given position in the segmentation layer
        """
        label_layer = grab_layer(self.viewer, self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            print("Tried to replace label but no label layer found")
            notify("Please make sure the label layer exists!")
            return

        x = int(round(event.position[2]))
        y = int(round(event.position[1]))
        z = int(round(event.position[0]))

        return label_layer.data[z, y, x]

    def _print_on_clicks(self):
        for layer in self.viewer.layers:
            print("layer: {}".format(layer))
            for callback in layer.mouse_drag_callbacks:
                print(callback)

    def _remove_on_clicks(self):
        for layer in self.viewer.layers:
            layer.mouse_drag_callbacks = []
        QApplication.restoreOverrideCursor()
