import napari
import multiprocessing
import warnings

from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QGridLayout,
    QScrollArea,
    QMessageBox,
    QApplication,
    QFileDialog,
    QComboBox,
    QTabWidget,
    QSizePolicy,
    QHBoxLayout,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap

from pathlib import Path
import numpy as np
import copy
import cv2
import zarr
from napari.layers.image.image import Image
from napari.layers.labels.labels import Labels
from napari.layers.tracks.tracks import Tracks

from ._analysis import AnalysisWindow
from ._evaluation import EvaluationWindow
from ._logger import notify
from ._reader import open_dialog, napari_get_reader
from ._segmentation import SegmentationWindow
from ._tracking import TrackingWindow
from ._writer import save_zarr
from ._grabber import grab_layer


class MMVH4TRACKS(QWidget):
    """
    The main widget of our application

    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance
    zarr : file
        The zarr file the data was loaded from / will be saved to

    Methods
    -------
    load()
        Opens a dialog for the user to choose a zarr file (directory)
    save()
        Writes the changes made to the opened zarr file
    processing()
        Opens a window to run processing on the data
    segmentation()
        Opens a window to correct the segmentation
    tracking()
        Opens a window to correct the tracking
    analysis()
        Opens a window to do analysis
    """

    def __init__(self, viewer: napari.Viewer = None, parent=None):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        """
        super().__init__(parent=parent)
        viewer = napari.current_viewer() if viewer is None else viewer
        self.viewer = viewer
        self.initial_layers = [None, None]

        ### QObjects

        # Logo
        filename = "celltracking_logo.jpg"
        path = Path(__file__).parent / "ressources" / filename
        image = cv2.imread(str(path))
        height, width, _ = image.shape
        logo = QPixmap(
            QImage(image.data, width, height, 3 * width, QImage.Format_BGR888)
        )

        # Labels
        logo_label = QLabel()
        logo_label.setPixmap(logo)
        logo_label.setMaximumHeight(150)
        logo_label.setMaximumWidth(150)
        logo_label.setScaledContents(True)
        logo_label.setAlignment(Qt.AlignCenter)
        title = QLabel("<h1><font color='green'>MMV_H4Tracks</font></h1>")
        title.setMaximumHeight(100)
        label_image = QLabel("Image:")
        label_segmentation = QLabel("Segmentation:")
        label_tracks = QLabel("Tracks:")

        # Buttons
        btn_load = QPushButton("Load")
        btn_load.setToolTip("Load a Zarr file")
        btn_save = QPushButton("Save")
        btn_save.setToolTip("Overwrite the loaded Zarr file")
        btn_save_as = QPushButton("Save as")
        btn_save_as.setToolTip("Save as a new Zarr file")

        btn_load.clicked.connect(self._load)
        btn_save.clicked.connect(self._save)
        btn_save_as.clicked.connect(self.save_as)

        # Radio Buttons
        self.rb_eco = QRadioButton("Eco")
        rb_heavy = QRadioButton("Regular")
        rb_heavy.toggle()

        # Comboboxes
        self.combobox_image = QComboBox()
        self.combobox_segmentation = QComboBox()
        self.combobox_tracks = QComboBox()
        self.layer_comboboxes = [
            self.combobox_image,
            self.combobox_segmentation,
            self.combobox_tracks,
        ]

        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")
        line2 = QWidget()
        line2.setFixedHeight(4)
        line2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line2.setStyleSheet("background-color: #c0c0c0")

        # Spacers
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(0)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_2 = QWidget()
        h_spacer_2.setFixedHeight(4)
        h_spacer_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_3 = QWidget()
        h_spacer_3.setFixedHeight(0)
        h_spacer_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_4 = QWidget()
        h_spacer_4.setFixedHeight(4)
        h_spacer_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_5 = QWidget()
        h_spacer_5.setFixedHeight(4)
        h_spacer_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        computation_mode = QGroupBox("Computation mode")
        computation_mode_tooltip = (
            "Select how much of your computer's resources napari should use for CPU-computing.<br>"
            "<ul>"
            "<li> Eco: Up to 40%</li>"
            "<li> Regular: Up to 80%</li>"
            "</ul>"
        )
        computation_mode.setToolTip(computation_mode_tooltip)
        computation_mode.setLayout(QGridLayout())
        computation_mode.layout().addWidget(h_spacer_1, 0, 0, 1, -1)
        computation_mode.layout().addWidget(self.rb_eco, 1, 0)
        computation_mode.layout().addWidget(rb_heavy, 1, 1)
        self.file_interaction = QGroupBox()
        self.file_interaction.setLayout(QGridLayout())
        self.file_interaction.layout().addWidget(h_spacer_3, 0, 0, 1, -1)
        self.file_interaction.layout().addWidget(btn_load, 1, 0)
        self.file_interaction.layout().addWidget(btn_save, 1, 1)
        self.file_interaction.layout().addWidget(btn_save_as, 1, 2)

        # QTabwidget
        tabwidget = QTabWidget()
        self.segmentation_window = SegmentationWindow(self)
        tabwidget.addTab(self.segmentation_window, "Segmentation")
        self.tracking_window = TrackingWindow(self)
        tabwidget.addTab(self.tracking_window, "Tracking")
        self.analysis_window = AnalysisWindow(self)
        tabwidget.addTab(self.analysis_window, "Analysis")
        self.evaluation_window = EvaluationWindow(self)
        tabwidget.addTab(self.evaluation_window, "Evaluation")

        ### Organize objects via widgets
        # widget: parent widget of all content
        widget = QWidget()
        widget.setLayout(QGridLayout())
        widget.layout().addWidget(logo_label, 0, 0, 1, 2)
        widget.layout().addWidget(title, 0, 2)
        widget.layout().addWidget(computation_mode, 1, 0, 1, -1)
        widget.layout().addWidget(h_spacer_2, 2, 0, 1, -1)
        widget.layout().addWidget(self.file_interaction, 3, 0, 1, -1)
        widget.layout().addWidget(h_spacer_4, 4, 0, 1, -1)
        widget.layout().addWidget(line, 5, 0, 1, -1)
        widget.layout().addWidget(label_image, 6, 0)
        widget.layout().addWidget(self.combobox_image, 6, 1, 1, 2)
        widget.layout().addWidget(label_segmentation, 7, 0)
        widget.layout().addWidget(self.combobox_segmentation, 7, 1, 1, 2)
        widget.layout().addWidget(label_tracks, 8, 0)
        widget.layout().addWidget(self.combobox_tracks, 8, 1, 1, 2)
        widget.layout().addWidget(line2, 9, 0, 1, -1)
        widget.layout().addWidget(h_spacer_5, 10, 0, 1, -1)
        widget.layout().addWidget(tabwidget, 11, 0, 1, -1)

        # Scrollarea allows content to be larger than the assigned space (small monitor)
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        self.setMinimumWidth(540)
        self.setMinimumHeight(900)

        hotkeys = self.viewer.keymap.keys()
        custom_binds = [
            ("E", self.hotkey_next_free),
            ("S", self.hotkey_overlap_single_tracking),
        ]
        for custom_bind in custom_binds:
            if not custom_bind[0] in hotkeys:
                viewer.bind_key(*custom_bind)

        self.viewer.layers.events.inserted.connect(self.add_entry_to_comboboxes)
        self.viewer.layers.events.removed.connect(self.remove_entry_from_comboboxes)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.layer_comboboxes[0].addItem(layer.name)
            elif isinstance(layer, Labels):
                self.layer_comboboxes[1].addItem(layer.name)
            elif isinstance(layer, Tracks):
                self.layer_comboboxes[2].addItem(layer.name)

            layer.events.name.connect(
                self.rename_entry_in_comboboxes
            )  # doesn't contain index
        for combobox in self.layer_comboboxes:
            if combobox.count() == 0:
                combobox.addItem("")
        self.viewer.layers.events.moving.connect(self.reorder_entry_in_comboboxes)

    def hotkey_next_free(self, _):
        """
        Hotkey for the next free label id
        """
        label_layer = grab_layer(self.viewer, self.combobox_segmentation.currentText())
        self.segmentation_window._set_label_id()
        label_layer.mode = "paint"

    def hotkey_overlap_single_tracking(self, _):
        """
        Hotkey for the overlap single tracking
        """
        self.tracking_window._add_auto_track_callback()

    def add_entry_to_comboboxes(self, event):
        """
        Adds a new entry to the comboboxes for the layers
        """
        if isinstance(event.value, Image):
            combobox = self.layer_comboboxes[0]
        elif isinstance(event.value, Labels):
            combobox = self.layer_comboboxes[1]
        elif isinstance(event.value, Tracks):
            combobox = self.layer_comboboxes[2]
        else:
            return

        combobox.addItem(event.value.name)
        empty_index = combobox.findText("")
        if empty_index != -1:
            combobox.removeItem(empty_index)
        event.value.events.name.connect(
            self.rename_entry_in_comboboxes
        )  # contains index

    def remove_entry_from_comboboxes(self, event):
        """
        Removes an entry from the comboboxes for the layers
        """
        if isinstance(event.value, Image):
            combobox = self.layer_comboboxes[0]
        elif isinstance(event.value, Labels):
            combobox = self.layer_comboboxes[1]
        elif isinstance(event.value, Tracks):
            combobox = self.layer_comboboxes[2]
        else:
            return
        index = combobox.findText(event.value.name)
        combobox.removeItem(index)
        if combobox.count() == 0:
            combobox.addItem("")

    def rename_entry_in_comboboxes(self, event):
        """
        Renames an entry in the comboboxes for the layers
        """
        if not hasattr(event, "index"):
            event.index = self.viewer.layers.index(event.source.name)
        layer = self.viewer.layers[event.index]
        if isinstance(layer, Image):
            combobox = self.layer_comboboxes[0]
        elif isinstance(layer, Labels):
            combobox = self.layer_comboboxes[1]
        elif isinstance(layer, Tracks):
            combobox = self.layer_comboboxes[2]
        else:
            return
        current_index = combobox.currentIndex()
        entries = [
            self.viewer.layers[i].name
            for i in range(len(self.viewer.layers))
            if isinstance(self.viewer.layers[i], layer.__class__)
        ]
        combobox.clear()
        combobox.addItem("")
        combobox.addItems(entries)
        combobox.setCurrentIndex(current_index)

    def reorder_entry_in_comboboxes(self, event):
        """
        Reorders an entry in the comboboxes for the layers
        """
        if event.index < event.new_index:
            target_index = event.new_index - 1
            low_index, high_index = event.index, target_index
        else:
            target_index = event.new_index
            low_index, high_index = event.new_index + 1, event.index + 1

        layer = self.viewer.layers[target_index]
        if isinstance(layer, Image):
            combobox = self.layer_comboboxes[0]
            layertype = Image
        elif isinstance(layer, Labels):
            layertype = Labels
            combobox = self.layer_comboboxes[1]
        elif isinstance(layer, Tracks):
            layertype = Tracks
            combobox = self.layer_comboboxes[2]
        else:
            return

        change = 0
        for layer in self.viewer.layers[low_index:high_index]:
            if isinstance(layer, layertype):
                change += 1
        if change == 0:
            return

        current_index = combobox.findText(layer.name)
        if event.index < event.new_index:
            new_index = current_index + change
        else:
            new_index = current_index - change
        current_item = combobox.currentText()
        combobox.removeItem(current_index)
        combobox.insertItem(new_index, layer.name)
        index = combobox.findText(current_item)
        combobox.setCurrentIndex(index)

    def _load(self):
        """
        Opens a dialog for the user to choose a zarr file to open. Checks if any layernames are blocked
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        filepath = open_dialog(self)
        file_reader = napari_get_reader(filepath)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zarr_file = file_reader(filepath)
        except TypeError:
            QApplication.restoreOverrideCursor()
            return

        # check all layer names
        for layername in zarr_file.__iter__():
            if layername in self.viewer.layers:
                msg = QMessageBox()
                msg.setWindowTitle("Layer already exists")
                msg.setText("Found layer with name " + layername)
                msg.setInformativeText(
                    "A layer with the name '"
                    + layername
                    + "' exists already."
                    + " Do you want to delete this layer to proceed?"
                )
                msg.addButton(QMessageBox.Yes)
                msg.addButton(QMessageBox.YesToAll)
                msg.addButton(QMessageBox.Cancel)
                ret = msg.exec()  # Yes -> 16384, YesToAll -> 32768, Cancel -> 4194304

                # Cancel
                if ret == 4194304:
                    QApplication.restoreOverrideCursor()
                    return

                # YesToAll -> Remove all layers with names in the file
                if ret == 32768:
                    for name in zarr_file.__iter__():
                        try:
                            self.viewer.layers.remove(name)
                        except ValueError:
                            pass
                    break

                # Yes -> Remove this layer
                self.viewer.layers.remove(layername)

        # add layers to viewer
        self.viewer.add_image(zarr_file["raw_data"][:], name="Raw Image")
        segmentation = zarr_file["segmentation_data"][:]

        self.viewer.add_labels(segmentation, name="Segmentation Data")
        # save tracks so we can delete one slice tracks first
        tracks = zarr_file["tracking_data"][:]
        # Filter track ids of tracks that just occur once
        count_of_track_ids = np.unique(tracks[:, 0], return_counts=True)
        filtered_track_ids = np.delete(
            count_of_track_ids, count_of_track_ids[1] == 1, 1
        )

        # Remove tracks that only exist in one slice
        filtered_tracks = np.delete(
            tracks,
            np.where(np.isin(tracks[:, 0], filtered_track_ids[0, :], invert=True)),
            0,
        )
        self.viewer.add_tracks(filtered_tracks, name="Tracks")

        self.zarr = zarr_file
        self.initial_layers = [
            copy.deepcopy(segmentation),
            copy.deepcopy(filtered_tracks),
        ]
        self.combobox_image.setCurrentText("Raw Image")
        self.combobox_segmentation.setCurrentText("Segmentation Data")
        self.combobox_tracks.setCurrentText("Tracks")
        self.file_interaction.setTitle(Path(filepath).name)
        QApplication.restoreOverrideCursor()

    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened or not all layers exist
        """
        if not hasattr(self, "zarr"):
            self.save_as()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)
        track_name = self.combobox_tracks.currentText()
        track_layer = grab_layer(self.viewer, track_name)
        layers = [raw_layer, segmentation_layer, track_layer]
        save_zarr(self.zarr, layers, self.tracking_window.cached_tracks)

    def save_as(self):
        """
        Opens a dialog for the user to choose a zarr file to save to.
        Fails if not all layers exist
        """
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        raw_data = grab_layer(self.viewer, raw_name).data
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)
        segmentation_data = grab_layer(self.viewer, segmentation_name).data
        tracks_name = self.combobox_tracks.currentText()
        tracks_layer = grab_layer(self.viewer, tracks_name)
        track_data = grab_layer(self.viewer, tracks_name).data

        dialog = QFileDialog()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        path = f"{dialog.getSaveFileName()[0]}"
        if not path.endswith(".zarr"):
            path += ".zarr"
        if path == ".zarr":
            return
        layers = [raw_layer, segmentation_layer, tracks_layer]

        zarrfile = zarr.open(path, mode="w")
        save_zarr(zarrfile, layers, self.tracking_window.cached_tracks)

    def get_process_limit(self):
        """
        Returns the number of processes to use for computation

        Returns
        -------
        int
            The number of processes to use for computation
        """
        if self.rb_eco.isChecked():
            return max(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            return max(1, int(multiprocessing.cpu_count() * 0.8))
