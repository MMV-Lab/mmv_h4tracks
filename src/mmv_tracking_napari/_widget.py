import napari

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
from ._logger import setup_logging, notify
from ._processing import ProcessingWindow
from ._reader import open_dialog, napari_get_reader
from ._segmentation import SegmentationWindow
from ._tracking import TrackingWindow
from ._writer import save_zarr
from ._grabber import grab_layer


class MMVTracking(QWidget):
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

    dock = None

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
        MMVTracking.dock = self

        # setup_logging()

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
        title = QLabel("<h1><font color='green'>HITL4Trk</font></h1>")
        title.setMaximumHeight(100)
        # self.loaded_file_name = QLabel()
        # computation_mode = QLabel("Computation mode")
        # computation_mode.setMaximumHeight(20)
        label_image = QLabel("Image:")
        label_segmentation = QLabel("Segmentation:")
        label_tracks = QLabel("Tracks:")

        # Buttons
        btn_load = QPushButton("Load")
        btn_load.setToolTip("Load a Zarr file")
        btn_save = QPushButton("Save")
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
        self.combobox_image.addItem("")
        self.combobox_segmentation = QComboBox()
        self.combobox_segmentation.addItem("")
        self.combobox_tracks = QComboBox()
        self.combobox_tracks.addItem("")
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
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(4)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        computation_mode = QGroupBox("Computation mode")
        computation_mode.setLayout(QHBoxLayout())
        computation_mode.layout().addWidget(self.rb_eco)
        computation_mode.layout().addWidget(rb_heavy)
        self.file_interaction = QGroupBox()
        self.file_interaction.setLayout(QHBoxLayout())
        self.file_interaction.layout().addWidget(btn_load)
        self.file_interaction.layout().addWidget(btn_save)
        self.file_interaction.layout().addWidget(btn_save_as)

        # QTabwidget
        tabwidget = QTabWidget()
        # self.processing_window = ProcessingWindow(self)
        # tabwidget.addTab(self.processing_window, "Data processing")
        self.segmentation_window = SegmentationWindow(self)
        tabwidget.addTab(self.segmentation_window, "Segmentation")
        self.tracking_window = TrackingWindow(self)
        tabwidget.addTab(self.tracking_window, "Tracking")
        self.analysis_window = AnalysisWindow(self)
        tabwidget.addTab(self.analysis_window, "Analysis")

        ### Organize objects via widgets
        # widget: parent widget of all content
        widget = QWidget()
        widget.setLayout(QGridLayout())
        widget.layout().addWidget(logo_label, 0, 0, 1, 2)
        widget.layout().addWidget(title, 0, 2)
        widget.layout().addWidget(computation_mode, 1, 0, 1, -1)
        widget.layout().addWidget(self.file_interaction, 2, 0, 1, -1)
        # widget.layout().addWidget(self.loaded_file_name, 3, 0, 1, 2)
        # widget.layout().addWidget(h_spacer_1, 3, 2, 1, -1)
        # widget.layout().addWidget(btn_load, 4, 0)
        # widget.layout().addWidget(btn_save, 4, 1)
        # widget.layout().addWidget(btn_save_as, 4, 2)
        widget.layout().addWidget(line, 5, 0, 1, -1)
        widget.layout().addWidget(label_image, 6, 0)
        widget.layout().addWidget(self.combobox_image, 6, 1, 1, 2)
        widget.layout().addWidget(label_segmentation, 7, 0)
        widget.layout().addWidget(self.combobox_segmentation, 7, 1, 1, 2)
        widget.layout().addWidget(label_tracks, 8, 0)
        widget.layout().addWidget(self.combobox_tracks, 8, 1, 1, 2)
        widget.layout().addWidget(line2, 9, 0, 1, -1)
        widget.layout().addWidget(tabwidget, 10, 0, 1, -1)

        # Scrollarea allows content to be larger than the assigned space (small monitor)
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setMinimumSize(250, 300)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)
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
        self.viewer.layers.events.moving.connect(self.reorder_entry_in_comboboxes)

    def add_entry_to_comboboxes(self, event):
        if isinstance(event.value, Image):
            self.layer_comboboxes[0].addItem(event.value.name)
        elif isinstance(event.value, Labels):
            self.layer_comboboxes[1].addItem(event.value.name)
        elif isinstance(event.value, Tracks):
            self.layer_comboboxes[2].addItem(event.value.name)
        else:
            return

        event.value.events.name.connect(
            self.rename_entry_in_comboboxes
        )  # contains index

    def remove_entry_from_comboboxes(self, event):
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

    def rename_entry_in_comboboxes(self, event):
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

    """def apply_on_clicks(self, event):
        for on_click in self.on_clicks:
            layer = event.value
            
            @layer.mouse_drag_callbacks.append
            on_click"""

    def _load(self):
        """
        Opens a dialog for the user to choose a zarr file to open. Checks if any layernames are blocked
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        print("Opening dialog")
        filepath = open_dialog(self)
        print("Dialog is closed, retrieving reader")
        file_reader = napari_get_reader(filepath)
        print("Got '{}' as file reader".format(file_reader))
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("Reading file")
                zarr_file = file_reader(filepath)
                print("File has been read")
        except TypeError:
            print("Could not read file")
            QApplication.restoreOverrideCursor()
            return

        # check all layer names
        for layername in zarr_file.__iter__():
            if layername in self.viewer.layers:
                print("Detected layer with name {}".format(layername))
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
                    print("Loading cancelled")
                    QApplication.restoreOverrideCursor()
                    return

                # YesToAll -> Remove all layers with names in the file
                if ret == 32768:
                    print("Removing all layers with names in zarr from viewer")
                    for name in zarr_file.__iter__():
                        try:
                            self.viewer.layers.remove(name)
                        except ValueError:
                            pass
                    break

                # Yes -> Remove this layer
                print("removing layer {}".format(layername))
                self.viewer.layers.remove(layername)

        print("Adding layers")
        # add layers to viewer
        # try:
        self.viewer.add_image(zarr_file["raw_data"][:], name="Raw Image")
        segmentation = zarr_file["segmentation_data"][:]

        self.viewer.add_labels(segmentation, name="Segmentation Data")
        # save tracks so we can delete one slice tracks first
        tracks = zarr_file["tracking_data"][:]
        """except:
            print(
                "File does not have the right structure of raw_data, segmentation_data and tracking_data!"
            )
        else:"""
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

        print("Layers have been added")

        self.zarr = zarr_file
        self.tracks = filtered_tracks
        self.initial_layers = [
            copy.deepcopy(segmentation),
            copy.deepcopy(filtered_tracks),
        ]
        self.combobox_image.setCurrentText("Raw Image")
        self.combobox_segmentation.setCurrentText("Segmentation Data")
        self.combobox_tracks.setCurrentText("Tracks")
        self.file_interaction.setTitle(Path(filepath).name)
        # self.loaded_file_name.setText(Path(filepath).name)
        QApplication.restoreOverrideCursor()

    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened or not all layers exist
        """
        if not hasattr(self, "zarr"):
            self.save_as()
            return
        raw_data = self.combobox_image.currentText()
        """raw_data = layer_select(self, "Raw Image")
        if not raw_data[1]:
            return
        raw_data = raw_data[0]"""
        raw_layer = grab_layer(self.viewer, raw_data)
        segmentation_data = self.combobox_segmentation.currentText()
        """segmentation_data = layer_select(self, "Segmentation Data")
        if not segmentation_data[1]:
            return
        segmentation_data = segmentation_data[0]"""
        segmentation_layer = grab_layer(self.viewer, segmentation_data)
        track_data = self.combobox_tracks.currentText()
        """track_data = layer_select(self, "Tracks")
        if not track_data[1]:
            return
        track_data = track_data[0]"""
        track_layer = grab_layer(self.viewer, track_data)
        layers = [raw_layer, segmentation_layer, track_layer]
        save_zarr(self, self.zarr, layers, self.tracks)

    def save_as(self):
        raw_name = self.combobox_image.currentText()
        """raw = layer_select(self, "Raw Image")
        if not raw[1]:
            return
        raw_name= raw[0]"""
        raw_data = grab_layer(self.viewer, raw_name).data
        segmentation_name = self.combobox_segmentation.currentText()
        """segmentation = layer_select(self, "Segmentation Data")
        if not segmentation[1]:
            return
        segmentation_name = segmentation[0]"""
        segmentation_data = grab_layer(self.viewer, segmentation_name).data
        tracks_name = self.combobox_tracks.currentText()
        """tracks = layer_select(self, "Tracks")
        if not tracks[1]:
            return
        track_name = tracks[0]"""
        track_data = grab_layer(self.viewer, tracks_name).data

        dialog = QFileDialog()
        path = f"{dialog.getSaveFileName()[0]}"
        if not path.endswith(".zarr"):
            path += ".zarr"
        if path == ".zarr":
            return
        print(path)
        z = zarr.open(path, mode="w")
        r = z.create_dataset(
            "raw_data", shape=raw_data.shape, dtype="f8", data=raw_data
        )
        s = z.create_dataset(
            "segmentation_data",
            shape=segmentation_data.shape,
            dtype="i4",
            data=segmentation_data,
        )
        t = z.create_dataset(
            "tracking_data", shape=track_data.shape, dtype="i4", data=track_data
        )

    @napari.Viewer.bind_key("Shift-r")
    def _hotkey_remove_label(self):
        if hasattr(MMVTracking.dock, "segmentation_window"):
            MMVTracking.dock.segmentation_window._add_remove_callback()

    @napari.Viewer.bind_key("Shift-l")
    def _hotkey_load_label(self):
        if hasattr(MMVTracking.dock, "segmentation_window"):
            MMVTracking.dock.segmentation_window._set_label_id()

    @napari.Viewer.bind_key("Shift-c")
    def _hotkey_separate_label(self):
        if hasattr(MMVTracking.dock, "segmentation_window"):
            MMVTracking.dock.segmentation_window._add_replace_callback()

    @napari.Viewer.bind_key("Shift-m")
    def _hotkey_merge_label(self):
        if hasattr(MMVTracking.dock, "segmentation_window"):
            MMVTracking.dock.segmentation_window._add_merge_callback()

    @napari.Viewer.bind_key("Shift-p")
    def _hotkey_select_label(self):
        if hasattr(MMVTracking.dock, "segmentation_window"):
            MMVTracking.dock.segmentation_window._add_select_callback()

    @napari.Viewer.bind_key("Shift-u")
    def _hotkey_unlink_track(self):
        if hasattr(MMVTracking.dock, "tracking_window"):
            MMVTracking.dock.tracking_window._unlink()

    @napari.Viewer.bind_key("l")
    def _hotkey_link_track(self):
        if hasattr(MMVTracking.dock, "tracking_window"):
            MMVTracking.dock.tracking_window._link()

    @napari.Viewer.bind_key("Shift-t")
    def _hotkey_track_single(self):
        if hasattr(MMVTracking.dock, "tracking_window"):
            MMVTracking.dock.tracking_window._add_auto_track_callback()
