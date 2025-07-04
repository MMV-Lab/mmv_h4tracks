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
    QProgressBar,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap

from pathlib import Path
import numpy as np
import copy
import cv2
from napari.layers.image.image import Image
from napari.layers.labels.labels import Labels
from napari.layers.tracks.tracks import Tracks

from ._assistant import AssistantWindow
from ._analysis import AnalysisWindow
from ._evaluation import EvaluationWindow

from ._reader import open_dialog, napari_get_reader
from ._segmentation import SegmentationWindow
from ._tracking import TrackingWindow
from ._writer import save_ome_zarr
from ._grabber import grab_layer
from ._utils import CallbackHandler


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
        self.callback_handler = CallbackHandler(self)

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
        self.combobox_segmentation.currentTextChanged.connect(
            self.update_evaluation_limits
        )
        self.combobox_tracks = QComboBox()
        self.layer_comboboxes = [
            self.combobox_image,
            self.combobox_segmentation,
            self.combobox_tracks,
        ]

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Dummy Loading %p%")
        self.progress_bar.setMaximum(1)
        # self.progress_bar.setValue(42)

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
        self.assistant_window = AssistantWindow(self)
        tabwidget.addTab(self.assistant_window, "Assistant")

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
        widget.layout().addWidget(self.progress_bar, 12, 0, 1, -1)

        # Scrollarea allows content to be larger than the assigned space (small monitor)
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        self.setMinimumWidth(540)
        self.setMinimumHeight(900)

        custom_binds = [
            ("W", self.hotkey_next_free),
            ("G", self.hotkey_overlap_single_tracking),
            ("H", self.hotkey_separate),
            ("Q", self.hotkey_select_id),
        ]
        for custom_bind in custom_binds:
            old_bind = viewer.bind_key(*custom_bind, overwrite=True)
            if old_bind is not None and old_bind.__name__ != custom_bind[1].__name__:
                print(old_bind)
                print(custom_bind)
                viewer.bind_key(custom_bind[0], old_bind, overwrite=True)
                raise ValueError(f"Hotkey {custom_bind[0]} already in use")

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

    def hotkey_separate(self, _):
        """
        Hotkey for separate
        """
        self.segmentation_window._add_replace_callback()

    def hotkey_select_id(self, _):
        """
        Hotkey for select ID
        """
        self.segmentation_window._add_select_callback()

    def update_evaluation_limits(self, event):
        """
        Updates the limits for the evaluation window
        """
        self.evaluation_window.update_limits(event)

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

    def _clear_layers(self, layer_names):
        """
        Prompts the user to clear layers with the given names.
        If the user confirms, the layers are removed from the viewer.

        returns
        -------
        bool
            True if the layers were cleared, False if the user canceled
        """
        for layer_name in layer_names:
            if layer_name in self.viewer.layers:
                msg = QMessageBox()
                msg.setWindowTitle("Layer already exists")
                msg.setText("Found layer with name " + layer_name)
                msg.setInformativeText(
                    "A layer with the name '"
                    + layer_name
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
                    return False

                # YesToAll -> Remove all layers with names in the file
                if ret == 32768:
                    for name in layer_names:
                        try:
                            self.viewer.layers.remove(name)
                        except ValueError:
                            pass

                # Yes -> Remove this layer
                self.viewer.layers.remove(layer_name)
        return True

    def _load_zarr(self, zarr_file):
        # check all layer names
        layernames = [
            layer.name for layer in self.viewer.layers if isinstance(layer, (Image, Labels, Tracks))
        ]
        if not self._clear_layers(layernames):
            # user canceled the operation
            return

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

        self.initial_layers = [
            copy.deepcopy(segmentation),
            copy.deepcopy(filtered_tracks),
        ]
        self.combobox_image.setCurrentText("Raw Image")
        self.combobox_segmentation.setCurrentText("Segmentation Data")
        self.combobox_tracks.setCurrentText("Tracks")

    def _load_ome_zarr(self, zarr_file):
        generic_metadata = dict(zarr_file.attrs)
        try:
            labels_metadata = dict(zarr_file.get("labels").attrs)
        except AttributeError:
            print("No labels found in OME-Zarr file.")
            return
        label_name = labels_metadata.get("labels", "Tracked Cells")[0]
        # read raw image
        raw_image = zarr_file.get("0")
        try:
            img_metadata = dict(raw_image.attrs)
        except AttributeError:
            print("No image metadata found in OME-Zarr file.")
            return
        # read segmentation
        segmentation = zarr_file.get(f"labels/{label_name}/0")
        try:
            seg_metadata = dict(zarr_file.get(f"labels/{label_name}").attrs)
        except AttributeError:
            seg_metadata = dict()
        # check if tracks are implied
        filtered_tracks = None
        if "implied_tracks" in seg_metadata and seg_metadata["implied_tracks"]:
            # if so: generate tracks from segmentation
            seg_data = segmentation[:]
            tracks = np.array(
                [
                    [seg_id, t, *np.round(np.mean(np.argwhere(seg_data[t] == seg_id), axis=0)).astype(int)]
                    for t in range(seg_data.shape[0])
                    for seg_id in np.unique(seg_data[t])[np.unique(seg_data[t]) != 0]
                ],
                dtype=np.int64,
            )
            # filter tracks to exclude single slice tracks
            count_of_track_ids = np.unique(tracks[:, 0], return_counts=True)
            filtered_track_ids = np.delete(
                count_of_track_ids, count_of_track_ids[1] == 1, 1
            )
            filtered_tracks = np.delete(
                tracks,
                np.where(np.isin(tracks[:, 0], filtered_track_ids[0, :], invert=True)),
                0,
            )
        # clear layers if they exist
        layernames = ["Raw Image", "Segmentation Data"]
        if filtered_tracks is not None:
            layernames.append("Tracks")
        if not self._clear_layers(layernames):
            # user canceled the operation
            return
        # add layers to viewer
        raw_layer = self.viewer.add_image(raw_image[:], name="Raw Image")
        self.viewer.add_labels(segmentation[:], name="Segmentation Data")
        if filtered_tracks is not None:
            self.viewer.add_tracks(filtered_tracks, name="Tracks")
        # add metadata to layers
        frames = generic_metadata.get("frames", None)
        unit = next(
            (ax.get("unit", "unit") for ax in img_metadata["multiscales"][0]["axes"] if ax["name"] in ["z", "y", "x"]),
            "unit"
        )
        raw_layer.metadata["raw_metadata"] = f"frames={frames}\nunit={unit}"
        # set initial layers
        self.initial_layers = [
            copy.deepcopy(segmentation),
            copy.deepcopy(filtered_tracks) if filtered_tracks is not None else None,
        ]

    def _load(self):
        """
        Opens a dialog for the user to choose an (ome-)zarr file to open. Passes the file for loading.
        """
        self.callback_handler.remove_callback_viewer()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        filepath = open_dialog(self)
        file_reader = napari_get_reader(filepath)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zarr_file, is_ome = file_reader(filepath)
        except TypeError:
            QApplication.restoreOverrideCursor()
            return

        if is_ome:
            self._load_ome_zarr(zarr_file)
        else:
            self._load_zarr(zarr_file)
        self.zarr = zarr_file
        self.file_interaction.setTitle(Path(filepath).name)
        QApplication.restoreOverrideCursor()

    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened or not all layers exist
        """
        # create new file if no file was opened
        # or the file is not an OME-zarr file
        if not hasattr(self, "zarr") or not "multiscales" in self.zarr.attrs:
            print("Calling save_as")
            self.save_as()
            return
        self.callback_handler.remove_callback_viewer()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.tracking_window.update_all_centroids()
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)

        layers = [raw_layer, segmentation_layer]

        self.assistant_window.align_ids_on_click(saving=True)
        save_ome_zarr(self.zarr, layers)
        QApplication.restoreOverrideCursor()

    def save_as(self):
        """
        Opens a dialog for the user to choose a zarr file to save to.
        Fails if not all layers exist
        """
        dialog = QFileDialog()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        path = f"{dialog.getSaveFileName()[0]}"
        # If user tries to save a zarr file without the .ome.zarr extension, we add it
        if path.endswith(".zarr") and not path.endswith(".ome.zarr"):
            path = path[: -len(".zarr")] + ".ome.zarr"
        if not path.endswith(".ome.zarr"):
            path += ".ome.zarr"
        if path == ".ome.zarr":
            QApplication.restoreOverrideCursor()
            return

        self.callback_handler.remove_callback_viewer()
        self.tracking_window.update_all_centroids()
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)

        layers = [raw_layer, segmentation_layer]

        self.assistant_window.align_ids_on_click(saving=True)
        save_ome_zarr(
            path,
            layers)
        QApplication.restoreOverrideCursor()

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

    def set_progress_range(self, min_: int, max_: int):
        """
        Sets the range of the progress bar

        Parameters
        ----------
        min : int
            The minimum value of the progress bar
        max : int
            The maximum value of the progress bar
        """
        self.progress_bar.setMinimum(min_)
        self.progress_bar.setMaximum(max_)
        self.progress_bar.setValue(min_)

    def set_progress_value(self, value: int):
        """
        Sets the value of the progress bar

        Parameters
        ----------
        value : int
            The value of the progress bar
        """
        self.progress_bar.setValue(value)

    def set_progress_text(self, text: str):
        """
        Sets the text of the progress bar

        Parameters
        ----------
        text : str
            The text of the progress bar
        """
        self.progress_bar.setFormat(text + " %p%")
        self.progress_bar.setTextVisible(True)
