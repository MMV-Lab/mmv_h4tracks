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
from qtpy.QtCore import Qt, QTimer
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
from ._logger import choice_dialog, notify

from ._reader import (
    open_dialog,
    napari_get_reader,
    check_multiscale_image,
    load_zarr_data,
    load_ome_zarr_data,
)
from ._segmentation import SegmentationWindow
from ._tracking import TrackingWindow, calculate_medoid
from ._writer import save_ome_zarr, save_zarr
from ._grabber import grab_layer
from ._session_trained_models import (
    remove_entries_for_layer,
    update_layer_name_in_store,
    register_session_trained_model as _register_session_trained_model_store,
)
from ._utils import CallbackHandler

import mmv_h4tracks._processing as mmv_processing


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
        # cache the segmentation since it was last aligned
        self.align_cache = None
        # cache segmentation&tracks since creation
        self.eval_cache = [None, None]
        self.callback_handler = CallbackHandler(self)
        # track if original raw data was multiscale
        self.is_multiscale = False
        # session Cellpose models: display_name -> {training_masks_dir, layer_prefix, frames}
        self.session_trained_models: dict[str, dict] = {}
        self._session_trained_layer_ids_hooked: set[int] = set()

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
            ("ctrl+T", self.create_implicit_tracks_wrapper),
            ("ctrl+A", self.hotkey_display_all),
            ("ctrl+L", self.hotkey_load_lineage_file)
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
            self._session_trained_attach_layer(layer)
        for combobox in self.layer_comboboxes:
            if combobox.count() == 0:
                combobox.addItem("")
        self.viewer.layers.events.moving.connect(self.reorder_entry_in_comboboxes)

        QTimer.singleShot(
            0,
            lambda: mmv_processing.scan_mmvh4tracks_training_temp_on_startup(self),
        )

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

    def hotkey_display_all(self, _):
        """
        Hotkey for displaying all tracks
        """
        self.tracking_window.show_all_tracks_on_click()

    def hotkey_load_lineage_file(self, _):
        """
        Hotkey for loading and parsing a lineage file.
        Checks for tracks layer selection, opens a file dialog filtered to .txt files,
        reads the lineage file, parses each line as track_id, start_t, end_t, parent_id,
        validates track_ids exist in tracks layer, and updates the tracks layer graph attribute.
        """
        # Check if tracks layer is selected
        tracks_name = self.combobox_tracks.currentText()
        if not tracks_name:
            print("Error: No tracks layer selected. Please select a tracks layer first.")
            return
        
        # Get tracks layer
        try:
            tracks_layer = grab_layer(self.viewer, tracks_name)
        except ValueError:
            print(f"Error: Tracks layer '{tracks_name}' not found.")
            return
        
        # Open file dialog filtered to .txt files
        retval = QFileDialog().getOpenFileName(
            self, "Select Lineage File", "", "Text files (*.txt)"
        )
        filepath = retval[0]
        
        # Return early if user canceled
        if not filepath:
            return
        
        try:
            # Get unique track IDs from tracks layer for validation
            tracks_data = tracks_layer.data
            valid_track_ids = set(np.unique(tracks_data[:, 0]))
            
            # Initialize graph if it doesn't exist
            if not hasattr(tracks_layer, 'graph') or tracks_layer.graph is None:
                tracks_layer.graph = {}
            
            def parse_parent_id(parent_str):
                """
                Parse parent_id from string. Handles three formats:
                1. Plain integer: "4" -> [4]
                2. Single integer in brackets: "[4]" -> [4]
                3. Multiple integers in brackets: "[4,5,6]" -> [4,5,6]
                
                Parameters
                ----------
                parent_str : str
                    String representation of parent_id
                    
                Returns
                -------
                list[int]
                    List of parent IDs
                """
                parent_str = parent_str.strip()
                # Check if it's in brackets
                if parent_str.startswith('[') and parent_str.endswith(']'):
                    # Remove brackets and split by comma
                    inner = parent_str[1:-1].strip()
                    if not inner:
                        return []
                    # Split by comma and convert to int
                    return [int(x.strip()) for x in inner.split(',') if x.strip()]
                else:
                    # Plain integer
                    return [int(parent_str)]
            
            # Read and parse the lineage file
            lineage_entries = []
            skipped_count = 0
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Split by whitespace and take first 4 values
                    values = line.split()
                    if len(values) < 4:
                        print(f"Warning: Line {line_num}: Skipping line with < 4 values: {line}")
                        skipped_count += 1
                        continue
                    
                    try:
                        # Parse as track_id, start_t, end_t, parent_id
                        track_id = int(values[0])
                        start_t = int(values[1])
                        end_t = int(values[2])
                        
                        # Parse parent_id (can be plain int, [int], or [int,int,...])
                        # Handle case where parent_id might be split across multiple values
                        # (e.g., "[4," "5," "6]" if brackets have spaces)
                        parent_str = values[3]
                        # If parent_id starts with '[' but doesn't end with ']', 
                        # it might be split - collect until we find closing bracket
                        if parent_str.startswith('[') and not parent_str.endswith(']'):
                            idx = 4
                            # Collect all parts until we find the closing bracket
                            parts = [parent_str]
                            while idx < len(values) and not values[idx].endswith(']'):
                                parts.append(values[idx])
                                idx += 1
                            if idx < len(values):
                                parts.append(values[idx])
                            # Join without spaces to preserve bracket structure (e.g., "[4,5,6]")
                            parent_str = ''.join(parts)
                        
                        parent_ids = parse_parent_id(parent_str)
                        
                        # Validate track_id exists in tracks layer
                        if track_id not in valid_track_ids:
                            print(f"Warning: Line {line_num}: Track ID {track_id} not found in tracks layer. Skipping.")
                            skipped_count += 1
                            continue
                        
                        lineage_entries.append((track_id, start_t, end_t, parent_ids))
                    except ValueError as e:
                        print(f"Warning: Line {line_num}: Could not parse line: {line}. Error: {e}")
                        skipped_count += 1
                        continue
            
            # Update tracks layer graph attribute
            if lineage_entries:
                for track_id, start_t, end_t, parent_ids in lineage_entries:
                    # Set graph entry: track_id -> parent_ids (already a list)
                    tracks_layer.graph[track_id] = parent_ids
                
                print(f"Successfully loaded {len(lineage_entries)} lineage entries into tracks layer graph.")
                if skipped_count > 0:
                    print(f"Skipped {skipped_count} invalid or unmatched entries.")
            else:
                print("No valid lineage entries found in file.")
                
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
        except (IOError, OSError, ValueError) as e:
            print(f"Error reading file: {e}")

    def update_evaluation_limits(self, event):
        """
        Updates the limits for the evaluation window
        """
        self.evaluation_window.update_limits(event)

    def register_session_trained_model(
        self,
        display_name: str,
        training_masks_dir,
        layer_prefix: str,
        frames: tuple[int, ...],
    ) -> None:
        """Record a model trained this session (for optional frame-exclusion on predict)."""
        _register_session_trained_model_store(
            self.session_trained_models,
            display_name,
            training_masks_dir,
            layer_prefix,
            frames,
        )

    def _session_trained_attach_layer(self, layer) -> None:
        if id(layer) in self._session_trained_layer_ids_hooked:
            return
        self._session_trained_layer_ids_hooked.add(id(layer))
        layer.events.name.connect(self._session_trained_on_layer_renamed)

    def _session_trained_on_layer_renamed(self, event) -> None:
        old = getattr(event, "old", None)
        new = getattr(event, "new", None)
        if old is not None and new is not None:
            update_layer_name_in_store(self.session_trained_models, old, new)

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
        self._session_trained_attach_layer(event.value)

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
        remove_entries_for_layer(self.session_trained_models, event.value.name)
        self._session_trained_layer_ids_hooked.discard(id(event.value))

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

        # Load data from zarr file
        raw_levels, segmentation, filtered_tracks, self.is_multiscale = load_zarr_data(zarr_file)
        
        # Add layers to viewer
        contrast_limits = (
            float(raw_levels[0].min()),
            float(raw_levels[0].max()),
        )
        self.viewer.add_image(
            raw_levels,
            name="Raw Image",
            multiscale=True,
            contrast_limits=contrast_limits,
        )
        self.viewer.add_labels(segmentation, name="Segmentation Data")
        self.viewer.add_tracks(filtered_tracks, name="Tracks")

        # Set widget state
        self.align_cache = copy.deepcopy(segmentation)
        self.eval_cache = [
            copy.deepcopy(segmentation),
            copy.deepcopy(filtered_tracks),
        ]
        self.combobox_image.setCurrentText("Raw Image")
        self.combobox_segmentation.setCurrentText("Segmentation Data")
        self.combobox_tracks.setCurrentText("Tracks")

    def _load_ome_zarr(self, zarr_file, zarr_path=None):
        # Load data from OME-Zarr file
        try:
            raw_levels, segmentation, metadata, self.is_multiscale, tracks = load_ome_zarr_data(zarr_file, zarr_path=zarr_path)
        except ValueError as e:
            print(f"Error loading OME-Zarr file: {e}")
            return
        
        # Clear layers if they exist
        layernames = ["Raw Image", "Segmentation Data"]
        if not self._clear_layers(layernames):
            # user canceled the operation
            return
        
        # Add layers to viewer
        contrast_limits = (
            float(raw_levels[0].min()),
            float(raw_levels[0].max()),
        )
        raw_layer = self.viewer.add_image(
            raw_levels,
            name="Raw Image",
            multiscale=True,
            contrast_limits=contrast_limits,
        )
        self.viewer.add_labels(segmentation[:], name="Segmentation Data")

        # Load tracks from file if it exists, otherwise create implicit tracks if needed
        if tracks is not None:
            # Load tracks from tracks.npy file (without scale)
            self.viewer.add_tracks(tracks, name="Tracks")
            self.eval_cache[1] = copy.deepcopy(tracks)
            self.combobox_tracks.setCurrentText("Tracks")
        elif metadata.get("implied_tracks", False):
            # Only create implicit tracks if no tracks.npy exists
            filtered_tracks = self.create_implicit_tracks()
            
            # Check if raw image layer has a scale attribute and pass it to add_tracks
            scale = None
            try:
                if hasattr(raw_layer, 'scale') and isinstance(raw_layer.scale, np.ndarray):
                    scale = raw_layer.scale
            except (AttributeError, TypeError):
                pass
            
            if scale is not None:
                self.viewer.add_tracks(filtered_tracks, name="Tracks", scale=scale)
            else:
                self.viewer.add_tracks(filtered_tracks, name="Tracks")
            self.eval_cache[1] = copy.deepcopy(filtered_tracks)
        
        # Add metadata to layers
        raw_layer.metadata["raw_metadata"] = f"frames={metadata['frames']}\nunit={metadata['unit']}"
        
        # Set alignment cache
        self.align_cache = copy.deepcopy(segmentation)
        self.eval_cache[0] = copy.deepcopy(segmentation)

    def create_implicit_tracks(self):
        """
        Creates implicit tracks from the segmentation data.
        
        Returns
        -------
        np.ndarray
            Array of filtered tracks with shape (n_tracks, 4) where columns are
            [track_id, time, y, x]. Tracks that only exist in a single frame are filtered out.
        """
        segmentation_name = self.combobox_segmentation.currentText()
        seg_layer = grab_layer(self.viewer, segmentation_name)
        seg_data = seg_layer.data

        tracks = []

        for t in range(seg_data.shape[0]):
            frame = seg_data[t]
            
            # Convert to numpy array to handle dask arrays from OME-Zarr
            # This is critical for lazy-loaded data
            frame = np.asarray(frame)

            # extract unique labels, excluding background (0)
            labels = np.unique(frame)
            labels = labels[labels != 0]

            coords_all = np.argwhere(frame)

            for seg_id in labels:
                coords = coords_all[frame[tuple(coords_all.T)] == seg_id]

                centroid = np.round(np.mean(coords, axis=0)).astype(int)

                if (
                    0 <= centroid[0] < frame.shape[0] and
                    0 <= centroid[1] < frame.shape[1] and
                    frame[tuple(centroid)] == seg_id
                ):
                    final_coord = centroid
                else:
                    coords = np.argwhere(frame == seg_id)
                    final_coord = calculate_medoid(coords)

                tracks.append([seg_id, t, *final_coord])

        tracks = np.array(tracks, dtype=np.int64)
        count_of_track_ids = np.unique(tracks[:, 0], return_counts=True)
        filtered_track_ids = np.delete(
            count_of_track_ids, count_of_track_ids[1] == 1, 1
        )
        filtered_tracks = np.delete(
            tracks,
            np.where(np.isin(tracks[:, 0], filtered_track_ids[0, :], invert=True)),
            0,
        )
        return filtered_tracks

    def create_implicit_tracks_wrapper(self, _):
        """
        Wrapper function for creating implicit tracks and adding them to the viewer.
        Can be called with a hotkey.
        Ignored parameter _ is required for the hotkey binding.
        Hotkey call passes viewer.
        """
        if _ is not None:
            print("Secret unlocked!")
        if not self._clear_layers(["Tracks"]):
            # user canceled the operation
            return
        filtered_tracks = self.create_implicit_tracks()
        
        # Check if raw image layer has a scale attribute and pass it to add_tracks
        scale = None
        try:
            raw_name = self.combobox_image.currentText()
            raw_layer = grab_layer(self.viewer, raw_name)
            if raw_layer is not None and hasattr(raw_layer, 'scale'):
                scale_attr = raw_layer.scale
                if isinstance(scale_attr, np.ndarray):
                    scale = scale_attr
        except (ValueError, AttributeError):
            # If layer doesn't exist or doesn't have scale, continue without it
            pass
        
        if scale is not None:
            self.viewer.add_tracks(filtered_tracks, name="Tracks", scale=scale)
        else:
            self.viewer.add_tracks(filtered_tracks, name="Tracks")
        self.eval_cache[1] = copy.deepcopy(filtered_tracks)

    def _load(self):
        """
        Opens a dialog for the user to choose an (ome-)zarr file to open. Passes the file for loading.
        """
        self.callback_handler.remove_callback_viewer()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        filepath = open_dialog(self)
        if filepath.endswith(".ome.zarr"):
            from importlib.metadata import version
            if int(version("zarr").split(".")[0]) < 3:
                QApplication.restoreOverrideCursor()
                choice = choice_dialog(f"Installed zarr version is {version('zarr')}, we recommed at least version 3.0.8. Loading could irrecoverably alter the loaded file. Are you sure you want to continue?", [("Yes", QMessageBox.YesRole), ("No", QMessageBox.NoRole)])
                if choice == QMessageBox.NoRole:
                    return
                QApplication.setOverrideCursor(Qt.WaitCursor)
        file_reader = napari_get_reader(filepath)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zarr_file, is_ome = file_reader(filepath)
        except TypeError:
            QApplication.restoreOverrideCursor()
            return

        if is_ome:
            # Pass filepath in case store path inference fails
            self._load_ome_zarr(zarr_file, zarr_path=filepath)
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
        if self.tracking_window.cached_tracks is not None:
            notify("Some tracks are not displayed, not saving")
            return
        # create new file if no file was opened
        # or the file is not an OME-zarr file
        # if not hasattr(self, "zarr") or not "multiscales" in self.zarr.attrs:
        if not hasattr(self, "zarr"):
            print("Calling save_as")
            self.save_as()
            return
        if "multiscales" in self.zarr.attrs:
            notify("Currently not supporting Ome-zarr saving, saving as zarr file")
            self.save_as()
            return
        self.callback_handler.remove_callback_viewer()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # self.tracking_window.update_all_centroids()
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)

        # layers = [raw_layer, segmentation_layer]
        tracks_name = self.combobox_tracks.currentText()
        tracks_layer = grab_layer(self.viewer, tracks_name)
        layers = [raw_layer, segmentation_layer, tracks_layer]

        # self.assistant_window.align_ids_on_click(saving=True)
        save_zarr(self.zarr, layers)
        # save_ome_zarr(self.zarr, layers, is_multiscale=self.is_multiscale)
        QApplication.restoreOverrideCursor()

    def save_as(self):
        """
        Opens a dialog for the user to choose a zarr file to save to.
        Fails if not all layers exist
        """
        if self.tracking_window.cached_tracks is not None:
            notify("Some tracks are not displayed, not saving")
            return
        dialog = QFileDialog()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        path = f"{dialog.getSaveFileName()[0]}"
        # If user tries to save a zarr file without the .ome.zarr extension, we add it
        # if path.endswith(".zarr") and not path.endswith(".ome.zarr"):
        #     path = path[: -len(".zarr")] + ".ome.zarr"
        # if not path.endswith(".ome.zarr"):
        #     path += ".ome.zarr"
        # if path == ".ome.zarr":
        #     QApplication.restoreOverrideCursor()
        #     return
        if not path.endswith(".zarr"):
            path += ".zarr"
        if path == ".zarr":
            QApplication.restoreOverrideCursor()
            return

        self.callback_handler.remove_callback_viewer()
        # self.tracking_window.update_all_centroids()
        raw_name = self.combobox_image.currentText()
        raw_layer = grab_layer(self.viewer, raw_name)
        segmentation_name = self.combobox_segmentation.currentText()
        segmentation_layer = grab_layer(self.viewer, segmentation_name)
        tracks_name = self.combobox_tracks.currentText()
        tracks_layer = grab_layer(self.viewer, tracks_name)

        # layers = [raw_layer, segmentation_layer]
        layers = [raw_layer, segmentation_layer, tracks_layer]

        # self.assistant_window.align_ids_on_click(saving=True)
        # save_ome_zarr(
        #     path,
        #     layers,
        #     is_multiscale=self.is_multiscale)
        save_zarr(path, layers)
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


