import napari

from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QScrollArea,
    QMessageBox,
    QApplication,
    QFileDialog,
)
from qtpy.QtCore import Qt

import numpy as np
import copy
import zarr

from ._analysis import AnalysisWindow
from ._logger import setup_logging, notify, layer_select
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

        # setup_logging() TODO: re-enable

        ### QObjects

        # Labels
        title = QLabel("<font color='green'>HITL4Trk</font>")
        computation_mode = QLabel("Computation mode")

        # Buttons
        btn_load = QPushButton("Load")
        btn_load.setToolTip("Load a Zarr file")
        btn_save = QPushButton("Save")
        btn_save_as = QPushButton("Save as")
        btn_save_as.setToolTip("Save as a new Zarr file")
        btn_processing = QPushButton("Data Processing")
        btn_segmentation = QPushButton("Segmentation correction")
        btn_tracking = QPushButton("Tracking correction")
        btn_analysis = QPushButton("Analysis")

        btn_load.clicked.connect(self._load)
        btn_save.clicked.connect(self._save)
        btn_save_as.clicked.connect(self.save_as)
        btn_processing.clicked.connect(self._processing)
        btn_segmentation.clicked.connect(self._segmentation)
        btn_tracking.clicked.connect(self._tracking)
        btn_analysis.clicked.connect(self._analysis)

        # Radio Buttons
        self.rb_eco = QRadioButton("Eco")
        # self.rb_eco.toggle()
        rb_heavy = QRadioButton("Regular")
        rb_heavy.toggle()

        ### Organize objects via widgets
        # widget: parent widget of all content
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(title)

        computation_mode_rbs = QWidget()
        computation_mode_rbs.setLayout(QGridLayout())
        computation_mode_rbs.layout().addWidget(computation_mode, 0, 0, 1, 2)
        computation_mode_rbs.layout().addWidget(self.rb_eco, 1, 0)
        computation_mode_rbs.layout().addWidget(rb_heavy, 1, 1)

        widget.layout().addWidget(computation_mode_rbs)

        read_write_files = QWidget()
        read_write_files.setLayout(QHBoxLayout())
        read_write_files.layout().addWidget(btn_load)
        read_write_files.layout().addWidget(btn_save)
        read_write_files.layout().addWidget(btn_save_as)

        widget.layout().addWidget(read_write_files)

        processing = QWidget()
        processing.setLayout(QVBoxLayout())
        processing.layout().addWidget(btn_processing)
        processing.layout().addWidget(btn_segmentation)
        processing.layout().addWidget(btn_tracking)
        processing.layout().addWidget(btn_analysis)

        widget.layout().addWidget(processing)

        # Scrollarea allows content to be larger than the assigned space (small monitor)
        scroll_area = QScrollArea()
        scroll_area.setWidget(widget)
        scroll_area.setWidgetResizable(True)

        self.setMinimumSize(250, 300)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)
        """self.viewer.layers.events.inserting.connect(self.ev_inserting)
        self.viewer.layers.events.inserted.connect(self.ev_inserted)
        self.viewer.layers.events.removing.connect(self.ev_removing)
        self.viewer.layers.events.removed.connect(self.ev_removed)
        self.viewer.layers.events.moving.connect(self.ev_moving)
        self.viewer.layers.events.moved.connect(self.ev_moved)
        self.viewer.layers.events.changed.connect(self.ev_changed)
        self.viewer.layers.events.reordered.connect(self.ev_reordered)
        self.viewer.layers.selection.events.changed.connect(self.ev_selection_events_changed)
        self.viewer.layers.selection.events.active.connect(self.ev_selection_events_active)
        
    def ev_inserting(self, event):
        pass
        #print(f'### emitted "inserting" with index {event.index}')
    def ev_inserted(self, event):
        event.value.events.visible.connect(self.shaco)
        event.value.events.name.connect(self.namechange_is_13500_ip)
        #print(f'### emitted "inserted" with index {event.index} and value {event.value}')
    def ev_removing(self, event):
        pass
        #print(f'### emitted "removing" with index {event.index}')
    def ev_removed(self, event):
        print(f'### emitted "removed" with index {event.index} and value {event.value}')
    def ev_moving(self, event):
        pass
        #print(f'### emitted "moving" with index {event.index} and new index {event.new_index}')
    def ev_moved(self, event):
        pass
        #print(f'### emitted "moved" with index {event.index}, new index {event.new_index} and value {event.value}')
    def ev_changed(self, event):
        print(f'### emitted "changed" with index {event.index}, old value {event.old_value} and value {event.value}')
    def ev_reordered(self, event):
        pass
        #print(f'### emitted "reordered" with value {event.value}')
    def ev_selection_events_changed(self, event):
        pass
        #print(f'### emitted "selection.events.changed" with added {event.added} and removed {event.removed}')
    def ev_selection_events_active(self, event):
        pass
        #print(f'### emitted "selection.events.active" with value {event.value}')
        
    def shaco(self, event):
        print(f"now you see me, now you don't!")
    def namechange_is_13500_ip(self, event):
        print(f"index: {event.index}, source: {event.source}, type: {event.type}")"""

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
        try:
            self.viewer.add_image(zarr_file["raw_data"][:], name="Raw Image")
            segmentation = zarr_file["segmentation_data"][:]

            self.viewer.add_labels(segmentation, name="Segmentation Data")
            # save tracks so we can delete one slice tracks first
            tracks = zarr_file["tracking_data"][:]
        except:
            print(
                "File does not have the right structure of raw_data, segmentation_data and tracking_data!"
            )
        else:
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
        self.initial_layers = [copy.deepcopy(segmentation), copy.deepcopy(filtered_tracks)]
        QApplication.restoreOverrideCursor()

    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened or not all layers exist
        """
        if not hasattr(self, "zarr"):
            self.save_as()
            return
        raw_data = layer_select(self, "Raw Image")
        if not raw_data[1]:
            return
        raw_data = raw_data[0]
        segmentation_data = layer_select(self, "Segmentation Data")
        if not segmentation_data[1]:
            return
        segmentation_data = segmentation_data[0]
        track_data = layer_select(self, "Tracks")
        if not track_data[1]:
            return
        track_data = track_data[0]
        layers = [raw_layer, segmentation_layer, track_layer]
        save_zarr(self, self.zarr, layers, self.tracks)
        """if not hasattr(self, "zarr"):
            notify("Open a zarr file before you save it")
            return
        try:
            save_zarr(self, self.zarr, self.viewer.layers, self.tracks)
        except ValueError as err:
            print("Caught ValueError: {}".format(err))
            if str(err) == "Raw Image layer missing!":
                notify("No layer named 'Raw Image' found!")
            if str(err) == "Segmentation layer missing!":
                notify("No layer named 'Segmentation Data' found!")
            if str(err) == "Tracks layer missing!":
                notify("No layer named 'Tracks' found!")"""
                
    def save_as(self):
        raw = layer_select(self, "Raw Image")
        if not raw[1]:
            return
        raw_name= raw[0]
        raw_data = grab_layer(self.viewer, raw_name).data
        segmentation = layer_select(self, "Segmentation Data")
        if not segmentation[1]:
            return
        segmentation_name = segmentation[0]
        segmentation_data = grab_layer(self.viewer, segmentation_name).data
        tracks = layer_select(self, "Tracks")
        if not tracks[1]:
            return
        track_name = tracks[0]
        track_data = grab_layer(self.viewer, track_name).data

        dialog = QFileDialog()
        path = f"{dialog.getSaveFileName()[0]}.zarr"
        if path == ".zarr":
            return
        print(path)
        z = zarr.open(path, mode='w')
        r = z.create_dataset('raw_data', shape = raw_data.shape, dtype = 'f8', data = raw_data)
        s = z.create_dataset('segmentation_data', shape = segmentation_data.shape, dtype = 'i4', data = segmentation_data)
        t = z.create_dataset('tracking_data', shape = track_data.shape, dtype = 'i4', data = track_data)

    def _processing(self):
        """
        Opens a [ProcessingWindow]
        """
        self.processing_window = ProcessingWindow(self)
        print("Opening processing window")
        self.processing_window.show()

    def _segmentation(self):
        """
        Opens a [SegmentationWindow]
        """
        self.segmentation_window = SegmentationWindow(self)
        print("Opening segmentation window")
        self.segmentation_window.show()

    def _tracking(self):
        """
        Opens a [TrackingWindow]
        """
        self.tracking_window = TrackingWindow(self)
        print("Opening tracking window")
        self.tracking_window.show()

    def _analysis(self):
        """
        Opens an [AnalysisWindow]
        """
        self.analysis_window = AnalysisWindow(self)
        print("Opening analysis window")
        self.analysis_window.show()
