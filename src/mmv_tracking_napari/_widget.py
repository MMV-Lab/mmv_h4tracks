
from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout,
                            QScrollArea, QMessageBox)

import numpy as np

from ._reader import open_dialog, napari_get_reader
from ._writer import save_zarr
from ._logger import setup_logging, notify
from ._processing import ProcessingWindow
from ._segmentation import SegmentationWindow
from ._tracking import TrackingWindow
from ._analysis import AnalysisWindow

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
    
    def __init__(self, napari_viewer):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        """
        super().__init__()
        self.viewer = napari_viewer
        
        setup_logging()
        
        ### QObjects
        
        # Labels
        title = QLabel("<font color='green'>HITL4Trk</font>")
        
        # Buttons
        btn_load = QPushButton("Load")
        btn_save = QPushButton("Save")
        btn_processing = QPushButton("Data Processing")
        btn_segmentation = QPushButton("Correct Segmentation")
        btn_tracking = QPushButton("Correct Tracking")
        btn_analysis = QPushButton("Analysis")
        
        btn_load.clicked.connect(self._load)
        btn_save.clicked.connect(self._save)
        btn_processing.clicked.connect(self._processing)
        btn_segmentation.clicked.connect(self._segmentation)
        btn_tracking.clicked.connect(self._tracking)
        btn_analysis.clicked.connect(self._analysis)
        
        # Radio Buttons
        rb_eco = QRadioButton("Eco")
        rb_eco.toggle()
        rb_heavy = QRadioButton("Heavy")
        
        ### Organize objects via widgets
        # widget: parent widget of all content
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(title)
        
        computation_mode_rbs = QWidget()
        computation_mode_rbs.setLayout(QHBoxLayout())
        computation_mode_rbs.layout().addWidget(rb_eco)
        computation_mode_rbs.layout().addWidget(rb_heavy)
        
        widget.layout().addWidget(computation_mode_rbs)
        
        read_write_files = QWidget()
        read_write_files.setLayout(QHBoxLayout())
        read_write_files.layout().addWidget(btn_load)
        read_write_files.layout().addWidget(btn_save)
        
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
        
        
    def _load(self):
        """
        Opens a dialog for the user to choose a zarr file to open. Checks if any layernames are blocked
        """
        print("opening dialog")
        filepath = open_dialog(self)
        print("dialog is closed, retrieving reader")
        file_reader = napari_get_reader(filepath)
        print("got '{}' as file reader".format(file_reader))
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("reading file")
                zarr_file = file_reader(filepath)
                print("file has been read")
        except TypeError:
            print("could not read file")
            return
        
        # check all layer names
        for layername in zarr_file.__iter__():
            if layername in self.viewer.layers:
                print("detected layer with name {}".format(layername))
                msg = QMessageBox()
                msg.setWindowTitle("Layer already exists")
                msg.setText("Found layer with name " + layername)
                msg.setInformativeText("A layer with the name \'" + layername + "\' exists already." +
                                       " Do you want to delete this layer to proceed?")
                msg.addButton(QMessageBox.Yes)
                msg.addButton(QMessageBox.YesToAll)
                msg.addButton(QMessageBox.Cancel)
                ret = msg.exec() # Yes -> 16384, YesToAll -> 32768, Cancel -> 4194304
                
                # Cancel
                if ret == 4194304:
                    print("loading cancelled")
                    return
                
                # YesToAll -> Remove all layers with names in the file
                if ret == 32768:
                    print("removing all layers with names in zarr from viewer")
                    for name in zarr_file.__iter__():
                        try:
                            self.viewer.layers.remove(name)
                        except ValueError:
                            pass
                    break
                
                # Yes -> Remove this layer
                print("removing layer {}".format(layername))
                self.viewer.layers.remove(layername)
            
        print("adding layers")
        # add layers to viewer
        try:
            self.viewer.add_image(zarr_file['raw_data'][:], name = 'Raw Image')
            segmentation = zarr_file['segmentation_data'][:]
            # save tracks so we can delete one slice tracks first
            tracks = zarr_file['tracking_data'][:]
        except:
            print("File does not have the right structure of raw_data, segmentation_data and tracking_data!")
        else:
            # Filter track ids of tracks that just occur once
            count_of_track_ids = np.unique(tracks[:,0], return_counts = True)
            filtered_track_ids = np.delete(count_of_track_ids, count_of_track_ids[1] == 1, 1)
            
            # Remove tracks that only exist in one slice
            filtered_tracks = np.delete(tracks, np.where(np.isin(tracks[:,0], filtered_track_ids[0,:], invert = True)), 0)
            self.viewer.add_tracks(filtered_tracks, name = 'Tracks')
            
            self.viewer.add_labels(segmentation, name = 'Segmentation Data')
        
        print("layers have been added")
        
        self.zarr = zarr_file
        self.tracks = filtered_tracks
        self.initial_layers = [segmentation, filtered_tracks]
    
    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened or not all layers exist
        """
        if not hasattr(self, 'zarr'):
            notify("open a zarr file before you save it")
            return
        try:
            save_zarr(self, self.zarr, self.viewer.layers, self.tracks)
        except ValueError as err:
            print("caught ValueError: {}".format(err))
            if str(err) == "Raw Image layer missing!":
                notify("No layer named 'Raw Image' found!")
            if str(err) == "Segmentation layer missing!":
                notify("No layer named 'Segmentation Data' found!")
            if str(err) == "Tracks layer missing!":
                notify("No layer named 'Tracks' found!")
                
    
    def _processing(self):
        """
        Opens a [ProcessingWindow]
        """
        self.processing_window = ProcessingWindow()
        print("opening processing window")
        self.processing_window.show()
        
    
    def _segmentation(self):
        """
        Opens a [SegmentationWindow]
        """
        self.segmentation_window = SegmentationWindow()
        print("opening segmentaiton window")
        self.segmentation_window.show()
        
    
    def _tracking(self):
        """
        Opens a [TrackingWindow]
        """
        self.tracking_window = TrackingWindow()
        print("opening tracking window")
        self.tracking_window.show()
        
    
    def _analysis(self):
        """
        Opens an [AnalysisWindow]
        """
        self.analysis_window = AnalysisWindow()
        print("opening analysis window")
        self.analysis_window.show()
        
        
    
    
    
    