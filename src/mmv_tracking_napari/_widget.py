
from qtpy.QtWidgets import (QWidget, QLabel, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout,
                            QScrollArea, QMessageBox)

import numpy as np

from ._reader import open_dialog, napari_get_reader
from ._logger import setup_logging

class MMVTracking(QWidget):
    """
    The main widget of our application
    
    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance
        
    Methods
    -------
    load()
        Opens a dialog for the user to choose a zarr file (directory)
    save()
        Writes the changes made to the opened zarr file
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
        btn_processing = QPushButton("Data processing")
        btn_segmentation = QPushButton("Correct segmentation")
        btn_tracking = QPushButton("Correct tracking")
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
        print("got {} as file reader".format(file_reader))
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("reading file")
                zarr_file = file_reader(filepath)
                print("file has been read")
        except TypeError:
            return
        
        # check all layer names
        for layername in zarr_file.__iter__():
            if layername in self.viewer.layers:
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
            
        print("adding layers")
        # add layers to viewer
        try:
            self.viewer.add_image(zarr_file['raw_data'][:], name = 'Raw Image')
            self.viewer.add_labels(zarr_file['segmentation_data'][:], name = 'Segmentation Data')
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
        
        print("layers have been added")
        # TODO: cache of segmentation and tracks
    
    def _save(self):
        """
        Writes the changes made to the opened zarr file to disk.
        Fails if no zarr file was opened
        """
        pass
    
    def _processing(self):
        """
        Opens a [ProcessingWindow]
        """
        pass
    
    def _segmentation(self):
        """
        Opens a [SegmentationWindow]
        """
        pass
    
    def _tracking(self):
        """
        Opens a [TrackingWindow]
        """
        pass
    
    def _analysis(self):
        """
        Opens an [AnalysisWindow]
        """
        pass
    
    
    
    