import os
import numpy as np
import multiprocessing
from multiprocessing import Pool

from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QGridLayout, QApplication
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker

from ._logger import notify
from ._grabber import grab_layer

class ProcessingWindow(QWidget):
    """
    A (QWidget) window to run processing steps on the data. Contains segmentation and tracking.
    
    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance
    
    Methods
    -------
    run_segmentation()
        Run segmentation on the raw image data
    run_demo_segmentation()
        Run the segmentation on the first 5 layers only
    run_tracking()
        Run tracking on the segmented cells
    adjust_ids()
        Replaces track ID 0 & adjusts segmentation IDs to match track IDs
    """
    
    def __init__(self, viewer, parent):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        parent : QWidget
            The parent widget
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Data Processing")
        self.viewer = viewer
        self.parent = parent
        
        ### QObjects
        # Labels
        
        # Buttons
        btn_segment = QPushButton("Run Instance Segmentation")
        btn_preview_segment = QPushButton("Preview Segmentation")
        btn_track = QPushButton("Run Tracking")
        btn_adjust_seg_ids = QPushButton("Adjust Segmentation IDs")
        btn_adjust_seg_ids.setToolTip(
            "WARNING: This will take a while"
            )
        
        btn_segment.clicked.connect(self._run_segmentation)
        btn_preview_segment.clicked.connect(self._run_demo_segmentation)
        btn_track.clicked.connect(self._run_tracking)
        btn_adjust_seg_ids.clicked.connect(self._adjust_ids)
        
        # Comboboxes
        self.combobox_segmentation = QComboBox()
        self.combobox_segmentation.addItem("select model")
        self.combobox_segmentation.addItem("model 1")
        self.combobox_segmentation.addItem("model 2")
        self.combobox_segmentation.addItem("model 3")
        self.combobox_segmentation.addItem("model 4")
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        
        layout_widget = QWidget()
        layout_widget.setLayout(QVBoxLayout())
        layout_widget.layout().addWidget(btn_segment)
        layout_widget.layout().addWidget(btn_preview_segment)
        
        content.layout().addWidget(layout_widget, 0, 0)
        content.layout().addWidget(btn_track, 0, 1)
        content.layout().addWidget(self.combobox_segmentation, 2, 0)
        content.layout().addWidget(btn_adjust_seg_ids, 3, 0)
        
        self.layout().addWidget(content)
        
    def _add_segmentation_to_viewer(self, mask):
        """
        Adds the segmentation as a layer to the viewer with a specified name
        
        Parameters
        ----------
        mask : array
            the segmentation data to add to the viewer
        """
        self.viewer.add_labels(mask, name = 'calculated segmentation')
        print("Added segmentation to viewer")
        
    def _run_segmentation(self):
        """
        Calls segmentation without demo flag set
        """
        print("Calling full segmentation")
        worker = self._segment_image()
        worker.returned.connect(self._add_segmentation_to_viewer)
        worker.start()
    
    def _run_demo_segmentation(self):
        """
        Calls segmentation with the demo flag set
        """
        print("Calling demo segmentation")
        worker = self._segment_image(True)
        worker.returned.connect(self._add_segmentation_to_viewer)
        worker.start()
    
    @thread_worker
    def _segment_image(self, demo = False):
        """
        Run segmentation on the raw image data
        
        Parameters
        ----------
        demo : Boolean
            whether or not to do a demo of the segmentation
        """
        print("Running segmentation")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        from cellpose import models
        try:
            data = grab_layer(self.viewer, "Raw Image").data
        except ValueError:
            print("Image layer not found in viewer")
            QApplication.restoreOverrideCursor()
            notify("No image layer found!")
            return
        
        selected_model = self.combobox_segmentation.currentText()
        
        try:
            parameters = self._get_parameters(selected_model)
        except UnboundLocalError:
            QApplication.restoreOverrideCursor()
            notify("Please select a different model")
            return
        
        # set process limit
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1,int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1,int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))
        
        global segment_slice
        
        def segment_slice(slice, parameters):
            """
            Calculate segmentation for a single slice
            
            Parameters
            ----------
            slice : napari
                the slice of raw image data to calculate segmentation for
            parameters : dict
                the parameters for the segmentation model
            """
            model = models.CellposeModel(gpu = False, pretrained_model = parameters["model_path"])
            mask = model.eval(
                slice,
                channels = [parameters["chan"], parameters["chan2"]],
                diameter = parameters["diameter"],
                flow_threshold = parameters["flow_threshold"],
                cellprob_threshold = parameters["cellprob_threshold"]
                )[0]
            return mask
        
        data_with_parameters = []
        for slice in data:
            data_with_parameters.append((slice, parameters))
            
        with Pool(AMOUNT_OF_PROCESSES) as p:
            mask = p.starmap(segment_slice, data_with_parameters)
            mask = np.asarray(mask)
            print("Done calculating segmentation")
        
        QApplication.restoreOverrideCursor()
        return mask
        
    def _get_parameters(self, model):
        """
        Get the parameters for the selected model
        
        Parameters
        ----------
        model : String
            The selected model
            
        Returns
        -------
        dict
            a dictionary of all the parameters based on selected model
        """
        print("Getting parameters")
        if model == "model 1":
            print("Selected model 1")
            params = {
                "model_path": '/models/cellpose_neutrophils',
                "diameter": 15,
                "chan": 0,
                "chan2": 0,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0
            }
        params["model_path"] = os.path.dirname(__file__) + params["model_path"]
        
        return params
    
    def _run_tracking(self):
        """
        Run tracking on the segmented data
        """
        print("Running tracking")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        
        QApplication.restoreOverrideCursor()
    
    def _adjust_ids(self):
        """
        Replaces track ID 0. Also adjusts segmentation IDs to match track IDs
        """
        print("Adjusting segmentation IDs")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        
        QApplication.restoreOverrideCursor()
        
        
        
        
        
        
        