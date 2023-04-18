
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QGridLayout
from ._logger import notify

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
    
    def __init__(self, viewer):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Data Processing")
        
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
        
    def _run_segmentation(self, demo = False):
        """
        Run segmentation on the raw image data
        
        Parameters
        ----------
        demo : Boolean
            whether or not to do a demo of the segmentation
        """
        from cellpose import models
        try:
            data = viewer.layers[viewer.layers.index("Raw Image")].data
        except ValueError:
            notify("No image layer found!")
            return
        
        selected_model = self.combobox_segmentation.currentText()
        
        parameters = self._get_parameters(selected_model)
        
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
        if model == "model 1":
            params = {
                "model_path": 'models/cellpose_neutrophils',
                "diameter": 15,
                "chan": 0,
                "chan2": 0,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0
            }
        
        return params
    
    def _run_demo_segmentation(self):
        """
        Calls segmentation with the demo flag set
        """
        self._run_segmentation(True)
    
    def _run_tracking(self):
        """
        Run tracking on the segmented data
        """
        
        pass
    
    def _adjust_ids(self):
        """
        Replaces track ID 0. Also adjusts segmentation IDs to match track IDs
        """
        pass
        
        
        
        
        
        