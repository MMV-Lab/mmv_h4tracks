
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QGridLayout

class ProcessingWindow(QWidget):
    """
    A (QWidget) window to run processing steps on the data. Contains segmentation and tracking.
    
    Attributes
    ----------
    
    Methods
    -------
    """
    
    def __init__(self):
        """
        Parameters
        ----------
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        
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
        
        # Comboboxes
        combobox_segmentation = QComboBox()
        combobox_segmentation.addItem("select model")
        combobox_segmentation.addItem("model 1")
        combobox_segmentation.addItem("model 2")
        combobox_segmentation.addItem("model 3")
        combobox_segmentation.addItem("model 4")
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        
        layout_widget = QWidget()
        layout_widget.setLayout(QVBoxLayout())
        layout_widget.layout().addWidget(btn_segment)
        layout_widget.layout().addWidget(btn_preview_segment)
        
        content.layout().addWidget(layout_widget, 0, 0)
        content.layout().addWidget(btn_track, 0, 1)
        content.layout().addWidget(combobox_segmentation, 2, 0)
        content.layout().addWidget(btn_adjust_seg_ids, 3, 0)
        
        self.layout().addWidget(content)
        
        
        
        
        