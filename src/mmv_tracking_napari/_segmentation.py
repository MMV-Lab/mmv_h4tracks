
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout

class SegmentationWindow(QWidget):
    """
    A (QWidget) window to correct the segmentation of the data.
    
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
        label_false_positive = QLabel("Remove false positive for ID:")
        label_next_free = QLabel("Next free label:")
        label_false_merge = QLabel("Separate falsely merged ID:")
        label_false_cut = QLabel("Merge falsely cut ID and second ID:")
        label_grab_label = QLabel("Select label:")
        
        # Buttons
        btn_false_positive = QPushButton("Remove")
        btn_false_positive.setToolTip(
            "Remove label from segmentation"
        )
        
        btn_free_label = QPushButton("Load Label")
        btn_free_label.setToolTip(
            "Load next free segmentation label"
        )
        
        btn_false_merge = QPushButton("Separate")
        btn_false_merge.setToolTip(
            "Split two separate parts of the same label into two"
        )
        
        btn_false_cut = QPushButton("Merge")
        btn_false_cut.setToolTip(
            "Merge two separate labels into one"
        )
        
        btn_grab_label = QPushButton("Select")
        btn_grab_label.setToolTip(
            "Load selected segmentation label"
        )
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())

        content.layout().addWidget(label_false_positive, 0, 0)
        content.layout().addWidget(btn_false_positive, 0, 1)
        content.layout().addWidget(label_next_free, 1, 0)
        content.layout().addWidget(btn_free_label, 1, 1)
        content.layout().addWidget(label_false_merge, 2, 0)
        content.layout().addWidget(btn_false_merge, 2, 1)
        content.layout().addWidget(label_false_cut, 3, 0)
        content.layout().addWidget(btn_false_cut, 3, 1)
        content.layout().addWidget(label_grab_label, 4, 0)
        content.layout().addWidget(btn_grab_label, 4, 1)
        
        self.layout().addWidget(content)
        
        
        
        
        