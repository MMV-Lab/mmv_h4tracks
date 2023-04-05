
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QGridLayout

class TrackingWindow(QWidget):
    """
    A (QWidget) window to correct the tracking within the data.
    
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
        label_trajectory = QLabel("Select ID for trajectory:")
        label_remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        label_insert_correspondence = QLabel("ID should be tracked with second ID:")
        
        # Buttons
        btn_remove_correspondence = QPushButton("Unlink")
        btn_remove_correspondence.setToolTip(
            "Remove cells from their tracks"
        )
        
        btn_insert_correspondence = QPushButton("Link")
        btn_insert_correspondence.setToolTip(
            "Add cells to new track"
        )
        
        btn_delete_displayed_tracks = QPushButton("Delete displayed tracks")
        btn_auto_track = QPushButton("Automatic tracking for single cell")
        btn_auto_track_all = QPushButton("Automatic tracking for all cells")
        
        # Line Edits
        le_trajectory = QLineEdit("")
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(label_trajectory, 0, 0)
        content.layout().addWidget(le_trajectory, 0, 1)
        content.layout().addWidget(btn_delete_displayed_tracks, 0, 2)
        content.layout().addWidget(label_remove_correspondence, 1, 0)
        content.layout().addWidget(btn_remove_correspondence, 1, 1)
        content.layout().addWidget(btn_auto_track, 1, 2)
        content.layout().addWidget(label_insert_correspondence, 2, 0)
        content.layout().addWidget(btn_insert_correspondence, 2, 1)
        content.layout().addWidget(btn_auto_track_all, 2, 2)
        
        self.layout().addWidget(content)
        
        
        
        
        