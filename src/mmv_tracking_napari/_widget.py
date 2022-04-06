from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QLabel, QFileDialog, QSlider, QLineEdit
from qtpy.QtCore import Qt


class MMVTracking(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Labels
        title = QLabel("<font color='green'>WIP title!</font>")
        next_free = QLabel("Next free label:")
        next_free_id = QLabel("next_free_id")
        trajectory = QLabel("Select ID for trajectory:")
        tail = QLabel("Set tail length:")
        load = QLabel("Load .zarr file:")
        false_positive = QLabel("Remove false positive for ID:")
        false_merge = QLabel("Cut falsely merged ID:")
        false_cut = QLabel("Merge falsely cut ID into second ID:")
        remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        insert_correspondence = QLabel("ID should be tracked with second ID:")
        extra = QLabel("Extra functions:")

        # Buttons
        btn_load = QPushButton("Load")
        btn_false_positive = QPushButton("Remove")
        btn_false_merge = QPushButton("Cut")
        btn_false_cut = QPushButton("Merge")
        btn_remove_correspondence = QPushButton("Unlink")
        btn_insert_correspondence = QPushButton("Link")
        btn_save = QPushButton("Save")
        btn_plot = QPushButton("Plot")

        # Linking buttons to functions
        btn_load.clicked.connect(self._load_zarr)

        # Sliders
        s_tail = QSlider()
        s_tail.setRange(0,10)
        s_tail.setValue(0)
        s_tail.setOrientation(Qt.Horizontal)
        s_tail.setPageStep(2)

        # Line Edits
        le_trajectory = QLineEdit("-1")
        le_false_positive = QLineEdit("-1")
        le_false_merge = QLineEdit("-1")
        le_false_cut_1 = QLineEdit("-1")
        le_false_cut_2 = QLineEdit("-1")
        le_remove_corespondence = QLineEdit("-1")
        le_insert_corespondence_1 = QLineEdit("-1")
        le_insert_corespondence_2 = QLineEdit("-1")

        # Loading .zarr file UI
        q_load = QWidget()
        q_load.setLayout(QHBoxLayout())
        q_load.layout().addWidget(load)
        q_load.layout().addWidget(btn_load)

        # Changing tail length UI
        q_tail = QWidget()
        q_tail.setLayout(QHBoxLayout())
        q_tail.layout().addWidget(tail)
        q_tail.layout().addWidget(s_tail)

        # Selecting trajectory UI
        q_trajectory = QWidget()
        q_trajectory.setLayout(QHBoxLayout())
        q_trajectory.layout().addWidget(trajectory)
        q_trajectory.layout().addWidget(le_trajectory)

        # Correcting segmentation UI
        help_false_positive = QWidget()
        help_false_positive.setLayout(QHBoxLayout())
        help_false_positive.layout().addWidget(false_positive)
        help_false_positive.layout().addWidget(le_false_positive)
        help_false_positive.layout().addWidget(btn_false_positive)
        help_false_negative = QWidget()
        help_false_negative.setLayout(QHBoxLayout())
        help_false_negative.layout().addWidget(next_free)
        help_false_negative.layout().addWidget(next_free_id)
        help_false_merge = QWidget()
        help_false_merge.setLayout(QHBoxLayout())
        help_false_merge.layout().addWidget(false_merge)
        help_false_merge.layout().addWidget(le_false_merge)
        help_false_merge.layout().addWidget(btn_false_merge)
        help_false_cut = QWidget()
        help_false_cut.setLayout(QHBoxLayout())
        help_false_cut.layout().addWidget(false_cut)
        help_false_cut.layout().addWidget(le_false_cut_1)
        help_false_cut.layout().addWidget(le_false_cut_2)
        help_false_cut.layout().addWidget(btn_false_cut)
        q_segmentation = QWidget()
        q_segmentation.setLayout(QVBoxLayout())
        q_segmentation.layout().addWidget(help_false_positive)
        q_segmentation.layout().addWidget(help_false_negative)
        q_segmentation.layout().addWidget(help_false_merge)
        q_segmentation.layout().addWidget(help_false_cut)

        # Correcting correspondence UI
        help_remove_correspondence = QWidget()
        help_remove_correspondence.setLayout(QHBoxLayout())
        help_remove_correspondence.layout().addWidget(remove_correspondence)
        help_remove_correspondence.layout().addWidget(le_remove_corespondence)
        help_remove_correspondence.layout().addWidget(btn_remove_correspondence)
        help_insert_correspondence = QWidget()
        help_insert_correspondence.setLayout(QHBoxLayout())
        help_insert_correspondence.layout().addWidget(insert_correspondence)
        help_insert_correspondence.layout().addWidget(le_insert_corespondence_1)
        help_insert_correspondence.layout().addWidget(le_insert_corespondence_2)
        help_insert_correspondence.layout().addWidget(btn_insert_correspondence)
        q_tracking = QWidget()
        q_tracking.setLayout(QVBoxLayout())
        q_tracking.layout().addWidget(help_remove_correspondence)
        q_tracking.layout().addWidget(help_insert_correspondence)

        # Extra elements UI
        q_extra = QWidget()
        q_extra.setLayout(QHBoxLayout())
        q_extra.layout().addWidget(extra)
        q_extra.layout().addWidget(btn_save)
        q_extra.layout().addWidget(btn_plot)

        # Assemble UI elements in ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setLayout(QVBoxLayout())
        scroll_area.layout().addWidget(title)
        scroll_area.layout().addWidget(q_load)
        scroll_area.layout().addWidget(q_tail)
        scroll_area.layout().addWidget(q_trajectory)
        scroll_area.layout().addWidget(q_segmentation)
        scroll_area.layout().addWidget(q_tracking)
        scroll_area.layout().addWidget(q_extra)

        # Set ScrollArea as content of plugin
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    # Functions
    def _load_zarr(self):
        zarr = QFileDialog.getOpenFileUrl(self, "Select Zarr-File")
        print("Selected " + zarr)