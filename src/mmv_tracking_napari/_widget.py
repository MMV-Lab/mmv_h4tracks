import napari
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QScrollArea, QSlider, QVBoxLayout,
                            QWidget, QMessageBox)


### TODO: Insert Manual Tracking
class MMVTracking(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Labels
        title = QLabel("<font color='green'>WIP title!</font>")
        next_free = QLabel("Next free label:")
        self.next_free_id = QLabel("next_free_id")
        trajectory = QLabel("Select ID for trajectory:")
        tail = QLabel("Set tail length:")
        load = QLabel("Load .zarr file:")
        false_positive = QLabel("Remove false positive for ID:")
        false_merge = QLabel("Cut falsely merged ID:")
        false_cut = QLabel("Merge falsely cut ID into second ID:")
        remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        insert_correspondence = QLabel("ID should be tracked with second ID:")
        extra = QLabel("Extra functions:")

        # Numeric Labels
        self.n_tail = QLabel()
        self.n_tail.setText("0")

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
        btn_plot.clicked.connect(self._get_current_slice)
        btn_save.clicked.connect(self._save_zarr)

        # Sliders
        self.s_tail = QSlider()
        self.s_tail.setRange(0,30)
        self.s_tail.setValue(0)
        self.s_tail.setOrientation(Qt.Horizontal)
        self.s_tail.setPageStep(2)

        # Link numeric labels to sliders
        self.s_tail.valueChanged.connect(self._update_tail)

        # Update tracks layer on release of slider
        self.s_tail.sliderReleased.connect(self._update_track_tails)

        # Line Edits
        self.le_trajectory = QLineEdit("-1")
        self.le_false_positive = QLineEdit("-1")
        self.le_false_merge = QLineEdit("-1")
        self.le_false_cut_1 = QLineEdit("-1")
        self.le_false_cut_2 = QLineEdit("-1")
        self.le_remove_corespondence = QLineEdit("-1")
        self.le_insert_corespondence_1 = QLineEdit("-1")
        self.le_insert_corespondence_2 = QLineEdit("-1")

        # Link functions to line edits
        self.le_trajectory.editingFinished.connect(self._select_track)

        # Loading .zarr file UI
        q_load = QWidget()
        q_load.setLayout(QHBoxLayout())
        q_load.layout().addWidget(load)
        q_load.layout().addWidget(btn_load)

        # Changing tail length UI
        q_tail = QWidget()
        q_tail.setLayout(QHBoxLayout())
        q_tail.layout().addWidget(tail)
        q_tail.layout().addWidget(self.s_tail)
        q_tail.layout().addWidget(self.n_tail)

        # Selecting trajectory UI
        q_trajectory = QWidget()
        q_trajectory.setLayout(QHBoxLayout())
        q_trajectory.layout().addWidget(trajectory)
        q_trajectory.layout().addWidget(self.le_trajectory)

        # Correcting segmentation UI
        help_false_positive = QWidget()
        help_false_positive.setLayout(QHBoxLayout())
        help_false_positive.layout().addWidget(false_positive)
        help_false_positive.layout().addWidget(self.le_false_positive)
        help_false_positive.layout().addWidget(btn_false_positive)
        help_false_negative = QWidget()
        help_false_negative.setLayout(QHBoxLayout())
        help_false_negative.layout().addWidget(next_free)
        help_false_negative.layout().addWidget(self.next_free_id)
        help_false_merge = QWidget()
        help_false_merge.setLayout(QHBoxLayout())
        help_false_merge.layout().addWidget(false_merge)
        help_false_merge.layout().addWidget(self.le_false_merge)
        help_false_merge.layout().addWidget(btn_false_merge)
        help_false_cut = QWidget()
        help_false_cut.setLayout(QHBoxLayout())
        help_false_cut.layout().addWidget(false_cut)
        help_false_cut.layout().addWidget(self.le_false_cut_1)
        help_false_cut.layout().addWidget(self.le_false_cut_2)
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
        help_remove_correspondence.layout().addWidget(self.le_remove_corespondence)
        help_remove_correspondence.layout().addWidget(btn_remove_correspondence)
        help_insert_correspondence = QWidget()
        help_insert_correspondence.setLayout(QHBoxLayout())
        help_insert_correspondence.layout().addWidget(insert_correspondence)
        help_insert_correspondence.layout().addWidget(self.le_insert_corespondence_1)
        help_insert_correspondence.layout().addWidget(self.le_insert_corespondence_2)
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
        self.file = QFileDialog.getExistingDirectory(self, "Select Zarr-File")
        self.z1 = zarr.load(self.file)
        self.viewer.add_image(self.z1['raw_data/Image 1'][:], name = 'Raw Image')
        self.viewer.add_labels(self.z1['segmentation_data/Image 1'][:], name = 'Segmentation Data')
        self.viewer.add_tracks(self.z1['tracking_data/Image 1'][:], name = 'Tracks', tail_length = self.s_tail.value()) # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
        self._get_next_free_id()

    def _get_current_slice(self):
        #napari.viewer.current_viewer().dims.set_current_step(0,5)
        print(napari.viewer.current_viewer().dims.current_step[0]) # prints current slice

    def _update_tail(self):
        self.n_tail.setText(str(self.s_tail.value()))

    def _update_track_tails(self):
        self.viewer.layers.remove('Tracks')
        self.viewer.add_tracks(self.z1['tracking_data/Image 1'][:], name='Tracks', tail_length=self.s_tail.value())
        self._get_next_free_id()

    def _save_zarr(self):
        zarr.save(self.file, self.z1)
        msg = QMessageBox()
        msg.setText("Zarr file has been saved.")
        msg.exec()

    def _temp(self):
        layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Tracks)
        ]
        print(str(len(layers)) + " tracks layers")
        print(layers[0].data)
        #print(napari.viewer.current_viewer().layers.selection)
        pass

    def _plot(self):
        pass

    def _select_track(self):
        try:
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        id = int(self.le_trajectory.text())
        if id < 0:
            self.viewer.add_tracks(self.z1['tracking_data/Image 1'][:], name='Tracks', tail_length=self.s_tail.value())
            self._get_next_free_id()
        else:
            tracks_data = [
                track
                for track in self.z1['tracking_data/Image 1'][:]
                if track[0] == id
            ]
            if not tracks_data:
                print("No tracking data found for id " + str(id))
                return
            self.viewer.add_tracks(tracks_data, name='Tracks', tail_length=self.s_tail.value())
            self._get_next_free_id()

    def _get_next_free_id(self):
        i = 0
        while True:
            for element in self.z1['tracking_data/Image 1'][:,0]:
                if i == element:
                    i = i + 1
            break
        self.next_free_id.setText(str(i))
