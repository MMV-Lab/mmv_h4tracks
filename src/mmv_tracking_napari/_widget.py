import napari
import numpy as np
import zarr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                            QMessageBox, QPushButton, QScrollArea, QSlider,
                            QVBoxLayout, QWidget)


### TODO: Insert Manual Tracking
class MMVTracking(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Labels
        title = QLabel("<font color='green'>Tracking, Visualization, Editing</font>")
        next_free = QLabel("Next free label:")
        self.next_free_id = QLabel("next_free_id")
        trajectory = QLabel("Select ID for trajectory:")
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
        btn_plot.clicked.connect(self._get_current_slice)
        btn_save.clicked.connect(self._save_zarr)
       

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
        scroll_area.layout().addWidget(q_trajectory)
        scroll_area.layout().addWidget(q_segmentation)
        scroll_area.layout().addWidget(q_tracking)
        scroll_area.layout().addWidget(q_extra)

        # Set ScrollArea as content of plugin
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    # Functions
    def _load_zarr(self):
        dialog = QFileDialog()
        dialog.setNameFilter('*.zarr')
        self.file = dialog.getExistingDirectory(self, "Select Zarr-File")
        if(self.file == ""):
            print("No file selected")
            return
        self.z1 = zarr.open(self.file,mode='a')
        try:
            self.viewer.add_image(self.z1['raw_data/Image 1'][:], name = 'Raw Image')
            self.viewer.add_labels(self.z1['segmentation_data/Image 1'][:], name = 'Segmentation Data')
            self.viewer.add_tracks(self.z1['tracking_data/Image 1'][:], name = 'Tracks') # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
        except:
            print("File is either no Zarr file or does not contain required groups")
        else:
            segmentation = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ][0]
            segmentation.events.mode.connect(self._get_next_free_id)
            self._get_next_free_id()

    def _get_current_slice(self):
        #napari.viewer.current_viewer().dims.set_current_step(0,5)
        print(napari.viewer.current_viewer().dims.current_step[0]) # prints current slice

    def _save_zarr(self):
        for layer in self.viewer.layers:
            if layer.name == 'Raw Image' and isinstance(layer, napari.layers.Image):
                self.z1['raw_data/Image 1'][:] = layer.data
                continue
            if layer.name == 'Segmentation Data' and isinstance(layer, napari.layers.Labels):
                self.z1['segmentation_data/Image 1'][:] = layer.data
                continue
            if layer.name == 'Tracks' and isinstance(layer, napari.layers.Tracks):
                self.z1['tracking_data/Image 1'][:] = layer.data
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
            id = int(self.le_trajectory.text())
        except ValueError:
            msg = QMessageBox()
            msg.setText("Please use integer (whole number)")
            msg.exec()
            return
        try:
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        if id < 0:
            self.viewer.add_tracks(self.z1['tracking_data/Image 1'][:], name='Tracks')
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
            self.viewer.add_tracks(tracks_data, name='Tracks')
            self._get_next_free_id()

    def _get_next_free_id(self):
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.next_free_id.setText(str(np.amax(layer.data)+1))
                return
            pass
        # TODO: handling for if the layer doesn't exist