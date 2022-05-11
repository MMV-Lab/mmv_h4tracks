import napari
import numpy as np
import zarr
from qtpy.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                            QLineEdit, QMessageBox, QPushButton, QScrollArea, QGridLayout,
                            QStackedWidget, QVBoxLayout, QWidget)

### TODO: change structure away from 'Image 1' as soon as data with new structure is available
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
        load_save = QLabel("Load/Save .zarr file:")
        false_positive = QLabel("Remove false positive for ID:")
        false_merge = QLabel("Cut falsely merged ID:")
        false_cut = QLabel("Merge falsely cut ID into second ID:")
        remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        insert_correspondence = QLabel("ID should be tracked with second ID:")
        metric = QLabel("Evaluation metrics:")

        # Tooltips for Labels
        load_save_tip = (
            "Loading: Select the .zarr directory to open the file.<br><br>\n\n"
            "Saving: Overwrites the file selected at the time of loading!"
        )
        load_save.setToolTip(load_save_tip)

        # Buttons
        btn_load = QPushButton("Load")
        btn_false_positive = QPushButton("Remove")
        btn_false_merge = QPushButton("Cut")
        btn_false_cut = QPushButton("Merge")
        btn_remove_correspondence = QPushButton("Unlink")
        btn_insert_correspondence = QPushButton("Link")
        btn_save = QPushButton("Save")
        btn_plot = QPushButton("Plot")
        btn_show_seg_track = QPushButton("temp_button_name_collapse")
        btn_segment = QPushButton("Run instance segmentation")
        btn_track = QPushButton("Run tracking")

        # Linking buttons to functions
        btn_load.clicked.connect(self._load_zarr)
        btn_plot.clicked.connect(self._plot)
        btn_save.clicked.connect(self._save_zarr)
        btn_show_seg_track.clicked.connect(self._show_seg_track)
        btn_false_positive.clicked.connect(self._remove_fp)
        btn_segment.clicked.connect(self._temp)
       
        # Combo Boxes
        c_segmentation = QComboBox()
        c_plots = QComboBox()

        # Adding entries to Combo Boxes
        c_segmentation.addItem("select model")
        c_segmentation.addItem("model 1")
        c_segmentation.addItem("model 2")
        c_segmentation.addItem("model 3")
        c_segmentation.addItem("model 4")
        c_plots.addItem("select metric")
        c_plots.addItem("metric 1")
        c_plots.addItem("metric 2")
        c_plots.addItem("metric 3")

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

        # Running segmentation/tracking UI
        q_empty = QWidget() # Empty widget to create blank space
        q_empty.setLayout(QHBoxLayout())
        q_run = QWidget()
        q_run.setLayout(QGridLayout())
        q_run.layout().addWidget(btn_segment,0,0)
        q_run.layout().addWidget(btn_track,0,1)
        q_run.layout().addWidget(c_segmentation,1,0)
        self.stack = QStackedWidget()
        self.stack.insertWidget(0,q_empty)
        self.stack.insertWidget(1,q_run)
        q_seg_track = QWidget()
        q_seg_track.setLayout(QVBoxLayout())
        q_seg_track.layout().addWidget(btn_show_seg_track)
        q_seg_track.layout().addWidget(self.stack)

        # Loading/Saving .zarr file UI
        q_load = QWidget()
        q_load.setLayout(QHBoxLayout())
        q_load.layout().addWidget(load_save)
        q_load.layout().addWidget(btn_load)
        q_load.layout().addWidget(btn_save)

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

        # Plot UI
        q_plot = QWidget()
        q_plot.setLayout(QHBoxLayout())
        q_plot.layout().addWidget(metric)
        q_plot.layout().addWidget(c_plots)
        q_plot.layout().addWidget(btn_plot)

        # Assemble UI elements in ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setLayout(QVBoxLayout())
        scroll_area.layout().addWidget(title)
        scroll_area.layout().addWidget(q_seg_track)
        scroll_area.layout().addWidget(q_load)
        scroll_area.layout().addWidget(q_trajectory)
        scroll_area.layout().addWidget(q_segmentation)
        scroll_area.layout().addWidget(q_tracking)
        scroll_area.layout().addWidget(q_plot)

        # Set ScrollArea as content of plugin
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    # Functions
    @napari.Viewer.bind_key('i')
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
            segmentation = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
            segmentation.events.mode.connect(self._get_next_free_id)
            self._get_next_free_id()
    
    def _save_zarr(self):
        # Useful if we later want to save to new file
        """try:
            raw = self.viewer.layers.index("Raw Image")
        except ValueError:
            err = QMessageBox()
            err.setText("No Raw Data layer found!")
            err.exec()
            return"""
        try:
            seg = self.viewer.layers.index("Segmentation Data")
        except ValueError:
            err = QMessageBox()
            err.setText("No Segmentation Data layer found!")
            err.exec()
            return
        try:
            track = self.viewer.layers.index("Tracking")
        except ValueError:
            err = QMessageBox()
            err.setText("No Tracking Data layer found!")
            err.exec()
            return
        #self.z1['raw_data/Image 1'][:] = self.viewer.layers[raw].data
        self.z1['segmentation_data/Image 1'][:] = self.viewer.layers[seg].data
        self.z1['tracking_data/Image 1'][:] = self.viewer.layers[track].data
        msg = QMessageBox()
        msg.setText("Zarr file has been saved.")
        msg.exec()

    def _temp(self):
        print(self.viewer.layers[0].data)
        #print(self.viewer.layers.index("Raw Image")) # Indexes go from bottom to top
        """layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Tracks)
        ]
        print(str(len(layers)) + " tracks layers")
        print(layers[0].data)"""
        #print(napari.viewer.current_viewer().layers.selection)
        pass

    def _plot(self):
        pass

    def _select_track(self):
        try: # Try for one value
            id = int(self.le_trajectory.text())
        except ValueError: # Try for list of values
            txt = self.le_trajectory.text()
            id = []
            try:
                for i in range(0,len(txt.split(","))):
                    id.append(int(txt.split(",")[i]))
            except ValueError:
                msg = QMessageBox()
                msg.setText("Please use a single integer (whole number) or a comma separated list of integers")
                msg.exec()
                return
        try:
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        if isinstance(id,int): # Single value
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
        else: # Multiple values, id is instance of "list"
            id = list(dict.fromkeys(id)) # Removes multiple values
            for i in range(0,len(id)): # Remove illegal values (<0) from id
                if id[i] < 0:
                    id.pop(i)
            # ID now only contains legal values, can be written back to line edit
            txt = ""
            for i in range(0,len(id)):
                if len(txt)>0:
                    txt = txt + ","
                txt = f'{txt}{id[i]}'
            self.le_trajectory.setText(txt)
            tracks_data = [
                track
                for track in self.z1['tracking_data/Image 1'][:]
                if track[0] in id
            ]
            if not tracks_data:
                print("No tracking data found for ids " + str(id))
                return
            self.viewer.add_tracks(tracks_data, name='Tracks')
            self._get_next_free_id()

    def _get_next_free_id(self):
        try:
            self.next_free_id.setText(str(np.amax(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data)+1))
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()

    def _show_seg_track(self): # Switches element of stack to be shown
        self.stack.setCurrentIndex(abs(self.stack.currentIndex()-1))

    def _remove_fp(self):
        try:
            data = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        try:
            if np.count_nonzero(data[napari.viewer.current_viewer().dims.current_step[0]] == int(self.le_false_positive.text())) < 1:
                msg = QMessageBox()
                msg.setText("ID doesn't exist in the current slice")
                msg.exec()
                return
        except ValueError:
            msg = QMessageBox()
            msg.setText("Please use an Integer (whole number) as ID")
            msg.exec()
            return
        # Replace all pixels with given ID with 0 in current slice
        np.place(data[napari.viewer.current_viewer().dims.current_step[0]], data[napari.viewer.current_viewer().dims.current_step[0]]==int(self.le_false_positive.text()), 0)
        self.viewer.layers.pop(self.viewer.layers.index("Segmentation Data"))
        self.viewer.add_labels(data, name = 'Segmentation Data')
        self.viewer.layers.move(self.viewer.layers.index("Segmentation Data"),1)
