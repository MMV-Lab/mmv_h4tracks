import napari
import numpy as np
import zarr
from qtpy.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                            QLabel, QLineEdit, QMessageBox, QPushButton,
                            QScrollArea, QToolBox, QVBoxLayout, QWidget)

### TODO: Insert Manual Tracking
class MMVTracking(QWidget):
    dock = None
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        MMVTracking.dock = self
        # Labels
        title = QLabel("<font color='green'>Tracking, Visualization, Editing</font>")
        next_free = QLabel("Next free label:")
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
        btn_load.setToolTip("Q")
        btn_false_positive = QPushButton("Remove")
        btn_false_positive.setToolTip("R")
        btn_false_merge = QPushButton("Cut")
        btn_false_merge.setToolTip("T")
        btn_false_cut = QPushButton("Merge")
        btn_false_cut.setToolTip("Z")
        btn_remove_correspondence = QPushButton("Unlink")
        btn_insert_correspondence = QPushButton("Link")
        btn_save = QPushButton("Save")
        btn_save.setToolTip("W")
        btn_plot = QPushButton("Plot")
        btn_segment = QPushButton("Run instance segmentation")
        btn_track = QPushButton("Run tracking")
        btn_free_label = QPushButton("Load Label")
        btn_free_label.setToolTip("E")

        # Linking buttons to functions
        btn_load.clicked.connect(self._load_zarr)
        btn_plot.clicked.connect(self._plot)
        btn_save.clicked.connect(self._save_zarr)
        btn_false_positive.clicked.connect(self._remove_fp)
        btn_segment.clicked.connect(self._temp)
        btn_false_merge.clicked.connect(self._false_merge)
        btn_free_label.clicked.connect(self._get_free_id)
        btn_false_cut.clicked.connect(self._false_cut)
       
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
        self.le_trajectory = QLineEdit("")
        self.le_false_positive = QLineEdit("")
        self.le_false_merge = QLineEdit("")
        self.le_false_cut_1 = QLineEdit("0")
        self.le_false_cut_2 = QLineEdit("0")
        self.le_remove_corespondence = QLineEdit("0")
        self.le_insert_corespondence_1 = QLineEdit("")
        self.le_insert_corespondence_2 = QLineEdit("0")

        # Link functions to line edits
        self.le_trajectory.editingFinished.connect(self._select_track)

        # Tool Box
        self.toolbox = QToolBox()

        # Running segmentation/tracking UI
        q_seg_track = QWidget()
        q_seg_track.setLayout(QGridLayout())
        q_seg_track.layout().addWidget(btn_segment,0,0)
        q_seg_track.layout().addWidget(btn_track,0,1)
        q_seg_track.layout().addWidget(c_segmentation,1,0)

        # Loading/Saving .zarr file UI
        q_load = QWidget()
        q_load.setLayout(QHBoxLayout())
        q_load.layout().addWidget(load_save)
        q_load.layout().addWidget(btn_load)
        q_load.layout().addWidget(btn_save)

        # Correcting segmentation UI
        help_false_positive = QWidget()
        help_false_positive.setLayout(QHBoxLayout())
        help_false_positive.layout().addWidget(false_positive)
        help_false_positive.layout().addWidget(self.le_false_positive)
        help_false_positive.layout().addWidget(btn_false_positive)
        help_false_negative = QWidget()
        help_false_negative.setLayout(QHBoxLayout())
        help_false_negative.layout().addWidget(next_free)
        help_false_negative.layout().addWidget(btn_free_label)
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
        q_segmentation.layout().addWidget(help_false_negative)
        q_segmentation.layout().addWidget(help_false_positive)
        q_segmentation.layout().addWidget(help_false_merge)
        q_segmentation.layout().addWidget(help_false_cut)

        # Postprocessing tracking UI
        help_trajectory = QWidget()
        help_trajectory.setLayout(QHBoxLayout())
        help_trajectory.layout().addWidget(trajectory)
        help_trajectory.layout().addWidget(self.le_trajectory)
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
        q_tracking.layout().addWidget(help_trajectory)
        q_tracking.layout().addWidget(help_remove_correspondence)
        q_tracking.layout().addWidget(help_insert_correspondence)

        # Evaluation UI
        q_eval = QWidget()
        q_eval.setLayout(QHBoxLayout())
        q_eval.layout().addWidget(metric)
        q_eval.layout().addWidget(c_plots)
        q_eval.layout().addWidget(btn_plot)

        # Add zones to self.toolbox
        self.toolbox.addItem(q_seg_track, "Data Processing")
        self.toolbox.addItem(q_segmentation, "Postprocessing Segmentation")
        self.toolbox.addItem(q_tracking, "Postprocessing Tracking")
        self.toolbox.addItem(q_eval, "Evaluation")

        # Assemble UI elements in ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setLayout(QVBoxLayout())
        scroll_area.layout().addWidget(title)
        scroll_area.layout().addWidget(q_load)
        scroll_area.layout().addWidget(self.toolbox)

        # Set ScrollArea as content of plugin
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    # Functions
    @napari.Viewer.bind_key('q')
    def _hotkey_load_zarr(self):
        MMVTracking.dock._load_zarr()
        
    def _load_zarr(self):
        dialog = QFileDialog()
        dialog.setNameFilter('*.zarr')
        self.file = dialog.getExistingDirectory(self, "Select Zarr-File")
        if(self.file == ""):
            print("No file selected")
            return
        self.z1 = zarr.open(self.file,mode='a')
        try:
            self.viewer.add_image(self.z1['raw_data'][:], name = 'Raw Image')
            s1 = self.viewer.add_labels(self.z1['segmentation_data'][:], name = 'Segmentation Data')
            self.viewer.add_tracks(self.z1['tracking_data'][:], name = 'Tracks') # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
        except:
            print("File is either no Zarr file or does not  adhere to required structure")
        """for layer in self.viewer.layers:
            @layer.mouse_drag_callbacks.append
            def _click(layer, event):
                print("Clicking on " + layer.name + " at + " + str(event.position) + "!")
                s1.selected_label = 2"""
    
    @napari.Viewer.bind_key('w')
    def _hotkey_save_zarr(self):
        MMVTracking.dock._save_zarr()

    def _save_zarr(self):
        # Useful if we later want to allow saving to new file
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
        #self.z1['raw_data'][:] = self.viewer.layers[raw].data
        self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
        self.z1['tracking_data'][:] = self.viewer.layers[track].data
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
                self.viewer.add_tracks(self.z1['tracking_data'][:], name='Tracks')
            else:
                tracks_data = [
                    track
                    for track in self.z1['tracking_data'][:]
                    if track[0] == id
                ]
                if not tracks_data:
                    print("No tracking data found for id " + str(id))
                    return
                self.viewer.add_tracks(tracks_data, name='Tracks')
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
                for track in self.z1['tracking_data'][:]
                if track[0] in id
            ]
            if not tracks_data:
                print("No tracking data found for ids " + str(id))
                return
            self.viewer.add_tracks(tracks_data, name='Tracks')

    @napari.Viewer.bind_key('e')
    def _hotkey_get_free_id(self):
        MMVTracking.dock._get_free_id()

    def _get_free_id(self):
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        label_layer.selected_label = (np.amax(label_layer.data)+1)
        napari.viewer.current_viewer().layers.select_all()
        napari.viewer.current_viewer().layers.selection.select_only(label_layer)
        label_layer.mode = "PAINT"

    @napari.Viewer.bind_key('r')
    def _hotkey_remove_fp(self):
        MMVTracking.dock._remove_fp()

    def _remove_fp(self):
        try:
            data = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        try:
            if np.count_nonzero(
                data[napari.viewer.current_viewer().dims.current_step[0]] == int(self.le_false_positive.text())
                ) < 1:
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
        np.place(
            data[napari.viewer.current_viewer().dims.current_step[0]],
            data[napari.viewer.current_viewer().dims.current_step[0]]==int(self.le_false_positive.text()),
            0
            )
        self.viewer.layers.pop(self.viewer.layers.index("Segmentation Data"))
        self.viewer.add_labels(data, name = 'Segmentation Data')
        self.viewer.layers.move(self.viewer.layers.index("Segmentation Data"),1)

    @napari.Viewer.bind_key('t')
    def _hotkey_false_merge(self):
        MMVTracking.dock._false_merge()

    def _false_merge(self):
        try:
            data = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        try:
            if np.count_nonzero(
                data[napari.viewer.current_viewer().dims.current_step[0]] == int(self.le_false_cut_1.text())
                ) < 1:
                msg = QMessageBox()
                msg.setText("ID doesn't exist in the current slice")
                msg.exec()
                return
        except ValueError:
            msg = QMessageBox()
            msg.setText("Please use an Integer (whole number) as ID")
            msg.exec()
            return
        # TODO: make functional
        # TODO: catch user error
        #self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((5,665,371),100)
        pass

    @napari.Viewer.bind_key('z')
    def _hotkey_false_cut(self):
        MMVTracking.dock._false_cut()

    def _false_cut(self):
        try:
            data = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        try:
            if np.count_nonzero(
                data[napari.viewer.current_viewer().dims.current_step[0]] == int(self.le_false_cut_1.text())
                ) < 1:
                msg = QMessageBox()
                msg.setText("ID doesn't exist in the current slice")
                msg.exec()
                return
        except ValueError:
            msg = QMessageBox()
            msg.setText("Please use an Integer (whole number) as ID")
            msg.exec()
            return
        # TODO: change colour!
        try:
            np.place(
                data[napari.viewer.current_viewer().dims.current_step[0]],
                data[napari.viewer.current_viewer().dims.current_step[0]]==int(self.le_false_cut_1.text()),
                int(self.le_false_cut_2.text())
            )
        except ValueError:
            msg = QMessageBox()
            msg.setText("Please use an Integer (whole number) as ID")
            msg.exec()

    @napari.Viewer.bind_key('1')
    def _hotkey_zone_1(self):
        MMVTracking.dock.toolbox.setCurrentIndex(0)

    @napari.Viewer.bind_key('2')
    def _hotkey_zone_2(self):
        MMVTracking.dock.toolbox.setCurrentIndex(1)

    @napari.Viewer.bind_key('3')
    def _hotkey_zone_3(self):
        MMVTracking.dock.toolbox.setCurrentIndex(2)

    @napari.Viewer.bind_key('4')
    def _hotkey_zone_4(self):
        MMVTracking.dock.toolbox.setCurrentIndex(3)