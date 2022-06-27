import enum

import napari
import numpy as np
import pandas as pd
import zarr
from qtpy.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                            QLabel, QLineEdit, QMessageBox, QPushButton,
                            QScrollArea, QToolBox, QVBoxLayout, QWidget)
from scipy import ndimage


class State(enum.Enum):
    test = -1
    default =  0
    remove = 1
    recolour = 2
    merge_from = 3
    merge_to = 4
    select = 5
    link = 6
    unlink = 7

class MMVTracking(QWidget):
    dock = None
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        MMVTracking.dock = self

        # Variables to store clicked centroids for Tracking
        self.to_track = []
        self.to_cut = []

        # Variable to hold complete (corrected) tracks layer
        self.tracks = np.zeros((1,4))

        # Labels
        title = QLabel("<font color='green'>HITL4Trk</font>")
        next_free = QLabel("Next free label:")
        trajectory = QLabel("Select ID for trajectory:")
        load_save = QLabel("Load/Save .zarr file:")
        false_positive = QLabel("Remove false positive for ID:")
        false_merge = QLabel("Cut falsely merged ID:")
        false_cut = QLabel("Merge falsely cut ID into second ID:")
        remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        insert_correspondence = QLabel("ID should be tracked with second ID:")
        metric = QLabel("Evaluation metrics:")
        grab_label = QLabel("Select label:")

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
        btn_remove_correspondence.setToolTip("I")
        btn_insert_correspondence = QPushButton("Link")
        btn_insert_correspondence.setToolTip("U")
        btn_save = QPushButton("Save")
        btn_save.setToolTip("W")
        btn_plot = QPushButton("Plot")
        btn_segment = QPushButton("Run instance segmentation")
        btn_track = QPushButton("Run tracking")
        btn_free_label = QPushButton("Load Label")
        btn_free_label.setToolTip("E")
        btn_grab_label = QPushButton("Select")
        btn_grab_label.setToolTip("A")

        # Linking buttons to functions
        btn_load.clicked.connect(self._load_zarr)
        btn_plot.clicked.connect(self._plot)
        btn_save.clicked.connect(self._save_zarr)
        btn_false_positive.clicked.connect(self._remove_fp)
        btn_segment.clicked.connect(self._temp)
        btn_false_merge.clicked.connect(self._false_merge)
        btn_free_label.clicked.connect(self._set_free_id)
        btn_false_cut.clicked.connect(self._false_cut)
        btn_grab_label.clicked.connect(self._grab_label)
        btn_remove_correspondence.clicked.connect(self._unlink)
        btn_insert_correspondence.clicked.connect(self._link)
       
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
        help_false_positive.layout().addWidget(btn_false_positive)
        help_false_negative = QWidget()
        help_false_negative.setLayout(QHBoxLayout())
        help_false_negative.layout().addWidget(next_free)
        help_false_negative.layout().addWidget(btn_free_label)
        help_false_merge = QWidget()
        help_false_merge.setLayout(QHBoxLayout())
        help_false_merge.layout().addWidget(false_merge)
        help_false_merge.layout().addWidget(btn_false_merge)
        help_false_cut = QWidget()
        help_false_cut.setLayout(QHBoxLayout())
        help_false_cut.layout().addWidget(false_cut)
        help_false_cut.layout().addWidget(btn_false_cut)
        help_grab_layer = QWidget()
        help_grab_layer.setLayout(QHBoxLayout())
        help_grab_layer.layout().addWidget(grab_label)
        help_grab_layer.layout().addWidget(btn_grab_label)
        q_segmentation = QWidget()
        q_segmentation.setLayout(QVBoxLayout())
        q_segmentation.layout().addWidget(help_grab_layer)
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
        help_remove_correspondence.layout().addWidget(btn_remove_correspondence)
        help_insert_correspondence = QWidget()
        help_insert_correspondence.setLayout(QHBoxLayout())
        help_insert_correspondence.layout().addWidget(insert_correspondence)
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
        self.toolbox.addItem(q_segmentation, "Segmentation correction")
        self.toolbox.addItem(q_tracking, "Tracking correction")
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
        self.setMinimumWidth(400)

        self._mouse(State.default)

    # Functions

    def _mouse(self,mode,id = 0, paint = False):
        for layer in self.viewer.layers:
            if len(layer.mouse_drag_callbacks):
                if layer.mouse_drag_callbacks[0].__name__ == "no_op":
                    layer.mouse_drag_callbacks.pop(-1)
                else:
                    layer.mouse_drag_callbacks.clear()

            if mode == State.default:
                self.viewer.layers.selection.active.help = "(0)"
            elif mode == State.test:
                self.viewer.layers.selection.active.help = "(-1)"
            elif mode == State.remove: # False Positive
                self.viewer.layers.selection.active.help = "(1)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "PAN_ZOOM"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    try:
                        self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                        return
                    false_id = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    np.place(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0])],self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0])]==false_id,0)
                    self._mouse(State.default)
                    pass
            elif mode == State.recolour: # False Merge
                self.viewer.layers.selection.active.help = "(2)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "PAN_ZOOM"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    try:
                        self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                        return
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),self._get_free_id(self.viewer.layers[self.viewer.layers.index("Segmentation Data")]))
                    self._mouse(State.default)
            elif mode == State.merge_from: # False Cut 1
                self.viewer.layers.selection.active.help = "(3)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "PAN_ZOOM"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                        return
                    self._mouse(State.merge_to, label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])])
            elif mode == State.merge_to: # False Cut 2
                self.viewer.layers.selection.active.help = "(4)"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    # Label layer can't be missing as this is only called from False Cut 1
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),id)
                    self._mouse(State.default)
            elif mode == State.select: # Correct Segmentation
                self.viewer.layers.selection.active.help = "(5)"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                        return
                    label_layer.selected_label = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    napari.viewer.current_viewer().layers.select_all()
                    napari.viewer.current_viewer().layers.selection.select_only(label_layer)
                    if paint:
                        label_layer.mode = "PAINT"
                    self._mouse(State.default)
            elif mode == State.link: # Creates Track
                self.viewer.layers.selection.active.help = "(6)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "PAN_ZOOM"
                @layer.mouse_drag_callbacks.append
                def _record(layer,event):
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                        return
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        self.viewer.layers.selection.active.help = "YOU MISSED THE CELL, PRESS THE BUTTON AGAIN AND CONTINUE FROM THE LAST VALID INPUT!"
                        self._link()
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self.to_track.append([int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))])

            elif mode == State.unlink: # Removes Track
                self.viewer.layers.selection.active.help = "(7)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "PAN_ZOOM"
                @layer.mouse_drag_callbacks.append
                def _cut(layer,event):
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        msg = QMessageBox()
                        msg.setText("Missing label layer")
                        msg.exec()
                        self._mouse(State.default)
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        self.viewer.layers.selection.active.help = "NO CELL SELECTED, DO BETTER NEXT TIME!"
                        self._mouse(State.default)
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self.to_cut.append([int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))])

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

        # check if "Raw Image", "Segmentation Data" or "Track" exist in self.viewer.layers
        if "Raw Image" in self.viewer.layers or "Segmentation Data" in self.viewer.layers or "Tracks" in self.viewer.layers:
            msg = QMessageBox()
            msg.setWindowTitle("Layer name blocked")
            msg.setText("Found layer name")
            msg.setInformativeText("One or more layers with the names \"Raw Image\", \"Segmentation Data\" or \"Tracks\" exists already. Continuing will delete those layers. Are you sure?")
            msg.addButton("Continue", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Cancel)
            ret = msg.exec() # ret = 0 means Continue was selected, ret = 4194304 means Cancel was selected
            if ret == 4194304:
                return
            try:
                self.viewer.layers.remove("Raw Image")
                self.viewer.layers.remove("Segmentation Data")
                self.viewer.layers.remove("Tracks")
            except ValueError: # only one or two layers may exist, so not all can be deleted
                pass
        try:
            self.viewer.add_image(self.z1['raw_data'][:], name = 'Raw Image')
            self.viewer.add_labels(self.z1['segmentation_data'][:], name = 'Segmentation Data')
            self.viewer.add_tracks(self.z1['tracking_data'][:], name = 'Tracks') # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
        except:
            print("File is either no Zarr file or does not adhere to required structure")
        self._mouse(State.default)
    
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
        try: # Check if segmentation layer exists
            seg = self.viewer.layers.index("Segmentation Data")
        except ValueError:
            err = QMessageBox()
            err.setText("No Segmentation Data layer found!")
            err.exec()
            return
        try: # Check if tracks layer exists
            track = self.viewer.layers.index("Tracks")
        except ValueError:
            err = QMessageBox()
            err.setText("No Tracks layer found!")
            err.exec()
            return

        ret = 1
        if self.le_trajectory.text() != "": # Some tracks are potentially left out
            msg = QMessageBox()
            msg.setWindowTitle("Tracks")
            msg.setText("Limited Tracks layer")
            msg.setInformativeText("It looks like you have selected only some of the tracks from your tracks layer. Do you want to save only the selected ones or all of them?") # ok clippy
            msg.addButton("Save Selected",QMessageBox.YesRole)
            msg.addButton("Save All",QMessageBox.NoRole)
            msg.addButton(QMessageBox.Cancel)
            ret = msg.exec() # Save Selected -> ret = 0, Save All -> ret = 1, Cancel -> ret = 4194304
            if ret == 4194304:
                return
        if ret == 0: # save current tracks layer
            #self.z1['raw_data'][:] = self.viewer.layers[raw].data
            self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
            self.z1.create_dataset('tracking_data', shape = self.viewer.layers[track].data.shape, dtype = 'i4', data = self.viewer.layers[track].data)
        else: # save complete tracks layer
            #self.z1['raw_data'][:] = self.viewer.layers[raw].data
            self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
            self.z1.create_dataset('tracking_data', shape = self.viewer.layers[track].data.shape, dtype = 'i4', data = self.tracks)
        msg = QMessageBox()
        msg.setText("Zarr file has been saved.")
        msg.exec()

    def _temp(self):
        #print(dir(napari.viewer.current_viewer().layers.selection))
        #napari.viewer.current_viewer().layers.selection.toggle(self.viewer.layers[0])
        print(self.viewer.layers.selection.active)

    def _plot(self):
        pass

    def _select_track(self):
        if self.le_trajectory.text() == "": # deleting the text returns the whole layer
            try:
                self.viewer.layers.remove('Tracks')
            except ValueError:
                print("No tracking layer found")
            self.viewer.add_tracks(self.tracks, name='Tracks')
            return
        try: # Try for single value
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
            self.tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        if isinstance(id,int): # Single value
            if id < 0:
                self.viewer.add_tracks(self.tracks, name='Tracks')
                self.le_trajectory.setText("") # No need to keep negative number
            else:
                tracks_data = [
                    track
                    for track in self.tracks
                    if track[0] == id
                ]
                if not tracks_data:
                    print("No tracking data found for id " + str(id))
                    return
                self.viewer.add_tracks(tracks_data, name='Tracks')
            self._mouse(State.default)
        else: # Multiple values, id is instance of "list"
            id = list(dict.fromkeys(id)) # Removes duplicate values
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
                for track in self.tracks
                if track[0] in id
            ]
            if not tracks_data:
                print("No tracking data found for ids " + str(id))
                return
            self.viewer.add_tracks(tracks_data, name='Tracks')
            self._mouse(State.default)

    @napari.Viewer.bind_key('e')
    def _hotkey_get_free_id(self):
        MMVTracking.dock._set_free_id()
        MMVTracking.dock.viewer.layers[MMVTracking.dock.viewer.layers.index("Segmentation Data")].mode = "PAINT"

    def _set_free_id(self):
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            msg = QMessageBox()
            msg.setText("Missing label layer")
            msg.exec()
            return
        label_layer.selected_label = self._get_free_id(label_layer)
        napari.viewer.current_viewer().layers.select_all()
        napari.viewer.current_viewer().layers.selection.select_only(label_layer)

    def _get_free_id(self, layer):
        return np.amax(layer.data)+1

    @napari.Viewer.bind_key('r')
    def _hotkey_remove_fp(self):
        MMVTracking.dock._remove_fp()

    def _remove_fp(self):
        self._mouse(State.remove)

    @napari.Viewer.bind_key('t')
    def _hotkey_false_merge(self):
        MMVTracking.dock._false_merge()

    def _false_merge(self):
        self._mouse(State.recolour)

    @napari.Viewer.bind_key('z')
    def _hotkey_false_cut(self):
        MMVTracking.dock._false_cut()

    def _false_cut(self):
        self._mouse(State.merge_from)

    # Tracking correction
    @napari.Viewer.bind_key('u')
    def _hotkey_unlink(self):
        MMVTracking.dock._link()

    def _link(self):
        try: # check if segmentation layer exists
            layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            err = QMessageBox()
            err.setText("No label layer found!")
            err.exec()
            return
        for i in range(len(layer.mouse_drag_callbacks)):
            if layer.mouse_drag_callbacks[i].__name__ == "_record":
                if len(self.to_track) < 2:
                    self.to_track = []
                    self._mouse(State.default)
                    return
                self.to_track.sort()
                try:
                    track = self.viewer.layers.index("Tracks")
                except ValueError:
                    id = 1
                else:
                    tracks = self.viewer.layers[track].data
                    self.viewer.layers.remove('Tracks')
                    id = max(np.amax(tracks[:,0]),np.amax(self.tracks[:,0])) + 1
                old_ids = [0,0]
                if id != 1: # tracking data is not empty
                    for j in range(len(tracks)):
                        if tracks[j][1] == self.to_track[0][0] and tracks[j][2] == self.to_track[0][1] and tracks[j][3] == self.to_track[0][2]: # new track starting point exists in tracking data
                            old_ids[0] = tracks[j][0]
                            self.to_track.remove(self.to_track[0])
                            break
                    for j in range(len(tracks)):
                        if tracks[j][1] == self.to_track[-1][0] and tracks[j][2] == self.to_track[-1][1] and tracks[j][3] == self.to_track[-1][2]: # new track end point exists in tracking data
                            old_ids[1] = tracks[j][0]
                            self.to_track.remove(self.to_track[-1])
                            break
                if max(old_ids) > 0:
                    if min(old_ids) == 0: # one end connects to existing track
                        id = max(old_ids)
                    else: # both ends connect to existing track, (higher) id of second existing track changed to id of first track
                        id = min(old_ids)
                        for track_entry in tracks:
                            if track_entry[0] == max(old_ids):
                                track_entry[0] = id
                for entry in self.to_track: # entries are added to tracking data (current and cached, in case those are different)
                    try:
                        tracks = np.r_[tracks, [[id] + entry]]
                    except UnboundLocalError:
                        tracks = [[id] + entry]
                    try:
                        self.tracks = np.r_[tracks, [[id] + entry]]
                    except UnboundLocalError:
                        self.tracks = [[id] + entry]
                self.to_track = []
                df = pd.DataFrame(tracks, columns=['ID', 'Z', 'Y', 'X'])
                df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                self.viewer.add_tracks(df.values, name='Tracks')
                self._mouse(State.default)
                return
        self.viewer.layers.selection.active.help = ""
        self.to_track = []
        self._mouse(State.link)

    @napari.Viewer.bind_key('i')
    def _hotkey_unlink(self):
        MMVTracking.dock._unlink()

    def _unlink(self):
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            err = QMessageBox()
            err.setText("No label layer found!")
            err.exec()
            return
        try:
            tracks_layer = self.viewer.layers[self.viewer.layers.index("Tracks")]
        except ValueError:
            err = QMessageBox()
            err.setText("No tracks layer found!")
            err.exec()
            return
        id = max(np.amax(tracks_layer.data[:,0]),np.amax(self.tracks[:,0])) + 1
        tracks = tracks_layer.data
        for i in range(len(label_layer.mouse_drag_callbacks)):
            if label_layer.mouse_drag_callbacks[i].__name__ == "_cut":
                if len(self.to_cut) < 2:
                    msg = QMessageBox()
                    msg.setText("Please select more than one cell!")
                    msg.exec()
                    self.to_cut = []
                    self._mouse(State.default)
                    return
                self.to_cut.sort()
                track = 0
                for j in range(len(tracks_layer.data)): # find track id
                    if tracks[j,1] == self.to_cut[0][0] and tracks[j,2] == self.to_cut[0][1] and tracks[j,3] == self.to_cut[0][2]:
                        track = tracks[j,0]
                        break
                for j in range(len(tracks_layer.data)):  # confirm track id
                    if tracks[j,1] == self.to_cut[-1][0] and tracks[j,2] == self.to_cut[-1][1] and tracks[j,3] == self.to_cut[-1][2]:
                        if track != tracks[j,0]:
                            msg = QMessageBox()
                            msg.setText("Please select cells that belong to the same Track!")
                            msg.exec()
                            self.to_cut = []
                            self._mouse(State.default)
                            return
                j = 0
                while j < len(tracks):
                    if tracks[j,0] == track:
                        if tracks[j,1] > self.to_cut[0][0]:
                            if tracks[j,1] < self.to_cut[-1][0]: # cells to remove from tracking
                                tracks = np.delete(tracks,j,0)
                                j = j - 1
                            elif tracks[j,1] >= self.to_cut[-1][0]: # cells to track with new id
                                tracks[j,0] = id
                    j = j + 1
                self.to_cut = []
                df = pd.DataFrame(tracks, columns=['ID', 'Z', 'Y', 'X'])
                df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                tracks = df.values
                tmp = np.unique(tracks[:,0],return_counts = True) # count occurences of each id
                tmp = np.delete(tmp,tmp[1] == 1,1)
                tracks = np.delete(tracks,np.where(np.isin(tracks[:,0],tmp[0,:],invert=True)),0)
                self.tracks = np.delete(self.tracks,np.where(np.isin(tracks[:,0],tmp[0,:],invert=True)),0) # TEST IF THIS WORKS FOR ISSUE 16
                self.viewer.layers.remove('Tracks')
                self.viewer.add_tracks(tracks, name='Tracks')
                self._mouse(State.default)
                return
        self.to_cut = []
        self._mouse(State.unlink)
            
    @napari.Viewer.bind_key('x')
    def _default_hotkey(self):
        MMVTracking.dock._default()

    def _default(self):
        self._mouse(State.default)

    @napari.Viewer.bind_key('a')
    def _hotkey_grab_label(self):
        MMVTracking.dock._grab_label(paint = True)

    def _grab_label(self, paint = False):
        self.viewer.layers[self.viewer.layers.index("Segmentation Data")].mode = "PAN_ZOOM"
        self._mouse(State.select, paint = paint)

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
