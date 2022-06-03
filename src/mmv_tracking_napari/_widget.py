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
    unlink2 = 8

### TODO: Insert Manual Tracking
class MMVTracking(QWidget):
    dock = None
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        MMVTracking.dock = self

        # Variable to store clicked centroids for Tracking
        self.to_track = []

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

    def _mouse(self,mode,id = 0):
        for layer in self.viewer.layers:
            if len(layer.mouse_drag_callbacks):
                if layer.mouse_drag_callbacks[0].__name__ == "no_op":
                    layer.mouse_drag_callbacks.pop(-1)
                else:
                    layer.mouse_drag_callbacks.clear()

            if mode == State.default:
                pass
            elif mode == State.test:
                pass
            elif mode == State.remove: # False Positive
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
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),0)
                    self._mouse(State.default)
                    pass
            elif mode == State.recolour: # False Merge
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
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    # Label layer can't be missing as this is only called from False Cut 1
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),id)
                    self._mouse(State.default)
            elif mode == State.select: # Correct Segmentation
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
                    self._mouse(State.default)
            elif mode == State.link: # Creates Track
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
                        msg = QMessageBox()
                        msg.setText("Please select a segmented cell")
                        msg.exec()
                        self._link()
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self.to_track.append([int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
            elif mode == State.unlink: # Removes Track
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
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        msg = QMessageBox()
                        msg.setText("Please select a segmented cell") #TODO: this locks the layer to the mouse, FIX!
                        msg.exec()
                        self._mouse(State.unlink)
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self._mouse(State.unlink2, id=(int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))))
            elif mode == State.unlink2:
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    if id[0] == event.position:
                        msg = QMessageBox()
                        msg.setText("Please select a cell from a different slice")
                        msg.exec()
                        self._mouse(State.unlink2)
                        return
                    if id[0] < event.position[0]:
                        # change trackid in this & following layers
                        
                        pass
                    else:
                        # change trackid in following layers
                        pass
                    self._mouse(State.default)

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
            self.viewer.add_labels(self.z1['segmentation_data'][:], name = 'Segmentation Data')
            self.viewer.add_tracks(self.z1['tracking_data'][:], name = 'Tracks') # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
            return
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
        try:
            seg = self.viewer.layers.index("Segmentation Data")
        except ValueError:
            err = QMessageBox()
            err.setText("No Segmentation Data layer found!")
            err.exec()
            return
        try:
            track = self.viewer.layers.index("Tracks")
        except ValueError:
            err = QMessageBox()
            err.setText("No Tracks layer found!")
            err.exec()
            return
        #self.z1['raw_data'][:] = self.viewer.layers[raw].data
        self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
        self.z1['tracking_data'][:] = self.viewer.layers[track].data
        msg = QMessageBox()
        msg.setText("Zarr file has been saved.")
        msg.exec()

    def _temp(self):
        print(self.viewer.layers[2].data[0:5])

    def _plot(self):
        pass

    def _select_track(self):
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
            self._mouse(State.default)
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
        try:
            layer = self.viewer.layers[0]
        except ValueError:
            err = QMessageBox()
            err.setText("No layer found!")
            err.exec()
            return
        for i in range(len(layer.mouse_drag_callbacks)):
            if layer.mouse_drag_callbacks[i].__name__ == "_record": #TODO: insert logic to evaluate inputs & create/combine tracks
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
                    id = max(tracks[:,0]) + 1
                if id != 1:
                    for entry in self.to_track:
                        # try to find entry in tracking data
                        for j in range(len(layer.data)):
                            if tracks[j][1:3] == entry:
                                # adding to existing track
                                id = tracks[j][0]
                                self.to_track.remove(entry)
                                break
                for entry in self.to_track:
                    try:
                        tracks = np.r_[tracks, [[id] + entry]]
                    except UnboundLocalError:
                        tracks = [[id] + entry]
                self.to_track = []
                df = pd.DataFrame(tracks, columns=['ID', 'Z', 'Y', 'X'])
                df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                self.viewer.add_tracks(df.values, name='Tracks')
                self._mouse(State.default)
                return
        self._mouse(State.link)

    @napari.Viewer.bind_key('i')
    def _hotkey_unlink(self):
        MMVTracking.dock._unlink()

    def _unlink(self):
        self._mouse(State.unlink)
            

    @napari.Viewer.bind_key('a')
    def _hotkey_grab_label(self):
        MMVTracking.dock._grab_label()
        MMVTracking.dock.viewer.layers[MMVTracking.dock.viewer.layers.index("Segmentation Data")].mode = "PAINT"

    def _grab_label(self):
        self._mouse(State.select)

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
