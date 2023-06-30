
import numpy as np
import pandas as pd

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
                            QGridLayout, QApplication, QMessageBox)
from qtpy.QtCore import Qt
from scipy import ndimage

from ._logger import notify, notify_with_delay, choice_dialog
from ._grabber import grab_layer

class TrackingWindow(QWidget):
    """
    A (QWidget) window to correct the tracking within the data.
    
    Attributes
    ----------
    
    Methods
    -------
    """
    
    def __init__(self, parent):
        """
        Parameters
        ----------
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Tracking correction")
        self.parent = parent
        self.viewer = parent.viewer
        
        ### QObjects
        
        # Labels
        label_trajectory = QLabel("Filter tracks by ID:")
        label_remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        label_insert_correspondence = QLabel("ID should be tracked with second ID:")
        
        # Buttons
        self.btn_remove_correspondence = QPushButton("Unlink")
        self.btn_remove_correspondence.setToolTip(
            "Remove cells from their tracks"
        )
        self.btn_remove_correspondence.clicked.connect(self._unlink)
        
        self.btn_insert_correspondence = QPushButton("Link")
        self.btn_insert_correspondence.setToolTip(
            "Add cells to new track"
        )
        self.btn_insert_correspondence.clicked.connect(self._link)
        
        btn_delete_displayed_tracks = QPushButton("Delete displayed tracks")
        btn_delete_displayed_tracks.clicked.connect(self._remove_displayed_tracks)
        btn_auto_track = QPushButton("Automatic tracking for single cell")
        btn_auto_track_all = QPushButton("Automatic tracking for all cells")
        
        # Line Edits
        self.lineedit_trajectory = QLineEdit("")
        self.lineedit_trajectory.editingFinished.connect(self._filter_tracks)
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(label_trajectory, 0, 0)
        content.layout().addWidget(self.lineedit_trajectory, 0, 1)
        content.layout().addWidget(btn_delete_displayed_tracks, 0, 2)
        content.layout().addWidget(label_remove_correspondence, 1, 0)
        content.layout().addWidget(self.btn_remove_correspondence, 1, 1)
        content.layout().addWidget(btn_auto_track, 1, 2)
        content.layout().addWidget(label_insert_correspondence, 2, 0)
        content.layout().addWidget(self.btn_insert_correspondence, 2, 1)
        content.layout().addWidget(btn_auto_track_all, 2, 2)
        
        self.layout().addWidget(content)
        
    def _filter_tracks(self):
        print("Filtering tracks")
        try:
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        input_text = self.lineedit_trajectory.text()
        if input_text == "":
            self._replace_tracks()
            return
        try:
            tracks = [int(input_text)]
        except ValueError:
            tracks = []
            split_input = input_text.split(",")
            try:
                for i in range(0,len(split_input)):
                    tracks.append(int((split_input[i])))
            except ValueError:
                notify("Please use a single integer (whole number) or a comma separated list of integers")
                return

        # Remove values < 0 and duplicates
        ids = filter(lambda value: value >= 0, tracks)
        ids = list(dict.fromkeys(ids))
        filtered_text = ""
        for i in range(0,len(ids)):
            if len(filtered_text) > 0:
                filtered_text += ","
            filtered_text = f'{filtered_text}{ids[i]}'
        self.lineedit_trajectory.setText(filtered_text)
        self._replace_tracks(ids)
    
    def _replace_tracks(self, ids = []):
        print("Displaying tracks {}".format(ids))
        if ids == []:
            self.viewer.add_tracks(self.parent.tracks, name = 'Tracks')
            return
        tracks_data = [
            track
            for track in self.parent.tracks
            if track[0] in ids
        ]
        if not tracks_data:
            print("No tracking data for ids " + str(tracks) + ", displaying all tracks instead")
            self.viewer.add_tracks(self.parent.tracks, name = 'Tracks')
            return
        self.viewer.add_tracks(tracks_data, name = 'Tracks')
        
    def _remove_displayed_tracks(self):
        print("Removing displayed tracks")
        try:
            tracks_layer = grab_layer(self.viewer, "Tracks")
        except ValueError:
            notify("No tracks layer found!")
            return
        
        to_remove = np.unique(tracks_layer.data[:,0])
        if np.array_equal(to_remove, np.unique(self.tracks[:,0])):
            notify("Can not delete whole tracks layer!")
            return
        
        ret = choice_dialog(
            "Are you sure? This will delete the following tracks: {}".format(to_remove),
            [("Continue", QMessageBox.AcceptRole), QMessageBox.Cancel]
        )
        # ret = 0 -> Continue, ret = 4194304 -> Cancel
        if ret == 4194304:
            return
        
        self.parent.tracks = np.delete(self.parent.tracks, np.isin(self.parent.tracks[:,0], to_remove), 0)
        self.viewer.layers.remove('Tracks')
        self.le_trajectroy.setText("")
        self.viewer.add_tracks(self.parent.tracks, name = 'Tracks')
        
    def _unlink(self):
        if self.btn_remove_correspondence.text() == "Unlink":
            self.btn_remove_correspondence.setText("Confirm")
            self.btn_insert_correspondence.setText("Link")
            self._remove_on_clicks()
            self._add_tracking_callback()
            return
            
        self.btn_remove_correspondence.setText("Unlink")
        self._remove_on_clicks()
        #print(self.track_cells)
        
        try:
            label_layer = grab_layer(self.viewer, "Segmentation Data")
        except ValueError:
            notify("Please make sure the label layer exists!")
            return
        
        try:
            tracks_layer = grab_layer(self.viewer, "Tracks")
        except ValueError:
            notify("Can not remove tracks when tracks layer is missing!")
            return
        
        tracks = tracks_layer.data
        new_track_id = np.amax(tracks[:,0]) + 1
        
        if len(self.track_cells) < 2:
            notify("Please select more than one cell to disconnect!")
            return
        
        track_id = -1
        for i in range(len(tracks)):
            if (tracks[i,1] == self.track_cells[0][0] and 
                tracks[i,2] == self.track_cells[0][1] and
                tracks[i,3] == self.track_cells[0][2]):
                if track_id != -1 and track_id != tracks[i,0]:
                    notify("Please select cells that are on the same track!")
                    return
                track_id = tracks[i,0]
        
        if track_id == -1:
            notify("Please select cells that are on any track!")
            return
        
        tracks_lists = [tracks, self.parent.tracks]
        cleaned_tracks = []
        for tracks_element in tracks_lists:
            selected_track = tracks_element[np.where(tracks_element[:,0] == track_id)]
            selected_track_bound_lower = selected_track[np.where(selected_track[:,1] > self.track_cells[0][0])]
            selected_track_to_delete = selected_track_bound_lower[np.where(selected_track_bound_lower[:,1] < self.track_cells[-1][0])]
            selected_track_to_reassign = selected_track[np.where(selected_track[:,1] >= self.track_cells[-1][0])]
            
            delete_indices = np.where(np.any(np.all(tracks_element[:, np.newaxis] == selected_track_to_delete, axis = 2), axis = 1))[0]
            tracks_filtered = np.delete(tracks_element, delete_indices, axis = 0)
    
            reassign_indices = np.where(np.any(np.all(tracks_filtered[:, np.newaxis] == selected_track_to_reassign, axis = 2), axis = 1))[0]
            tracks_filtered[reassign_indices, 0] = new_track_id
            
            df = pd.DataFrame(tracks_filtered, columns = ['ID', 'Z', 'Y', 'X'])
            df.sort_values(['ID', 'Z'], ascending = True, inplace = True)
            cleaned_tracks.append(df.values)
            
        tracks = cleaned_tracks[0]
        self.parent.tracks = cleaned_tracks[1]
        
        self.viewer.layers.remove('Tracks')
        if tracks.size > 0:
            self.viewer.add_tracks(tracks, name = 'Tracks')
        elif self.parent.tracks.size > 0:
            self.viewer.add_tracks(self.parent.tracks, name = 'Tracks')
        
    def _link(self):
        if self.btn_insert_correspondence.text() == "Link":
            self.btn_insert_correspondence.setText("Confirm")
            self.btn_remove_correspondence.setText("Unlink")
            self._remove_on_clicks()
            self._add_tracking_callback()
            return
            
        self.btn_insert_correspondence.setText("Link")
        self._remove_on_clicks()
        #print(self.track_cells)
        
        try:
            label_layer = grab_layer(self.viewer, "Segmentation Data")
        except ValueError:
            notify("Please make sure the label layer exists!")
            return
        
        try:
            tracks_layer = grab_layer(self.viewer, "Tracks")
        except ValueError:
            pass
        else:
            if not np.array_equal(tracks_layer, self.parent.tracks):
                ret = choice_dialog(
                    "All tracks need to be visible, but some tracks are hidden. Do you want to display them now?",
                    [("Display all", QMessageBox.AcceptRole), QMessageBox.Cancel]
                )
                # ret = 0 -> Display all, ret = 4194304 -> Cancel
                if ret == 4194304:
                    return
                self.lineedit_trajectory.setText("")
                self._replace_tracks()
        
        if len(self.track_cells) < 2:
            notify("Less than two cells can not be tracked")
            return
        
        if len(np.asarray(self.track_cells)[:,0]) != len(set(np.asarray(self.track_cells)[:,0])):
            notify("Looks like you selected more than one cell per slice. This makes the tracks freak out, so please don't do it. Thanks!")
            return
        
        if not 'tracks_layer' in locals():
            track_id = 1
        else:
            tracks = tracks_layer.data
            self.viewer.layers.remove('Tracks')
            track_id = np.amax(tracks[:,0]) + 1
            
        connected_ids = [0,0]
        if track_id != 1:
            for i in range(len(tracks)):
                if (tracks[i,1] == self.track_cells[0][0] and 
                    tracks[i,2] == self.track_cells[0][1] and
                    tracks[i,3] == self.track_cells[0][2]):
                    connected_ids[0] = self.tracks[i,0]
                    
                if (tracks[i,1] == self.track_cells[-1][0] and 
                    tracks[i,2] == self.track_cells[-1][1] and
                    tracks[i,3] == self.track_cells[-1][2]):
                    connected_ids[0] = self.tracks[i,0]
                    
        if max(old_ids) > 0:
            if old_ids[0] > 0:
                self.track_cells.remove(self.track_cells[0])
            if old_ids[1] > 0:
                self.track_cells.remove(self.track_cells[-1])
            
            if min(old_ids) == 0:
                track_id = max(old_ids)
            else:
                track_id = min(old_ids)
                tracks[np.where(tracks[:,0] == max(old_ids)), 0] = track_id
                
        for line in self.track_cells:
            try:
                tracks = np.r_[tracks, [[track_id] + line]]
            except UnboundLocalError:
                tracks = [[track_id] + line]
        
        df = pd.DataFrame(tracks, columns = ['ID', 'Z', 'Y', 'X'])
        df.sort_values(['ID', 'Z'], ascending = True, inplace = True)
        self.parent.tracks = df.values
        self.viewer.add_tracks(self.parent.tracks, name = 'Tracks')
        
    def _add_tracking_callback(self):
        QApplication.setOverrideCursor(Qt.CrossCursor)
        self.track_cells = []
        self._remove_on_clicks()
        for layer in self.viewer.layers:
            @layer.mouse_drag_callbacks.append
            def _select_cells(layer, event):
                try:
                    label_layer = grab_layer(self.viewer, "Segmentation Data")
                except ValueError:
                    notify("Please make sure the label layer exists!")
                    self._remove_on_clicks()
                    return
                z = int(event.position[0])
                selected_id = label_layer.data[
                    z,
                    int(event.position[1]),
                    int(event.position[2])
                ]
                if selected_id == 0:
                    worker = notify_with_delay("no clicky")
                    worker.start()
                    return
                centroid = ndimage.center_of_mass(
                    label_layer.data[z],
                    labels = label_layer.data[z],
                    index = selected_id
                )
                cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
                self.track_cells.append(cell)
                self.track_cells.sort()
                print("Added cell {} to list for track cells".format(cell))
        print("Added callback to record track cells")
        
    def _remove_on_clicks(self):
        for layer in self.viewer.layers:
            layer.mouse_drag_callbacks = []
        QApplication.restoreOverrideCursor()
        
        
        
        
        
        
        
        
        