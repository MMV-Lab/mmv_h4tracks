import multiprocessing
from multiprocessing import Pool
from threading import Event

import napari
import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)
from scipy import ndimage, stats

from ._logger import notify, notify_with_delay, choice_dialog, handle_exception
from ._grabber import grab_layer
from ._logger import choice_dialog, notify, notify_with_delay
import mmv_hitl4trk._processing as processing

LINK_TEXT = "Link tracks"
UNLINK_TEXT = "Unlink tracks"
CONFIRM_TEXT = "Confirm"


class TrackingWindow(QWidget):
    """
    A (QWidget) window to correct the tracking within the data.

    Attributes
    ----------

    Methods
    -------
    """

    MIN_TRACK_LENGTH = 5

    def __init__(self, parent):
        """
        Parameters
        ----------
        parent : QWidget
        Parent widget for the tracking
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.parent = parent
        self.viewer = parent.viewer
        self.choice_event = Event()
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        self.cached_callback = None

        ### QObjects

        # Labels
        label_display_ids = QLabel("Enter specific track IDs to display:")
        label_delete_specific_ids = QLabel("Delete specified tracks:")

        # Buttons
        btn_centroid_tracking = QPushButton("Coordinate-based tracking")
        btn_centroid_tracking_tooltip = (
            "Start coordinate-based tracking for all slices\n"
            "Faster than the overlap-based tracking\n"
            "More tracks will be created, can track complex migration\n"
            "Tracks may jump between cells if given imperfect segmentation"
        )
        btn_centroid_tracking.setToolTip(btn_centroid_tracking_tooltip)
        btn_auto_track_all = QPushButton("Overlap-based tracking")
        btn_auto_track_all_tooltip = (
            "Start overlap-based tracking for all slices\n"
            "Slower than the coordinate-based tracking\n"
            "Less tracks will be created, but tracks are less likely to be incorrect\n"
        )
        btn_auto_track_all.setToolTip(btn_auto_track_all_tooltip)
        btn_auto_track = QPushButton(
            "Overlap-based tracking (single cell)"
        )  # TODO: clicking an already tracked cell breaks the onclicks/cursor
        btn_auto_track.setToolTip(
            "Click on a cell to track based on overlap \n\n"
            "Hotkey: S"
            )

        self.btn_remove_correspondence = QPushButton(UNLINK_TEXT)
        self.btn_remove_correspondence.setToolTip("Remove cells from their tracks")

        self.btn_insert_correspondence = QPushButton(LINK_TEXT)
        self.btn_insert_correspondence.setToolTip("Add cells to new track")

        btn_delete_displayed_tracks = QPushButton("Delete all displayed tracks")
        btn_filter_tracks = QPushButton("Filter")
        btn_delete_selected_tracks = QPushButton("Delete")

        btn_centroid_tracking.clicked.connect(self._run_tracking)
        btn_auto_track_all.clicked.connect(self._proximity_track_all)
        btn_auto_track.clicked.connect(self._add_auto_track_callback)
        self.btn_remove_correspondence.clicked.connect(self._unlink)
        self.btn_insert_correspondence.clicked.connect(self._link)
        btn_delete_displayed_tracks.clicked.connect(self._remove_displayed_tracks)
        btn_filter_tracks.clicked.connect(self._filter_tracks)
        btn_delete_selected_tracks.clicked.connect(self._delete_selected)

        # Line Edits
        self.lineedit_filter = QLineEdit("")
        self.lineedit_delete = QLineEdit("")
        self.lineedit_filter.returnPressed.connect(self._filter_tracks)

        # Spacers
        v_spacer = QWidget()
        v_spacer.setFixedWidth(4)
        v_spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(0)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_2 = QWidget()
        h_spacer_2.setFixedHeight(0)
        h_spacer_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_3 = QWidget()
        h_spacer_3.setFixedHeight(0)
        h_spacer_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_4 = QWidget()
        h_spacer_4.setFixedHeight(20)
        h_spacer_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        automatic_tracking = QGroupBox("Automatic tracking")
        automatic_tracking.setLayout(QGridLayout())
        automatic_tracking.layout().addWidget(h_spacer_1, 0, 0, 1, -1)
        automatic_tracking.layout().addWidget(btn_centroid_tracking, 1, 0)
        automatic_tracking.layout().addWidget(btn_auto_track_all, 2, 0)
        automatic_tracking.layout().addWidget(btn_auto_track, 2, 1)

        tracking_correction = QGroupBox("Tracking correction")
        tracking_correction.setLayout(QGridLayout())
        tracking_correction.layout().addWidget(h_spacer_2, 0, 0, 1, -1)
        tracking_correction.layout().addWidget(self.btn_insert_correspondence, 1, 0)
        tracking_correction.layout().addWidget(self.btn_remove_correspondence, 1, 1)

        filter_tracks = QGroupBox("Visualize && filter tracks")
        filter_tracks.setLayout(QGridLayout())
        filter_tracks.layout().addWidget(h_spacer_3, 0, 0, 1, -1)
        filter_tracks.layout().addWidget(label_display_ids, 2, 0)
        filter_tracks.layout().addWidget(self.lineedit_filter, 2, 1)
        filter_tracks.layout().addWidget(btn_filter_tracks, 2, 2)
        filter_tracks.layout().addWidget(h_spacer_4, 3, 2, 1, -1)
        filter_tracks.layout().addWidget(label_delete_specific_ids, 4, 0)
        filter_tracks.layout().addWidget(self.lineedit_delete, 4, 1)
        filter_tracks.layout().addWidget(btn_delete_selected_tracks, 4, 2)
        filter_tracks.layout().addWidget(btn_delete_displayed_tracks, 5, 0, 1, -1)

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())

        content.layout().addWidget(automatic_tracking)
        content.layout().addWidget(tracking_correction)
        content.layout().addWidget(filter_tracks)
        content.layout().addWidget(v_spacer)

        self.layout().addWidget(content)

    def _delete_selected(self):
        """
        Deletes the tracks specified in the lineedit_delete text field
        """
        input_text = self.lineedit_delete.text()
        if input_text == "":
            return
        try:
            tracks_to_delete = [int(input_text)]
        except ValueError:
            tracks_to_delete = []
            split_input = input_text.split(",")
            try:
                for i in range(0, len(split_input)):
                    tracks_to_delete.append(int((split_input[i])))
            except ValueError:
                notify(
                    "Please use a single integer (whole number) or a comma separated list of integers"
                )
                return

        ids = filter(lambda value: value >= 0, tracks_to_delete)
        ids = list(set(ids))

        filter_text = self.lineedit_filter.text().split(",")
        try:
            visible_tracks = [int(track) for track in filter_text]
        except ValueError:
            visible_tracks = np.unique(self.parent.tracks[:, 0]).tolist()

        if visible_tracks:
            # if trying to delete non-displayed tracks, ask for confirmation
            if not set(ids).issubset(set(visible_tracks)):
                if not set(ids).issubset(set(np.unique(self.parent.tracks[:, 0]))):
                    wrong_ids = [
                        value for value in ids if value not in self.parent.tracks[:, 0]
                    ]
                    notify(
                        f"There are no tracks with id(s) {wrong_ids}, the others will be deleted."
                    )
                else:
                    ret = choice_dialog(
                        "Some of the tracks you are trying to delete are not displayed. Do you want to delete them anyway?",
                        [("Delete anyway", QMessageBox.AcceptRole), QMessageBox.Cancel],
                    )
                    # ret = 0 -> Delete anyway, ret = 4194304 -> Cancel
                    if ret == 4194304:
                        return
            for track in ids:
                # if visible tracks are deleted, remove their entry from the filter text
                if track in visible_tracks:
                    visible_tracks.remove(track)

            self.parent.tracks = np.delete(
                self.parent.tracks, np.isin(self.parent.tracks[:, 0], ids), 0
            )
            if not visible_tracks or np.array_equal(
                visible_tracks, np.unique(self.parent.tracks[:, 0])
            ):
                # if all visible tracks are deleted, show all tracks again
                self.lineedit_filter.setText("")
                self._replace_tracks()
            else:
                # if only some visible tracks are deleted, update filter text
                filtered_text = ""
                for i in range(0, len(visible_tracks)):
                    if len(filtered_text) > 0:
                        filtered_text += ","
                    filtered_text = f"{filtered_text}{visible_tracks[i]}"
                self.lineedit_filter.setText(filtered_text)
                self._replace_tracks(visible_tracks)

        self.lineedit_delete.setText("")

    def _run_tracking(self):
        """
        Calls the centroid based tracking function
        """

        def on_yielded(value):
            """
            Prompts the user to replace the tracks layer if it already exists
            """
            if value == "Replace tracks layer":
                ret = choice_dialog(
                    "Tracks layer found. Do you want to replace it?",
                    [QMessageBox.Yes, QMessageBox.No],
                )
                self.ret = ret
                if ret == 65536:
                    worker.quit()
                self.choice_event.set()

        worker = processing._track_segmentation(self)
        worker.returned.connect(processing._add_tracks_to_viewer)
        worker.yielded.connect(on_yielded)

    def _filter_tracks(self):
        """
        Filters the tracks layer to only display the selected tracks
        """
        input_text = self.lineedit_filter.text()
        if input_text == "":
            self._replace_tracks()
            return
        try:
            tracks = [int(input_text)]
        except ValueError:
            tracks = []
            split_input = input_text.split(",")
            try:
                for i in range(0, len(split_input)):
                    tracks.append(int((split_input[i])))
            except ValueError:
                notify(
                    "Please use a single integer (whole number) or a comma separated list of integers"
                )
                return

        # Remove values < 0 and duplicates
        ids = filter(lambda value: value >= 0, tracks)
        ids = list(set(ids))
        filtered_text = ""
        for i in range(0, len(ids)):
            if len(filtered_text) > 0:
                filtered_text += ","
            filtered_text = f"{filtered_text}{ids[i]}"
        self.lineedit_filter.setText(filtered_text)
        self._replace_tracks(ids)

    def _replace_tracks(self, selected_ids: list = None):
        """
        Replaces the tracks layer with the selected tracks

        Parameters
        ----------
        selected_ids : list
            The ids of the tracks to display
        """
        if selected_ids is None:
            ids = []
        else:
            ids = selected_ids

        tracks_name = self.parent.combobox_tracks.currentText()
        try:
            tracks_layer = grab_layer(self.viewer, tracks_name)
        except ValueError as exc:
            handle_exception(exc)
            return

        if len(ids) == 0:
            tracks_layer.data = self.parent.tracks
            return
        tracks_data = [track for track in self.parent.tracks if track[0] in ids]
        if not tracks_data:
            tracks_layer.data = self.parent.tracks
            return
        filtered_text = ""
        for i in range(0, len(ids)):
            if len(filtered_text) > 0:
                filtered_text += ","
            filtered_text = f"{filtered_text}{ids[i]}"
        self.lineedit_filter.setText(filtered_text)
        tracks_layer.data = tracks_data

    def _remove_displayed_tracks(self):
        """
        Removes the displayed tracks from the tracks layer and the cached tracks"""
        tracks_name = self.parent.combobox_tracks.currentText()
        try:
            tracks_layer = grab_layer(self.viewer, tracks_name)
        except ValueError as exc:
            handle_exception(exc)
            return

        to_remove = np.unique(tracks_layer.data[:, 0])
        if np.array_equal(to_remove, np.unique(self.parent.tracks[:, 0])):
            notify("Can not delete whole tracks layer!")
            return

        ret = choice_dialog(
            "Are you sure? This will delete the following tracks: {}".format(to_remove),
            [("Continue", QMessageBox.AcceptRole), QMessageBox.Cancel],
        )
        # ret = 0 -> Continue, ret = 4194304 -> Cancel
        if ret == 4194304:
            return

        self.parent.tracks = np.delete(
            self.parent.tracks, np.isin(self.parent.tracks[:, 0], to_remove), 0
        )
        tracks_layer.data = self.parent.tracks
        self.lineedit_filter.setText("")

    def _unlink(self):
        """
        Calls the unlink function to prepare or perform the unlink
        """
        if self.btn_remove_correspondence.text() == UNLINK_TEXT:
            self._prepare_for_unlink()
        else:
            self._perform_unlink()

    def _prepare_for_unlink(self):
        """
        Prepares the unlink by adding a callback to the viewer
        """
        self._add_tracking_callback()
        self.btn_remove_correspondence.setText(CONFIRM_TEXT)

    def _perform_unlink(self):
        """
        Performs the unlink by removing the selected cells from their tracks
        """
        self._update_callbacks()

        label_layer = self._get_layer(self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            return

        tracks_layer = self._get_layer(self.parent.combobox_tracks.currentText())
        if tracks_layer is None:
            return

        if len(self.track_cells) < 2:
            notify("Please select more than one cell to disconnect!")
            return

        if len(np.asarray(self.track_cells)[:, 0]) != len(
            set(np.asarray(self.track_cells)[:, 0])
        ):
            most_common_value = stats.mode(np.array(self.track_cells)[:, 0])[0]
            notify(
                f"Looks like you selected multiple cells in slice {most_common_value}. You can only remove cells from one track at a time."
            )
            return

        track_id = self._get_track_id(tracks_layer.data)
        if track_id is None:
            return

        self._update_tracks(tracks_layer, track_id)

    def _get_layer(self, layer_name):
        """
        Returns the layer with the specified name

        Parameters
        ----------
        layer_name : str
            The name of the layer to return

        Returns
        -------
        layer : napari layer
            The layer with the specified name
        """
        try:
            return grab_layer(self.viewer, layer_name)
        except ValueError as exc:
            handle_exception(exc)
            return None

    def _get_track_id(self, tracks):
        """
        Returns the track id of the selected cells

        Parameters
        ----------
        tracks : np.ndarray
            The tracks to check for the selected cells

        Returns
        -------
        track_id : int
            The track id of the selected cells
        """
        track_id = -1
        for i in range(len(tracks)):
            if (
                tracks[i, 1] == self.track_cells[0][0]
                and tracks[i, 2] == self.track_cells[0][1]
                and tracks[i, 3] == self.track_cells[0][2]
            ):
                if track_id != -1 and track_id != tracks[i, 0]:
                    notify("Please select cells that are on the same track!")
                    return None
                track_id = tracks[i, 0]

        if track_id == -1:
            notify("Please select cells that are on any track!")
            return None

        return track_id

    def _update_tracks(self, tracks_layer, track_id):
        """
        Updates the tracks layer and the cached tracks by only displaying the selected tracks

        Parameters
        ----------
        tracks_layer : napari layer
            The tracks layer to update
        track_id : int
            The track id of the selected cells
        """
        tracks_list = [tracks_layer.data, self.parent.tracks]
        cleaned_tracks = []
        for tracks_element in tracks_list:
            cleaned_tracks.append(self._clean_tracks(tracks_element, track_id))

        if cleaned_tracks[0].size > 0:
            tracks_layer.data = cleaned_tracks[0]
        elif cleaned_tracks[1].size > 0:
            tracks_layer.data = cleaned_tracks[1]
        self.parent.tracks = cleaned_tracks[1]

    def _clean_tracks(self, tracks, track_id):
        """
        Cleans the tracks by removing the selected cells from the tracks
        Updates the track ids of the cells after the selected cells
        Removes tracks that only contain one cell

        Parameters
        ----------
        tracks : np.ndarray
            The tracks to remove the cells from
        track_id : int
            The track id of the selected cells

        Returns
        -------
        tracks_filtered : np.ndarray
            The tracks with the selected cells removed
        """
        new_track_id = np.amax(self.parent.tracks[:, 0]) + 1
        selected_track = tracks[np.where(tracks[:, 0] == track_id)]
        selected_track_bound_lower = selected_track[
            np.where(selected_track[:, 1] > self.track_cells[0][0])
        ]
        selected_track_to_delete = selected_track_bound_lower[
            np.where(selected_track_bound_lower[:, 1] < self.track_cells[-1][0])
        ]
        selected_track_to_reassign = selected_track[
            np.where(selected_track[:, 1] >= self.track_cells[-1][0])
        ]
        delete_indices = np.where(
            np.any(
                np.all(
                    tracks[:, np.newaxis] == selected_track_to_delete,
                    axis=2,
                ),
                axis=1,
            )
        )[0]
        tracks_filtered = np.delete(tracks, delete_indices, axis=0)
        reassign_indices = np.where(
            np.any(
                np.all(
                    tracks_filtered[:, np.newaxis] == selected_track_to_reassign,
                    axis=2,
                ),
                axis=1,
            )
        )[0]
        tracks_filtered[reassign_indices, 0] = new_track_id
        ids, counts = np.unique(tracks_filtered[:, 0], return_counts=True)
        unique_ids = ids[counts == 1]
        mask = np.isin(tracks_filtered[:, 0], unique_ids, invert=True)
        tracks_filtered = tracks_filtered[mask]
        df = pd.DataFrame(tracks_filtered, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        return df.values

    def _link(self):
        """
        Calls the link function to prepare or perform the link
        """
        if self.btn_insert_correspondence.text() == LINK_TEXT:
            self._prepare_for_link()
        else:
            self._perform_link()

    def _prepare_for_link(self):
        """
        Prepares the link by adding a callback to the viewer
        """
        self._add_tracking_callback()
        self.btn_insert_correspondence.setText(CONFIRM_TEXT)

    def _perform_link(self):
        """
        Performs the link by adding the selected cells to a new track
        """
        self._update_callbacks()

        label_layer = self._get_layer(self.parent.combobox_segmentation.currentText())
        if label_layer is None:
            return

        tracks_layer = self._get_tracks_layer()
        if tracks_layer is not None and not np.array_equal(
            tracks_layer.data, self.parent.tracks
        ):
            self.handle_hidden_tracks()
            return

        if not self._validate_track_cells():
            return

        matches = 0

        for entry in self.track_cells:
            if tracks_layer is None:
                break
            for track_line in tracks_layer.data:
                if np.all(entry == track_line[1:4]):
                    matches += 1
                    if matches == 1:
                        match_id = track_line[0]
                        
        if matches > 1:
            if matches != np.count_nonzero(tracks_layer.data[:,0] == match_id):
                notify(
                    "Cell is already tracked. Please unlink it first if you want to change anything."
                )
                return
            
            mask = tracks_layer.data[:,0] != match_id
            tracks = tracks_layer.data[mask]
            if len(tracks) == 0:
                tracks = np.insert(self.track_cells, 0, np.zeros((1, len(self.track_cells))), axis=1)
                self.track_cells = []
                self._update_parent_tracks(tracks, tracks_layer)
            tracks_layer.data = tracks

        track_id, tracks = self._get_track_id_and_tracks(tracks_layer)
        connected_ids = self._get_connected_ids(track_id, tracks)
        track_id, tracks = self._update_track_id_and_tracks(
            connected_ids, track_id, tracks
        )
        tracks = self._add_new_track_cells_to_tracks(track_id, tracks)
        self._update_parent_tracks(tracks, tracks_layer)

    def _get_tracks_layer(self):
        """
        Returns the tracks layer

        Returns
        -------
        tracks_layer : napari layer
            The tracks layer
        """
        tracks_name = self.parent.combobox_tracks.currentText()
        try:
            return grab_layer(self.viewer, tracks_name)
        except ValueError:
            return None

    def handle_hidden_tracks(self):
        """
        Handles hidden tracks by asking the user if they want to display them
        """
        ret = choice_dialog(
            "All tracks need to be visible, but some tracks are hidden. Please retry with all tracks displayed. Do you want to display them now?",
            [("Display all", QMessageBox.AcceptRole), QMessageBox.Cancel],
        )
        # ret = 0 -> Display all, ret = 4194304 -> Cancel
        if ret == 0:
            self.lineedit_filter.setText("")
            self._replace_tracks()

    def _validate_track_cells(self):
        """
        Validates the selected cells for tracking

        Returns
        -------
        valid : bool
            True if the selected cells are valid, False otherwise
        """
        if not self.track_cells:
            return False

        if len(self.track_cells) < 2:
            notify("Less than two cells can not be tracked")
            return False

        if len(np.asarray(self.track_cells)[:, 0]) != len(
            set(np.asarray(self.track_cells)[:, 0])
        ):
            most_common_value = stats.mode(np.array(self.track_cells)[:, 0])[0]
            notify(
                f"Looks like you selected multiple cells in slice {most_common_value}. Cells have to be tracked one by one."
            )
            return False

        for i in range(len(self.track_cells) - 1):
            if self.track_cells[i][0] + 1 != self.track_cells[i + 1][0]:
                notify(
                    f"Looks like you missed a cell in slice {self.track_cells[i][0] + 1}. Please try again."
                )
                return False

        return True

    def _get_track_id_and_tracks(self, tracks_layer):
        """
        Returns a free track id and tracks

        Parameters
        ----------
        tracks_layer : napari layer
            The tracks layer

        Returns
        -------
        track_id : int
            The track id of the selected cells
        tracks : np.ndarray
            The tracks
        """
        if tracks_layer is None:
            return 1, None
        else:
            tracks = tracks_layer.data
            track_id = np.amax(tracks[:, 0]) + 1
            return track_id, tracks

    def _get_connected_ids(self, track_id, tracks):
        """
        Returns the ids of the tracks connected to the selected cells

        Parameters
        ----------
        track_id : int
            The track id of the selected cells
        tracks : np.ndarray
            The tracks

        Returns
        -------
        connected_ids : list
            The ids of the tracks connected to the selected cells
        """
        connected_ids = [-1, -1]
        if track_id != 1 and tracks is not None:
            for i in range(len(tracks)):
                if np.array_equal(tracks[i, 1:4], self.track_cells[0]):
                    connected_ids[0] = self.parent.tracks[i, 0]
                if np.array_equal(tracks[i, 1:4], self.track_cells[-1]):
                    connected_ids[1] = self.parent.tracks[i, 0]
        return connected_ids

    def _update_track_id_and_tracks(self, connected_ids, track_id, tracks):
        """
        Updates the track id and tracks

        Parameters
        ----------
        connected_ids : list
            The ids of the tracks connected to the selected cells
        track_id : int
            The track id of the selected cells
        tracks : np.ndarray
            The tracks

        Returns
        -------
        track_id : int
            The track id of the selected cells
        tracks : np.ndarray
            The tracks
        """
        if max(connected_ids) > -1:
            if connected_ids[0] > -1:
                self.track_cells.remove(self.track_cells[0])
            if connected_ids[1] > -1:
                self.track_cells.remove(self.track_cells[-1])

            if min(connected_ids) == -1:
                track_id = max(connected_ids)
            else:
                track_id = min(connected_ids)
                tracks[np.where(tracks[:, 0] == max(connected_ids)), 0] = track_id
        return track_id, tracks

    def _add_new_track_cells_to_tracks(self, track_id, tracks):
        """
        Adds the selected cells to a new track

        Parameters
        ----------
        track_id : int
            The track id of the selected cells
        tracks : np.ndarray
            The tracks

        Returns
        -------
        tracks : np.ndarray
            The tracks
        """
        for line in self.track_cells:
            new_track = [[track_id] + line]
            if tracks is None:
                tracks = new_track
            else:
                tracks = np.r_[tracks, new_track]
        return tracks

    def _update_parent_tracks(self, tracks, tracks_layer):
        """
        Updates the parent tracks and the tracks layer

        Parameters
        ----------
        tracks : np.ndarray
            The tracks
        tracks_layer : napari layer
            The tracks layer
        """
        df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        self.parent.tracks = df.values
        if tracks_layer is not None:
            tracks_layer.data = self.parent.tracks
        else:
            layer = self.viewer.add_tracks(self.parent.tracks, name="Tracks")
            self.parent.combobox_tracks.setCurrentText(layer.name)

    def _add_tracking_callback(self):
        """
        Adds a tracking callback to the viewer
        """
        self.track_cells = []

        def _select_cells(layer, event):
            try:
                label_layer = grab_layer(
                    self.viewer, self.parent.combobox_segmentation.currentText()
                )
            except ValueError as exc:
                handle_exception(exc)
                self._update_callbacks()
                return
            z = int(event.position[0])
            selected_id = label_layer.data[
                z, int(event.position[1]), int(event.position[2])
            ]
            if selected_id == 0:
                worker = notify_with_delay("The background can not be tracked")
                worker.start()
                return
            centroid = ndimage.center_of_mass(
                label_layer.data[z], labels=label_layer.data[z], index=selected_id
            )
            cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            if not cell in self.track_cells:
                self.track_cells.append(cell)
                self.track_cells.sort()

        self._update_callbacks(_select_cells)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def _add_auto_track_callback(self):
        """
        Adds an auto tracking callback to the viewer
        """

        def _proximity_track(layer, event):
            """
            Performs the proximity tracking
            """
            try:
                label_layer = grab_layer(
                    self.viewer, self.parent.combobox_segmentation.currentText()
                )
            except ValueError as exc:
                handle_exception(exc)
                return

            selected_cell = label_layer.data[
                int(round(event.position[0])),
                int(round(event.position[1])),
                int(round(event.position[2])),
            ]
            if selected_cell == 0:
                notify_with_delay("The background can not be tracked!")
                return
            worker = self._proximity_track_cell(
                label_layer, int(event.position[0]), selected_cell
            )
            worker.finished.connect(self._link)

        self._update_callbacks(_proximity_track)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    @thread_worker(connect={"errored": handle_exception})
    def _proximity_track_cell(self, label_layer, start_slice, id):
        """
        Performs the proximity tracking for a single cell

        Parameters
        ----------
        label_layer : napari layer
            The segmentation layer
        start_slice : int
            The slice to start the tracking
        id : int
            The id of the cell to track

        Returns
        -------
        track_cells : list
            The tracked cells
        """
        self._update_callbacks()
        self.track_cells = func(label_layer.data, start_slice, id)
        self.btn_insert_correspondence.setText("Tracking..")

    def _proximity_track_all(self):
        """
        Calls the overlap based tracking function for all cells
        """
        worker = self._proximity_track_all_worker()
        worker.returned.connect(self._restore_tracks)

    def _restore_tracks(self, tracks):
        """
        Restores the tracks layer with the new tracks

        Parameters
        ----------
        tracks : np.ndarray
            The tracks
        """
        if tracks is None:
            return
        self.parent.tracks = tracks
        layername = self.parent.combobox_tracks.currentText()
        if layername == "":
            layername = "Tracks"
        layer = self.viewer.add_tracks(tracks, name=layername)
        self.parent.combobox_tracks.setCurrentText(layer.name)
        QApplication.restoreOverrideCursor()

    @thread_worker(connect={"errored": handle_exception})
    def _proximity_track_all_worker(self):
        """
        Performs the overlap based tracking for all cells

        Returns
        -------
        tracks : np.ndarray
            The tracks
        """
        self._update_callbacks()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.parent.tracks = np.empty((1, 4), dtype=np.int8)
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )

        AMOUNT_OF_PROCESSES = self.parent.get_process_limit()

        data = []
        for start_slice in range(len(label_layer.data) - self.MIN_TRACK_LENGTH):
            for id in np.unique(label_layer.data[start_slice]):
                if id == 0:
                    continue
                data.append([label_layer.data, start_slice, id])

        with Pool(AMOUNT_OF_PROCESSES) as p:
            ret = p.starmap(func, data)

        track_id = 1
        for entry in ret:
            if entry is None:
                continue
            if "tracks" in locals() and entry[0] in tracks[:, 1:4].tolist():
                continue
            for line in entry:
                try:
                    tracks = np.r_[tracks, [[track_id] + line]]
                except UnboundLocalError:
                    tracks = [[track_id] + line]
            track_id += 1

        if not "tracks" in locals():
            return None
        tracks = np.array(tracks)
        df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        self.parent.tracks = df.values
        self.parent.initial_layers[1] = df.values
        return df.values

    def _update_callbacks(self, callback=None):
        """
        Updates the callbacks of the viewer

        Parameters
        ----------
        callback : function
            The callback to add to the viewer
        """
        if not len(self.viewer.layers):
            return
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if callback:
            current_callback = (
                label_layer.mouse_drag_callbacks[0]
                if label_layer.mouse_drag_callbacks
                else None
            )
            if current_callback and current_callback.__qualname__ in ["draw", "pick"]:
                self.cached_callback = current_callback
            for layer in self.viewer.layers:
                layer.mouse_drag_callbacks = [callback]
        else:
            for layer in self.viewer.layers:
                if layer == label_layer and self.cached_callback:
                    layer.mouse_drag_callbacks = [self.cached_callback]
                else:
                    layer.mouse_drag_callbacks = []
            self.cached_callback = None
        self._reset_button_labels()
        QApplication.restoreOverrideCursor()

    def _reset_button_labels(self):
        """
        Resets the button labels
        """
        self.btn_insert_correspondence.setText(LINK_TEXT)
        self.btn_remove_correspondence.setText(UNLINK_TEXT)


def func(label_data, start_slice, id):
    """
    Performs the proximity tracking for a single cell

    Parameters
    ----------
    label_data : np.ndarray
        The label data
    start_slice : int
        The slice to start the tracking
    id : int
        The id of the cell to track

    Returns
    -------
    track_cells : list
        The tracked cells
    """
    MIN_OVERLAP = 0.7

    slice = start_slice

    track_cells = []
    cell = np.where(label_data[start_slice] == id)
    while slice + 1 < len(label_data):
        matching = label_data[slice + 1][cell]
        matches = np.unique(matching, return_counts=True)
        maximum = np.argmax(matches[1])
        if (
            matches[1][maximum] <= MIN_OVERLAP * np.sum(matches[1])
            or matches[0][maximum] == 0
        ):
            if len(track_cells) < TrackingWindow.MIN_TRACK_LENGTH:
                return
            return track_cells

        if slice == start_slice:
            centroid = ndimage.center_of_mass(
                label_data[slice], labels=label_data[slice], index=id
            )
            track_cells.append(
                [slice, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            )
        centroid = ndimage.center_of_mass(
            label_data[slice + 1],
            labels=label_data[slice + 1],
            index=matches[0][maximum],
        )

        track_cells.append(
            [slice + 1, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
        )

        id = matches[0][maximum]
        slice += 1
        cell = np.where(label_data[slice] == id)
    if len(track_cells) < TrackingWindow.MIN_TRACK_LENGTH:
        return
    return track_cells
