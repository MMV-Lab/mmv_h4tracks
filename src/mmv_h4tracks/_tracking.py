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
    QSizePolicy,
    QWidget,
)
from scipy import ndimage, stats

from ._logger import notify, choice_dialog, handle_exception
from ._grabber import grab_layer
import mmv_h4tracks._processing as processing

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

        self.cached_callback = []
        self.cached_tracks = None
        self.selected_cells = []
        # initial layers saved in self.parent.initial_layers
        # segmentation -> 0, tracks -> 1

        ### QObjects

        # Labels
        label_display_ids = QLabel("Enter specific track IDs to display:")
        label_display_ids.setToolTip(
            "In order to display all tracks, clear the filter field and click 'Filter'"
        )
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
            "Click on a cell to track based on overlap \n\n" "Hotkey: S"
        )

        self.btn_remove_correspondence = QPushButton(UNLINK_TEXT)
        self.btn_remove_correspondence.setToolTip("Remove cells from their tracks")

        self.btn_insert_correspondence = QPushButton(LINK_TEXT)
        self.btn_insert_correspondence.setToolTip("Add cells to new track")

        btn_delete_displayed_tracks = QPushButton("Delete all displayed tracks")
        btn_filter_tracks = QPushButton("Filter")
        # btn_filter_tracks.setToolTip(
        #     "In order to display all tracks, clear the filter field and click here"
        # )
        btn_show_all_tracks = QPushButton("Show all tracks")
        btn_delete_selected_tracks = QPushButton("Delete")

        btn_update_centroids = QPushButton("Update centroids")
        btn_update_centroids.clicked.connect(self.update_all_centroids)

        btn_centroid_tracking.clicked.connect(self.coordinate_tracking_on_click)
        btn_auto_track_all.clicked.connect(self.overlap_tracking_on_click)
        btn_auto_track.clicked.connect(self.single_overlap_tracking_on_click)
        self.btn_remove_correspondence.clicked.connect(self.unlink_tracks_on_click)
        self.btn_insert_correspondence.clicked.connect(self.link_tracks_on_click)
        btn_delete_displayed_tracks.clicked.connect(
            self.delete_displayed_tracks_on_click
        )
        btn_filter_tracks.clicked.connect(self.filter_tracks_on_click)
        btn_show_all_tracks.clicked.connect(self.show_all_tracks_on_click)
        btn_delete_selected_tracks.clicked.connect(self.delete_listed_tracks_on_click)

        # Line Edits
        self.lineedit_filter = QLineEdit("")
        self.lineedit_delete = QLineEdit("")
        self.lineedit_filter.setPlaceholderText("e.g. 1, 2, 3")
        self.lineedit_filter.returnPressed.connect(self.filter_tracks_on_click)

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
        filter_tracks.layout().addWidget(btn_show_all_tracks, 3, 0, 1, -1)
        filter_tracks.layout().addWidget(h_spacer_4, 4, 2, 1, -1)
        filter_tracks.layout().addWidget(label_delete_specific_ids, 5, 0)
        filter_tracks.layout().addWidget(self.lineedit_delete, 5, 1)
        filter_tracks.layout().addWidget(btn_delete_selected_tracks, 5, 2)
        filter_tracks.layout().addWidget(btn_delete_displayed_tracks, 6, 0, 1, -1)

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())

        content.layout().addWidget(automatic_tracking)
        content.layout().addWidget(tracking_correction)
        content.layout().addWidget(filter_tracks)
        content.layout().addWidget(btn_update_centroids)
        content.layout().addWidget(v_spacer)

        self.layout().addWidget(content)

    def coordinate_tracking_on_click(self):
        """
        Runs the coordinate based tracking
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
        worker.returned.connect(self.process_new_tracks)
        worker.yielded.connect(on_yielded)

    def overlap_tracking_on_click(self):
        """
        Runs the overlap based tracking
        """
        worker = self.worker_overlap_tracking()
        worker.returned.connect(self.process_new_tracks)

    @thread_worker(connect={"errored": handle_exception})
    def worker_overlap_tracking(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.restore_callbacks()
        self.reset_button_labels()
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            raise ValueError("No segmentation layer to track")

        AMOUNT_OF_PROCESSES = self.parent.get_process_limit()

        track_id = 1
        for start_slice in range(len(label_layer.data) - self.MIN_TRACK_LENGTH):
            threads_input = []
            for label_id in np.unique(label_layer.data[start_slice]):
                if label_id == 0:
                    continue
                if track_id > 1:
                    # calculate centroid of the cell
                    centroid = ndimage.center_of_mass(
                        label_layer.data[start_slice],
                        labels=label_layer.data[start_slice],
                        index=label_id,
                    )
                    centroid = [
                        start_slice,
                        int(np.rint(centroid[0])),
                        int(np.rint(centroid[1])),
                    ]
                    # check if the cell is already tracked
                    tracked = False
                    for track in tracks:
                        if np.all(centroid == track[1:4]):
                            tracked = True
                    if tracked:
                        continue
                threads_input.append([label_layer.data, start_slice, label_id])

            with Pool(AMOUNT_OF_PROCESSES) as pool:
                track_cells = pool.starmap(func, threads_input)

            for entry in track_cells:
                if entry is None:
                    continue
                for line in entry:
                    try:
                        tracks = np.r_[tracks, [[track_id] + line]]
                    except UnboundLocalError:
                        tracks = np.array([[track_id] + line])
                track_id += 1

        if not "tracks" in locals():
            return None

        tracks = np.array(tracks)
        df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        QApplication.restoreOverrideCursor()
        return df.values

    def single_overlap_tracking_on_click(self):
        if self.cached_tracks is not None:
            notify("New tracks can only be added if all tracks are displayed.")
            return

        def overlap_tracking_callback(_, event):
            """
            Callback for the overlap based tracking
            """
            if len(event.position) == 2:
                raise ValueError("2D image can not be tracked.")

            try:
                label_layer = grab_layer(
                    self.viewer, self.parent.combobox_segmentation.currentText()
                )
            except ValueError as exc:
                handle_exception(exc)
                return

            selected_cell = label_layer.get_value(event.position)
            if selected_cell == 0:
                raise ValueError("The background can not be tracked.")

            worker = self.worker_single_overlap_tracking(
                label_layer.data, int(event.position[0]), selected_cell
            )
            worker.returned.connect(self.evaluate_proposed_track)

        self.set_callback(overlap_tracking_callback)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    @thread_worker(connect={"errored": handle_exception})
    def worker_single_overlap_tracking(
        self, segmentation: np.ndarray, slice_id: int, selected_cell: int
    ):
        """
        Perform single cell overlap based tracking

        Parameters
        ----------
        label_layer : napari layer
            The label layer
        slice : int
            The slice to track the cell from
        selected_cell : int
            The selected cell

        Returns
        -------
        track: np.ndarray
            The proposed track
        """
        self.restore_callbacks()
        start_slice = slice_id
        track = []
        MIN_OVERLAP = 0.7
        if segmentation.shape[0] - slice_id < self.MIN_TRACK_LENGTH:
            # Cell is too close to the end to have a track long enough
            return track
        cell_indices = np.where(segmentation[slice_id] == selected_cell)
        while slice_id + 1 < segmentation.shape[0]:
            matched_ids = segmentation[slice_id + 1][cell_indices]
            matched_ids_counted = np.unique(matched_ids, return_counts=True)
            index_highest_overlap = np.argmax(matched_ids_counted[1])
            if (
                matched_ids_counted[1][index_highest_overlap]
                <= MIN_OVERLAP * np.sum(matched_ids_counted[1])
                or matched_ids_counted[0][index_highest_overlap] == 0
            ):
                # No cell with big enough overlap found
                return track

            if slice_id == start_slice:
                centroid = ndimage.center_of_mass(
                    segmentation[slice_id],
                    labels=segmentation[slice_id],
                    index=selected_cell,
                )
                track.append(
                    [slice_id, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
                )
            centroid = ndimage.center_of_mass(
                segmentation[slice_id + 1],
                labels=segmentation[slice_id + 1],
                index=matched_ids_counted[0][index_highest_overlap],
            )
            track.append(
                [slice_id + 1, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            )

            selected_cell = matched_ids_counted[0][index_highest_overlap]
            slice_id += 1
            cell_indices = np.where(segmentation[slice_id] == selected_cell)

        return track

    def evaluate_proposed_track(self, proposed_track: list):
        """
        Evaluate the proposed track

        Parameters
        ----------
        proposed_track : list
            The proposed track
        """
        if len(proposed_track) < self.MIN_TRACK_LENGTH:
            QApplication.restoreOverrideCursor()
            raise ValueError("Could not find a track of sufficient length.")
        # Check if any of the cells are already tracked
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            self.viewer.add_tracks(
                np.insert(np.array(proposed_track), 0, 1, axis=1), name="Tracks"
            )
            QApplication.restoreOverrideCursor()
            return
        entries_to_add = []
        ids_to_change = []

        track_id: int = None
        for entry in proposed_track:
            # Check if the entry exists in the tracks layer
            existing_entry = [
                track for track in tracks_layer.data if np.all(track[1:4] == entry)
            ]
            if len(existing_entry) > 1 and track_id is None:
                track_id = existing_entry[0][0]
            diverging_entries = [
                track
                for track in tracks_layer.data
                if (track[0] in ids_to_change or track[0] == track_id)
                and track[1] == entry[0]
                and len(existing_entry) == 0
            ]

            if diverging_entries:
                # Diverging tracks
                break

            # If the entry does not exist it can be staged for addition
            if not existing_entry:
                entries_to_add.append(entry)
            else:
                if track_id is None:
                    track_id = existing_entry[0][0]
                    existing_track = np.array(
                        [track for track in tracks_layer.data if track[0] == track_id]
                    )
                    if np.min(existing_track[:, 1]) < existing_entry[0][1]:
                        # Converging tracks
                        track_id = None
                        break

                elif (
                    existing_entry[0][0] != track_id
                    and not existing_entry[0][0] in ids_to_change
                ):
                    existing_track = np.array(
                        [
                            track
                            for track in tracks_layer.data
                            if track[0] == existing_entry[0][0]
                        ]
                    )
                    if np.min(existing_track[:, 1]) < existing_entry[0][1]:
                        # Converging tracks
                        break
                    else:
                        ids_to_change.append(existing_entry[0][0])

        if track_id is None:
            track_id = np.amax(tracks_layer.data[:, 0]) + 1

        for old_id in ids_to_change:
            self.assign_new_track_id(tracks_layer, old_id, track_id)

        if entries_to_add:
            if len(entries_to_add) < self.MIN_TRACK_LENGTH and track_id == np.amax(
                tracks_layer.data[:, 0] + 1
            ):
                QApplication.restoreOverrideCursor()
                raise ValueError("Could not find a track of sufficient length.")
            if entries_to_add == proposed_track:
                self.add_track_to_tracks(np.array(proposed_track))
            else:
                self.add_entries_to_tracks(entries_to_add, track_id)

        QApplication.restoreOverrideCursor()

    def assign_new_track_id(self, tracks_layer, old_id: int, new_id: int):
        """
        Assign a new track id to the cells with the old id

        Parameters
        ----------
        tracks_layer : napari layer
            The tracks to update
        old_id : int
            The old id
        new_id : int
            The new id
        """
        tracks = tracks_layer.data
        tracks[tracks[:, 0] == old_id, 0] = new_id
        df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        tracks_layer.data = df.values

    def link_tracks_on_click(self):
        """
        Calls the link function to store selected cells or perform the link
        """

        def store_cell_for_link(_, event):
            """
            Callback for the unlink function to store the selected cells
            """
            if len(event.position) == 2:
                raise ValueError("2D image can not be tracked.")

            try:
                label_layer = grab_layer(
                    self.viewer, self.parent.combobox_segmentation.currentText()
                )
            except ValueError as exc:
                handle_exception(exc)
                self.reset_button_labels
                self.restore_callbacks()
                QApplication.restoreOverrideCursor()
                return
            z = int(event.position[0])
            selected_id = label_layer.get_value(event.position)
            if selected_id == 0:
                raise ValueError("The background can not be tracked.")
            centroid = ndimage.center_of_mass(
                label_layer.data[z],
                labels=label_layer.data[z],
                index=selected_id,
            )
            cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            if not cell in self.selected_cells:
                self.selected_cells.append(cell)
                self.selected_cells.sort()

        # check if button text is confirm or link
        if self.btn_insert_correspondence.text() == LINK_TEXT:
            if self.cached_tracks is not None:
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText(
                    "New tracks can only be added if all tracks are displayed! Display all now?"
                )
                msg.addButton("Display all", QMessageBox.AcceptRole)
                msg.addButton(QMessageBox.Cancel)
                retval = msg.exec()
                if retval != 0:
                    return
                self.parent.tracking_window.display_cached_tracks()
            self.reset_button_labels()
            self.selected_cells = []
            self.btn_insert_correspondence.setText(CONFIRM_TEXT)
            self.set_callback(store_cell_for_link)
            QApplication.setOverrideCursor(Qt.CrossCursor)
        else:
            self.reset_button_labels()
            self.restore_callbacks()
            QApplication.restoreOverrideCursor()
            if self.cached_tracks is not None:
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText(
                    "New tracks can only be added if all tracks are displayed! Display all now?"
                )
                msg.addButton("Display all", QMessageBox.AcceptRole)
                msg.addButton(QMessageBox.Cancel)
                retval = msg.exec()
                if retval != 0:
                    return
                self.parent.tracking_window.display_cached_tracks()
            self.link_stored_cells()

    def link_stored_cells(self):
        """
        Perform checks on the selected cells and add them to the tracks
        """
        # assure enough cells are selected
        if len(self.selected_cells) < 2:
            notify("Please select more than one cell to connect!")
            return
        # assure no two selected cells are from the same slice
        if len(np.asarray(self.selected_cells)[:, 0]) != len(
            set(np.asarray(self.selected_cells)[:, 0])
        ):
            most_common_value = stats.mode(np.array(self.selected_cells)[:, 0])[0]
            notify(
                f"Looks like you selected multiple cells in slice {most_common_value}. You can only connect cells from different slices."
            )
            return
        # assure there is no gap in z between the selected cells
        if (
            np.max(np.asarray(self.selected_cells)[:, 0])
            - np.min(np.asarray(self.selected_cells)[:, 0])
            != len(self.selected_cells) - 1
        ):
            missing_slices = [
                i
                for i in range(
                    np.min(np.asarray(self.selected_cells)[:, 0]),
                    np.max(np.asarray(self.selected_cells)[:, 0]),
                )
                if i not in np.asarray(self.selected_cells)[:, 0]
            ]
            notify(
                f"Gaps in the tracks are not supported yet. Please also select cells in frames {missing_slices}."
            )
            return

        tracks_layer = self.get_tracks_layer()

        track_id_matches = []
        for cell in self.selected_cells:
            if tracks_layer is None:
                break
            for track_line in tracks_layer.data:
                if np.all(track_line[1:4] == cell):
                    track_id_matches.append(track_line[0])

        track_id_matches = list(set(track_id_matches))

        contained_tracks = []
        cell_tuples = [tuple(cell) for cell in self.selected_cells]
        for track_id in track_id_matches:
            track = tracks_layer.data[tracks_layer.data[:, 0] == track_id]
            track_tuples = [tuple(track_line[1:4]) for track_line in track]
            # check if all of track's cells are in the selected cells
            if all(track_tuple in cell_tuples for track_tuple in track_tuples):
                contained_tracks.append(track_id)
                self.remove_entries_from_tracks([track_entry for track_entry in track])

        if len(track_id_matches) - len(contained_tracks) == 0:
            self.add_track_to_tracks(np.array(self.selected_cells))
            if tracks_layer is None:
                tracks_layer = self.get_tracks_layer()
        elif len(track_id_matches) - len(contained_tracks) == 1:
            touched_track_id = [
                track_id
                for track_id in track_id_matches
                if track_id not in contained_tracks
            ][0]
            track = tracks_layer.data[tracks_layer.data[:, 0] == touched_track_id]
            entries_to_add = []
            for cell in self.selected_cells:
                tracked = False
                for track_line in track:
                    if cell[0] == track_line[1]:
                        if not np.all(cell == track_line[1:4]):
                            notify(
                                f"You selected a cell in frame {cell[0]}, but track {touched_track_id} already contains a cell in this frame."
                            )
                            return
                        else:
                            tracked = True
                if not tracked:
                    # only add the cell if it is not already in the track
                    entries_to_add.append(cell)
            self.add_entries_to_tracks(entries_to_add, touched_track_id)
        else:
            # multiple tracks contain cells from the selected cells & are not contained
            # check if more than one track has same z
            z_values = [
                track_line[1]
                for track_line in tracks_layer.data
                if track_line[0] in track_id_matches
                and not track_line[0] in contained_tracks
            ]
            if len(z_values) != len(set(z_values)):
                # Not sure if there is a simple way to find the offending tracks
                notify(
                    "Connecting tracks that contain cells in the same slice is not possible."
                )
                return
            entries_not_to_add = []
            for touched_track_id in [
                track_id
                for track_id in track_id_matches
                if track_id not in contained_tracks
            ]:
                track = tracks_layer.data[tracks_layer.data[:, 0] == touched_track_id]
                for cell in self.selected_cells:
                    for track_line in track:
                        if cell[0] == track_line[1]:
                            if not np.all(cell == track_line[1:4]):
                                notify(
                                    f"You selected a cell in frame {cell[0]}, but track {touched_track_id} already contains a cell in this frame."
                                )
                                return
                            # cell should not be added since it would be duplicate
                            entries_not_to_add.append(cell)
            entries_to_add = [
                list(cell)
                for cell in set(tuple(entry) for entry in self.selected_cells)
                - set(tuple(entry) for entry in entries_not_to_add)
            ]
            if len(entries_to_add) > 0:
                self.add_entries_to_tracks(entries_to_add, track_id_matches[0])

        for track_id in track_id_matches[1:]:
            self.assign_new_track_id(tracks_layer, track_id, track_id_matches[0])

    def unlink_tracks_on_click(self):
        """
        Calls the unlink function to store selected cells or perform the unlink
        """

        def store_cell_for_unlink(_, event):
            """
            Callback for the unlink function to store the selected cells
            """
            if len(event.position) == 2:
                raise ValueError("2D image can not be tracked.")
            try:
                label_layer = grab_layer(
                    self.viewer, self.parent.combobox_segmentation.currentText()
                )
            except ValueError as exc:
                handle_exception(exc)
                self.reset_button_labels
                self.restore_callbacks()
                QApplication.restoreOverrideCursor()
                return
            z = int(event.position[0])
            selected_id = label_layer.get_value(event.position)
            if selected_id == 0:
                raise ValueError("The background can not be tracked.")
            centroid = ndimage.center_of_mass(
                label_layer.data[z],
                labels=label_layer.data[z],
                index=selected_id,
            )
            cell = [z, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            if not cell in self.selected_cells:
                self.selected_cells.append(cell)
                self.selected_cells.sort(key=lambda x: x[0])

        # check if button text is confirm or unlink
        if self.btn_remove_correspondence.text() == UNLINK_TEXT:
            self.reset_button_labels()
            self.selected_cells = []
            self.btn_remove_correspondence.setText(CONFIRM_TEXT)
            self.set_callback(store_cell_for_unlink)
            QApplication.setOverrideCursor(Qt.CrossCursor)
        else:
            self.reset_button_labels()
            self.restore_callbacks()
            QApplication.restoreOverrideCursor()
            self.unlink_stored_cells()

    def unlink_stored_cells(self):
        """
        Perform checks on the selected cells and remove them from the tracks
        """
        # assure enough cells are selected
        if len(self.selected_cells) < 2:
            notify("Please select at least two cells to disconnect!")
            return
        # check if all selected cells are on the same track
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            notify("Please select a valid tracks layer.")
            return

        track_id_matches = []
        for cell in self.selected_cells:
            if not np.any(np.all(tracks_layer.data[:, 1:4] == cell, axis=1)):
                # try updating the track entry
                # search for closest centroid in a spiral pattern
                # (pattern is not actually spiral, but it's a start)
                test_candidate = cell
                track_id = None
                for i in range(1, 10):
                    for j in range(-i, i + 1):
                        test_candidate = [cell[0], cell[1] + j, cell[2] + i]
                        test_candidate_2 = [cell[0], cell[1] + j, cell[2] - i]
                        track_ids = []
                        track_ids.extend(
                            [
                                track_line[0]
                                for track_line in tracks_layer.data
                                if np.all(track_line[1:4] == test_candidate)
                                or np.all(track_line[1:4] == test_candidate_2)
                            ]
                        )
                        if len(track_ids) > 0:
                            track_id = track_ids[0]
                            self.update_single_centroid(track_id, cell[0])
                            break
                    if track_id is not None:
                        break
                if track_id is not None:
                    track_id_matches.append(track_id)
                    continue

            for track_line in tracks_layer.data:
                if np.all(track_line[1:4] == cell):
                    track_id_matches.append(track_line[0])
                    break

        if len(set(track_id_matches)) != 1:
            notify("Please select cells from the same track to disconnect.")
            return

        if len(track_id_matches) != len(self.selected_cells):
            notify("All selected cells must be tracked.")
            return

        print(f"Selected cells initially: {self.selected_cells}")

        min_z = np.min(np.asarray(self.selected_cells)[:, 0])
        max_z = np.max(np.asarray(self.selected_cells)[:, 0])
        track = [
            track_line[1:4]
            for track_line in tracks_layer.data
            if track_line[0] == track_id_matches[0]
        ]
        min_z_track = np.min(np.asarray(track)[:, 0])
        max_z_track = np.max(np.asarray(track)[:, 0])
        if max_z - min_z != len(self.selected_cells) - 1:
            # fill in missing cells
            missing_cells = [
                list(cell)
                for cell in track
                if list(cell) not in self.selected_cells
                and cell[0] >= min_z
                and cell[0] <= max_z
            ]
            print(f"Missing cells: {missing_cells}")
            self.selected_cells.extend(missing_cells)

        self.selected_cells.sort(key=lambda x: x[0])

        print(f"Selected cells with missing cells added: {self.selected_cells}")
        # if only part of the track is removed the outermost entries must remain
        if min_z_track < min_z:
            self.selected_cells.pop(0)
        if max_z_track > max_z:
            self.selected_cells.pop(-1)
        print(f"Cells being removed: {self.selected_cells}")

        # remove the selected cells from the tracks
        self.remove_entries_from_tracks(self.selected_cells)
        if min_z_track < min_z and max_z_track > max_z:
            print("Splitting track")
            # split the track
            track_id = np.amax(tracks_layer.data[:, 0]) + 1
            if self.cached_tracks is not None:
                track_id = np.amax(self.cached_tracks[:, 0]) + 1
            print(f"New track id: {track_id}")
            track_to_reassign = [entry for entry in track if entry[0] >= max_z]
            self.remove_entries_from_tracks(track_to_reassign)
            self.add_entries_to_tracks(track_to_reassign, track_id)

    def filter_tracks_on_click(self):
        """
        Filters the tracks layer to only display the selected tracks
        """
        input_text = self.lineedit_filter.text()
        if input_text == "":
            if self.cached_tracks is not None:
                self.display_cached_tracks()
            return
        try:
            tracks_to_display = [
                int(track_id) for track_id in input_text.split(",") if track_id != ""
            ]
        except ValueError:
            notify("Please use a comma separated list of integers (whole numbers).")
            return
        tracks_to_display = list(set(tracks_to_display))
        tracks_to_display = [
            track_id
            for track_id in tracks_to_display
            if track_id in self.get_tracks_layer().data[:, 0]
            or self.cached_tracks is not None
            and track_id in self.cached_tracks[:, 0]
        ]
        tracks_to_display.sort()
        if len(tracks_to_display) < 1:
            self.lineedit_filter.clear()
            return
        self.lineedit_filter.setText(
            ", ".join([str(track_id) for track_id in tracks_to_display])
        )
        self.display_selected_tracks(tracks_to_display)

    def show_all_tracks_on_click(self):
        """
        Displays all tracks
        """
        self.lineedit_filter.setText("")
        if self.cached_tracks is not None:
            self.display_cached_tracks()

    def delete_listed_tracks_on_click(self):
        """
        Deletes the tracks specified in the lineedit_delete text field
        """
        input_text = self.lineedit_delete.text()
        if input_text == "":
            return
        try:
            tracks_to_delete = [
                int(track_id) for track_id in input_text.split(",") if track_id != ""
            ]
        except ValueError:
            notify("Please use a comma separated list of integers (whole numbers).")
            return

        tracks_layer = self.get_tracks_layer()

        # filter out the track ids that do not exist
        if self.cached_tracks is not None:
            non_existing_tracks = [
                track_id
                for track_id in tracks_to_delete
                if track_id not in self.cached_tracks[:, 0]
            ]
            tracks_to_delete = [
                track_id
                for track_id in tracks_to_delete
                if track_id in self.cached_tracks[:, 0]
            ]
        else:
            if tracks_layer is None:
                notify("Please select a valid tracks layer.")
                return
            non_existing_tracks = [
                track_id
                for track_id in tracks_to_delete
                if track_id not in tracks_layer.data[:, 0]
            ]
            tracks_to_delete = [
                track_id
                for track_id in tracks_to_delete
                if track_id in tracks_layer.data[:, 0]
            ]

        # check if only displayed tracks are selected for deletion
        if not np.all(np.isin(tracks_to_delete, tracks_layer.data[:, 0])):
            notify("Only displayed tracks can be deleted.")
            return

        # check if all displayed tracks are selected for deletion
        if np.all(np.isin(tracks_layer.data[:, 0], tracks_to_delete)):
            if self.cached_tracks is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Are you sure you want to delete all displayed tracks?")
                msg.setInformativeText(
                    "This action can not be undone. If you want to delete only some tracks, use the delete field."
                )
                msg.setWindowTitle("Delete all displayed tracks")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                ret = msg.exec_()
                if ret == QMessageBox.Yes:
                    self.viewer.layers.remove(tracks_layer.name)
                    self.viewer.layers.select_next()
                return
            # remove selected tracks from the cached tracks
            self.cached_tracks = np.delete(
                self.cached_tracks,
                np.isin(self.cached_tracks[:, 0], tracks_to_delete),
                0,
            )
            tracks_layer.data = self.cached_tracks
            self.cached_tracks = None
        else:
            tracks_layer.data = np.delete(
                tracks_layer.data, np.isin(tracks_layer.data[:, 0], tracks_to_delete), 0
            )
            if self.cached_tracks is not None:
                self.cached_tracks = np.delete(
                    self.cached_tracks,
                    np.isin(self.cached_tracks[:, 0], tracks_to_delete),
                    0,
                )
        if len(non_existing_tracks) > 0:
            message = f"Tracks {non_existing_tracks} do not exist and were not deleted."
            if len(tracks_to_delete) > 0:
                message += f" Tracks {tracks_to_delete} were removed successfully."
            notify(message)
        self.lineedit_delete.clear()

    def delete_displayed_tracks_on_click(self):
        """
        Deletes all displayed tracks
        """
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            notify("Please select a valid tracks layer.")
            return

        if self.cached_tracks is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Are you sure you want to delete all tracks?")
            msg.setInformativeText(
                "This action can not be undone. If you want to delete only some tracks, use the delete field."
            )
            msg.setWindowTitle("Delete all tracks")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            ret = msg.exec_()
            if ret == QMessageBox.No:
                return
            self.viewer.layers.remove(tracks_layer.name)
            self.viewer.layers.select_next()
        else:
            cached_tracks_view = self.cached_tracks.view(
                [("", self.cached_tracks.dtype)] * self.cached_tracks.shape[1]
            )
            tracks_layer_data_view = tracks_layer.data.view(
                [("", tracks_layer.data.dtype)] * tracks_layer.data.shape[1]
            )

            diff_view = np.setdiff1d(cached_tracks_view, tracks_layer_data_view)

            tracks_layer.data = diff_view.view(tracks_layer.data.dtype).reshape(
                -1, tracks_layer.data.shape[1]
            )
            self.cached_tracks = None
            self.lineedit_filter.clear()

    def process_new_tracks(self, tracks: np.ndarray):
        """
        Remove cached tracks, add new tracks to viewer and store as initial layer

        Parameters
        ----------
        tracks : np.ndarray
            The tracks to process
        """
        if tracks is None:
            QApplication.restoreOverrideCursor()
            return
        assert type(tracks) == np.ndarray, "Tracks are not numpy array."
        self.cached_tracks = None
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            self.viewer.add_tracks(tracks, name="Tracks")
        else:
            tracks_layer.data = tracks
        self.parent.initial_layers[1] = tracks
        QApplication.restoreOverrideCursor()

    def remove_entries_from_tracks(self, cells: list):
        """
        Remove cells from the tracks layer and the cached tracks

        Parameters
        ----------
        cells : list
            The cells to remove
        """
        print(f"Amount of cells to remove: {len(cells)}")
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            raise ValueError("Can't remove tracks from non-existing layer")
        displayed_tracks = tracks_layer.data
        tracks_objects = [displayed_tracks]
        track_results = []
        if self.cached_tracks is not None:
            tracks_objects.append(self.cached_tracks)
        for tracks in tracks_objects:
            old_length = len(tracks)
            mask = np.ones(tracks.shape[0], dtype=bool)

            for cell in cells:
                if len(cell) == 4:
                    cell = cell[1:4]
                mask &= ~np.all(tracks[:, 1:4] == cell, axis=1)
            tracks = tracks[mask]
            track_results.append(tracks)
            print(f"Removed {old_length - len(tracks)} cells")
        if len(track_results[0]) < 1:
            if len(track_results) > 1 and len(track_results[1]) > 1:
                tracks_layer.data = track_results[1]
            else:
                self.viewer.layers.remove(tracks_layer.name)
                self.viewer.layers.select_next()
            self.cached_tracks = None
        else:
            tracks_layer.data = track_results[0]
            if len(track_results) > 1:
                self.cached_tracks = track_results[1]

    def display_cached_tracks(self):
        """
        Display the cached tracks, remove them from cache
        """
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            self.viewer.add_tracks(self.cached_tracks, name="Tracks")
        else:
            tracks_layer.data = self.cached_tracks
        self.cached_tracks = None
        self.lineedit_delete.clear()

    def display_selected_tracks(self, track_ids: list):
        """
        Display the selected tracks
        Cache the displayed tracks if no cached tracks exist
        """
        if len(track_ids) < 1:
            notify("No tracks selected.")
            return
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            notify("Please select a valid tracks layer.")
            return
        if self.cached_tracks is None:
            self.cached_tracks = tracks_layer.data

        # filter the tracks
        selected_tracks = self.cached_tracks[
            np.isin(self.cached_tracks[:, 0], track_ids)
        ]
        if len(selected_tracks) < 1:
            notify("No tracks found for the selected track IDs.")
            return
        tracks_layer.data = selected_tracks
        self.lineedit_delete.clear()

    def add_entries_to_tracks(self, cells: list, track_id: int):
        """
        Add cells to the tracks layer and the cached tracks

        Parameters
        ----------
        cells : list
            The cells to add
        track_id : int
            The track id of the cells
        """
        print(f"Amount of cells to add: {len(cells)}")
        if len(cells) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("All selected cells are already tracked.")
            msg.setWindowTitle("No cells to add.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        # if self.cached_tracks is not None:
        #     raise ValueError("Can't add to tracks if there are cached tracks")
        # assume that a track is being reassigned if cache exists
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            tracks_layer = self.viewer.add_tracks(cells, name="Tracks")
            return
            # raise ValueError("Can't add to tracks if there are no tracks")
        tracks_objects = [tracks_layer.data]
        results_tracks = []
        if self.cached_tracks is not None:
            tracks_objects.append(self.cached_tracks)
        cells = [np.insert(cell, 0, track_id) for cell in cells]
        for tracks in tracks_objects:
            tracks = np.r_[tracks, cells]
            df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
            df.sort_values(["ID", "Z"], ascending=True, inplace=True)
            results_tracks.append(df.values)

        tracks_layer.data = results_tracks[0]
        if len(results_tracks) > 1:
            self.cached_tracks = results_tracks[1]

    def add_track_to_tracks(self, track: np.ndarray):
        """
        Add a track to the tracks layer

        Parameters
        ----------
        track : np.ndarray
            The track to add
        """
        if self.cached_tracks is not None:
            raise ValueError("Can't add to tracks if there are cached tracks")
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            track = np.insert(track, 0, 1, axis=1)
            self.viewer.add_tracks(track, name="Tracks")
            return
        tracks = tracks_layer.data
        track_id = np.amax(tracks[:, 0]) + 1
        track = np.insert(track, 0, track_id, axis=1)
        tracks = np.r_[tracks, track]
        tracks_layer.data = tracks

    def get_tracks_layer(self):
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

    def restore_callbacks(self):
        if len(self.viewer.layers) == 0:
            return
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            return
        for layer in self.viewer.layers:
            layer.mouse_drag_callbacks = []
        label_layer.mouse_drag_callbacks = self.cached_callback
        self.cached_callback = []

    def set_callback(self, callback):
        if len(self.viewer.layers) == 0:
            return
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            return
        self.cached_callback = label_layer.mouse_drag_callbacks
        for layer in self.viewer.layers:
            layer.mouse_drag_callbacks = [callback]

    def reset_button_labels(self):
        """
        Resets the button labels
        """
        self.btn_insert_correspondence.setText(LINK_TEXT)
        self.btn_remove_correspondence.setText(UNLINK_TEXT)

    def update_all_centroids(self):
        """
        Updates all centroids to account for changed segmentation
        """
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            return
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            return

        label_data = np.array(label_layer.data)
        tracks = np.array(tracks_layer.data)
        original_label_data = np.array(self.parent.initial_layers[0])

        frames_to_update = []

        for z in range(len(label_data)):
            frame_o = original_label_data[z]
            frame = label_data[z]
            if not np.array_equal(frame_o[frame_o > 0], frame[frame > 0]):
                frames_to_update.append(z)

        tracks_to_update = [track for track in tracks if track[1] in frames_to_update]
        unchanged_tracks = [
            track for track in tracks if track[1] not in frames_to_update
        ]
        updated_tracks = []
        for track in tracks_to_update:
            updated_track = update_centroid(label_data, tracks, track)
            updated_tracks.append(updated_track)

        updated_tracks = [track for track in updated_tracks if track is not None]
        updated_tracks.extend(unchanged_tracks)
        df = pd.DataFrame(updated_tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        updated_tracks = df.values
        updated_tracks = processing.split_noncontinuous_tracks(updated_tracks)
        updated_tracks = processing.remove_dot_tracks(updated_tracks)
        tracks_layer.data = updated_tracks

    def update_single_centroid(self, track_id: int, frame: int):
        """
        Updates a single centroid to account for changed segmentation
        """
        # starttime = time.time()
        label_layer = grab_layer(
            self.viewer, self.parent.combobox_segmentation.currentText()
        )
        if label_layer is None:
            return
        tracks_layer = self.get_tracks_layer()
        if tracks_layer is None:
            return
        tracks = tracks_layer.data
        track_entry = [
            entry for entry in tracks if entry[0] == track_id and entry[1] == frame
        ][0]

        filter_values = None
        if self.cached_tracks is not None:
            filter_values = np.unique(self.cached_tracks[:, 0])
            tracks = self.cached_tracks

        updated_entry = update_centroid(track_entry, label_layer.data[frame], tracks)
        if updated_entry is None:
            tracks = processing.remove_frame_from_track(tracks, track_entry)
        else:
            index = np.where(np.all(tracks == track_entry, axis=1))[0]
            tracks[index] = updated_entry

        df = pd.DataFrame(tracks, columns=["ID", "Z", "Y", "X"])
        df.sort_values(["ID", "Z"], ascending=True, inplace=True)
        tracks = df.values

        if filter_values is not None:
            self.cached_tracks = tracks
            self.display_selected_tracks(filter_values)
        else:
            tracks_layer.data = tracks

        # endtime = time.time()
        # print(f"Updating single centroid took {endtime - starttime} seconds.")


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

    slice_id = start_slice

    track_cells = []
    cell = np.where(label_data[start_slice] == id)
    while slice_id + 1 < len(label_data):
        matching = label_data[slice_id + 1][cell]
        matches = np.unique(matching, return_counts=True)
        maximum = np.argmax(matches[1])
        if (
            matches[1][maximum] <= MIN_OVERLAP * np.sum(matches[1])
            or matches[0][maximum] == 0
        ):
            if len(track_cells) < TrackingWindow.MIN_TRACK_LENGTH:
                return
            return track_cells

        if slice_id == start_slice:
            centroid = ndimage.center_of_mass(
                label_data[slice_id], labels=label_data[slice_id], index=id
            )
            track_cells.append(
                [slice_id, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
            )
        centroid = ndimage.center_of_mass(
            label_data[slice_id + 1],
            labels=label_data[slice_id + 1],
            index=matches[0][maximum],
        )

        track_cells.append(
            [slice_id + 1, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
        )

        id = matches[0][maximum]
        slice_id += 1
        cell = np.where(label_data[slice_id] == id)
    if len(track_cells) < TrackingWindow.MIN_TRACK_LENGTH:
        return
    return track_cells


def update_centroid(labels: np.ndarray, tracks: np.ndarray, track_entry: np.ndarray):
    """
    Updates the centroid of a track
    """
    frame_data = labels[track_entry[1]]
    # track entry: find if centroid is centroid of an existing cell
    # if not: find if centroid is close to an existing cell
    track_id, z, old_y, old_x = track_entry

    tolerance = 100
    offset = (max(0, old_y - tolerance), max(0, old_x - tolerance))
    roi = frame_data[
        offset[0] : min(frame_data.shape[0], old_y + tolerance + 1),
        offset[1] : min(frame_data.shape[1], old_x + tolerance + 1),
    ]

    def fast_center_of_mass(frame_data, label, offset):
        # about 20% faster than ndimage.center_of_mass in this context
        binary_mask = frame_data == label
        coords = np.argwhere(binary_mask)
        if coords.size == 0:
            return None
        coords += offset
        return np.mean(coords, axis=0)

    unique_labels = np.unique(roi)
    centroids = {
        label: fast_center_of_mass(roi, label, offset)
        for label in unique_labels
        if label != 0
    }
    closest_candidate = [None, float("inf")]
    # find closest candidate to old centroid
    for _, centroid in centroids.items():
        centroid = [int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
        # use candidate if centroids match
        if centroid[0] == old_y and centroid[1] == old_x:
            return np.array([track_id, z, old_y, old_x])
        distance = np.sqrt((old_y - centroid[0]) ** 2 + (old_x - centroid[1]) ** 2)
        # skip if candidate is already part of another track
        if np.any(
            np.all(tracks[:, 1:] == np.array([z, centroid[0], centroid[1]]), axis=1)
        ):
            continue
        if distance < closest_candidate[1]:
            closest_candidate = [centroid, distance]
    if closest_candidate[0] is not None:
        y, x = closest_candidate[0]
        return np.array([track_id, z, y, x])
    return None
