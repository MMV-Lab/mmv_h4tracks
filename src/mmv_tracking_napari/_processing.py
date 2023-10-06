import os
import numpy as np
import multiprocessing
from multiprocessing import Pool

import napari
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QGridLayout,
    QApplication,
    QMessageBox,
    QSizePolicy,
)
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker
from scipy import ndimage, spatial, optimize
import napari

from ._logger import notify, choice_dialog
from ._grabber import grab_layer


class ProcessingWindow(QWidget):
    """
    A (QWidget) window to run processing steps on the data. Contains segmentation and tracking.

    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance

    Methods
    -------
    run_segmentation()
        Run segmentation on the raw image data
    run_demo_segmentation()
        Run the segmentation on the first 5 layers only
    run_tracking()
        Run tracking on the segmented cells
    adjust_ids()
        Replaces track ID 0 & adjusts segmentation IDs to match track IDs
    """

    dock = None

    def __init__(self, parent):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        parent : QWidget
            The parent widget
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Data processing")
        self.parent = parent
        self.viewer = parent.viewer
        ProcessingWindow.dock = self
        self.setStyleSheet(napari.qt.get_stylesheet(theme = "dark"))

        ### QObjects
        # Labels
        label_segmentation = QLabel("Segmentation")
        label_tracking = QLabel("Tracking")

        # Buttons
        self.btn_segment = QPushButton("Run Instance Segmentation")
        self.btn_preview_segment = QPushButton("Preview Segmentation")
        self.btn_preview_segment.setToolTip("Segment the first 5 frames")
        btn_track = QPushButton("Run Tracking")
        btn_adjust_seg_ids = QPushButton("Harmonize segmentation colors")
        btn_adjust_seg_ids.setToolTip("WARNING: This will take a while")
        
        self.btn_segment.setEnabled(False)
        self.btn_preview_segment.setEnabled(False)

        self.btn_segment.clicked.connect(self._run_segmentation)
        self.btn_preview_segment.clicked.connect(self._run_demo_segmentation)
        btn_track.clicked.connect(self._run_tracking)
        btn_adjust_seg_ids.clicked.connect(self._adjust_ids)

        # Comboboxes
        self.combobox_segmentation = QComboBox()
        self.combobox_segmentation.addItem("select model")
        self.read_models()
        self.combobox_segmentation.currentTextChanged.connect(self.toggle_segmentation_buttons)
        self.combobox_seg_layers = QComboBox()
        self.combobox_track_layers = QComboBox()
        self.layer_comboboxes = [self.combobox_seg_layers, self.combobox_track_layers]
        for layer in self.viewer.layers:
            for comobox in self.layer_comboboxes:
                combobox.addItem(layer.name)
        
        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())
        
        content.layout().addWidget(label_segmentation)
        content.layout().addWidget(self.combobox_seg_layers)
        content.layout().addWidget(self.combobox_segmentation)
        content.layout().addWidget(self.btn_preview_segment)
        content.layout().addWidget(self.btn_segment)
        content.layout().addWidget(line)
        content.layout().addWidget(label_tracking)
        content.layout().addWidget(self.combobox_track_layers)
        content.layout().addWidget(btn_track)
        content.layout().addWidget(btn_adjust_seg_ids)

        self.layout().addWidget(content)
        
        self.viewer.layers.events.inserted.connect(self.add_entry_to_comboboxes)
        self.viewer.layers.events.removed.connect(self.remove_entry_from_comboboxes)
        for layer in self.viewer.layers:
            layer.events.name.connect(self.rename_entry_in_comboboxes) # doesn't contain index
        self.viewer.layers.events.moving.connect(self.reorder_entry_in_comboboxes)
        
    def toggle_segmentation_buttons(self, text):
        if text == "select model":
            self.btn_segment.setEnabled(False)
            self.btn_preview_segment.setEnabled(False)
        else:
            self.btn_segment.setEnabled(True)
            self.btn_preview_segment.setEnabled(True)
        
    def read_models(self):
        path = f"{os.path.dirname(__file__)}/models"
        for file in os.listdir(path):
            self.combobox_segmentation.addItem(file)
        
    def add_entry_to_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            combobox.addItem(event.value.name)
        event.value.events.name.connect(self.rename_entry_in_comboboxes) # contains index
        
    def remove_entry_from_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            combobox.removeItem(event.index)

    def rename_entry_in_comboboxes(self, event):
        if not hasattr(event, 'index'):
            event.index = self.viewer.layers.index(event.source.name)
        for combobox in self.layer_comboboxes:
            index = combobox.currentIndex()
            combobox.removeItem(event.index)
            combobox.insertItem(event.index, event.source.name)
            combobox.setCurrentIndex(index)
        
    def reorder_entry_in_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            index = combobox.currentIndex()
            item = combobox.itemText(event.index)
            combobox.removeItem(event.index)
            combobox.insertItem(event.new_index, item)
            combobox.setCurrentIndex(index)

    def _add_segmentation_to_viewer(self, mask):
        """
        Adds the segmentation as a layer to the viewer with a specified name

        Parameters
        ----------
        mask : array
            the segmentation data to add to the viewer
        """
        try:
            self.viewer.add_labels(mask, name="calculated segmentation")
        except Exception as e:
            print("ran into some error: {}".format(e))
            return
        print("Added segmentation to viewer")

    @napari.Viewer.bind_key("Shift-s")
    def _hotkey_run_segmentation(self):
        ProcessingWindow.dock._run_segmentation()

    def _run_segmentation(self):
        """
        Calls segmentation without demo flag set
        """
        print("Calling full segmentation")

        worker = self._segment_image()
        worker.returned.connect(self._add_segmentation_to_viewer)
        worker.start()

    def _run_demo_segmentation(self):
        """
        Calls segmentation with the demo flag set
        """
        print("Calling demo segmentation")

        worker = self._segment_image(True)
        worker.returned.connect(self._add_segmentation_to_viewer)
        worker.start()

    @thread_worker
    def _segment_image(self, demo=False):
        """
        Run segmentation on the raw image data

        Parameters
        ----------
        demo : Boolean
            whether or not to do a demo of the segmentation
        Returns
        -------
        """
        print("Running segmentation")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        from cellpose import models

        try:
            data = grab_layer(self.viewer, self.combobox_seg_layers.currentText()).data
        except ValueError:
            print("Image layer not found in viewer")
            QApplication.restoreOverrideCursor()
            notify("No image layer found!")
            return

        if demo:
            data = data[0:5]

        selected_model = self.combobox_segmentation.currentText()

        try:
            parameters = self._get_parameters(selected_model)
        except UnboundLocalError:
            QApplication.restoreOverrideCursor()
            notify("Please select a different model")
            return

        # set process limit
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))

        global segment_slice

        def segment_slice(slice, parameters):
            """
            Calculate segmentation for a single slice

            Parameters
            ----------
            slice : napari
                the slice of raw image data to calculate segmentation for
            parameters : dict
                the parameters for the segmentation model

            Returns
            -------
            """
            model = models.CellposeModel(
                gpu=False, pretrained_model=parameters["model_path"]
            )
            mask = model.eval(
                slice,
                channels=[parameters["chan"], parameters["chan2"]],
                diameter=parameters["diameter"],
                flow_threshold=parameters["flow_threshold"],
                cellprob_threshold=parameters["cellprob_threshold"],
            )[0]
            return mask

        data_with_parameters = []
        for slice in data:
            data_with_parameters.append((slice, parameters))

        with Pool(AMOUNT_OF_PROCESSES) as p:
            mask = p.starmap(segment_slice, data_with_parameters)
            mask = np.asarray(mask)
            print("Done calculating segmentation")

        QApplication.restoreOverrideCursor()
        return mask

    def _get_parameters(self, model):
        """
        Get the parameters for the selected model

        Parameters
        ----------
        model : String
            The selected model

        Returns
        -------
        dict
            a dictionary of all the parameters based on selected model
        """
        print("Getting parameters")
        if model == "cellpose_neutrophils":
            print("Selected model 1")
            params = {
                "model_path": f"/models/{model}",
                "diameter": 15,
                "chan": 0,
                "chan2": 0,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0,
            }
        params["model_path"] = os.path.dirname(__file__) + params["model_path"]

        return params

    def _add_tracks_to_viewer(self, tracks):
        """
        Adds the tracks as a layer to the viewer with a specified name

        Parameters
        ----------
        tracks : array
            the tracks data to add to the viewer
        """
        # check if tracks are usable
        try:
            self.viewer.add_tracks(tracks, name="calculated tracks")
        except Exception as e:
            print("ran into some error: {}".format(e))
            return
        print("Added tracks to viewer")

    def _run_tracking(self):
        """
        Calls the tracking function
        """
        print("Calling tracking")

        worker = self._track_segmentation()
        worker.returned.connect(self._add_tracks_to_viewer)
        worker.start()

    @thread_worker
    def _track_segmentation(self):
        """
        Run tracking on the segmented data
        """
        print("Running tracking")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # get segmentation data
        try:
            #data = grab_layer(self.viewer, "Segmentation Data").data
            data = grab_layer(self.viewer, self.combobox_track_layers.currentText()).data
            #print(f"equal: {np.array_equal(data, other_data)}")
        except ValueError:
            print("Segmentation layer not found in viewer")
            QApplication.restoreOverrideCursor()
            notify("No segmentation layer found!")
            return

        # check for tracks layer
        if "Tracks" in self.viewer.layers:
            QApplication.restoreOverrideCursor()
            ret = choice_dialog(
                "Tracks layer found. Do you want to replace it?",
                [QMessageBox.Yes, QMessageBox.No],
            )
            QApplication.setOverrideCursor(Qt.WaitCursor)
            # 16384 is yes, 65536 is no
            if ret == 65536:
                QApplication.restoreOverrideCursor()
                return

        # set process limit
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))

        global calculate_centroids

        # calculate centroids
        def calculate_centroids(slice):
            labels = np.unique(slice)[1:]
            centroids = ndimage.center_of_mass(slice, labels=slice, index=labels)

            return (centroids, labels)

        with Pool(AMOUNT_OF_PROCESSES) as p:
            extended_centroids = p.map(calculate_centroids, data)

        # calculate connections between centroids of adjacent slices

        slice_pairs = []
        for i in range(1, len(data)):
            slice_pairs.append((extended_centroids[i - 1], extended_centroids[i]))

        APPROX_INF = 65535
        MAX_MATCHING_DIST = 45

        global match_centroids

        def match_centroids(slice_pair):
            num_cells_parent = len(slice_pair[0][0])
            num_cells_child = len(slice_pair[1][0])

            # calculate distance between each pair of cells

            cost_mat = spatial.distance.cdist(slice_pair[0][0], slice_pair[1][0])

            # if the distance is too far, change to approx. Inf.
            cost_mat[cost_mat > MAX_MATCHING_DIST] = APPROX_INF

            # add edges from cells in previous frame to auxillary vertices
            # in order to accomendate segmentation errors and leaving cells
            cost_mat_aug = (
                MAX_MATCHING_DIST
                * 1.2
                * np.ones(
                    (num_cells_parent, num_cells_child + num_cells_parent), dtype=float
                )
            )
            cost_mat_aug[:num_cells_parent, :num_cells_child] = cost_mat[:, :]

            # solve the optimization problem

            if (
                sum(sum(1 * np.isnan(cost_mat))) > 0
            ):  # check if there is at least one np.nan in cost_mat
                print("TODO: Remove this (Justin)")
                return
            row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)

            matched_pairs = []

            parent_centroids = slice_pair[0][0]
            parent_ids = slice_pair[0][1]
            child_centroids = slice_pair[1][0]
            child_ids = slice_pair[1][1]

            for i in range(len(row_ind)):
                parent_centroid = np.around(parent_centroids[row_ind[i]])
                parent_id = parent_ids[row_ind[i]]
                try:
                    child_centroid = np.around(child_centroids[col_ind[i]])
                    child_id = child_ids[col_ind[i]]
                except:
                    continue

                matched_pairs.append(
                    ([parent_centroid, parent_id], [child_centroid, child_id])
                )

            return matched_pairs

        with Pool(AMOUNT_OF_PROCESSES) as p:
            matches = p.map(match_centroids, slice_pairs)

        tracks = np.array([])
        next_id = 0
        visited = []
        for i in range(len(matches)):
            visited.append([0] * len(matches[i]))

        for i in range(len(visited)):
            for j in range(len(visited[i])):
                if visited[i][j]:
                    continue
                entry = [
                    next_id,
                    i,
                    int(matches[i][j][0][0][0]),
                    int(matches[i][j][0][0][1]),
                ]
                try:
                    tracks = np.append(tracks, np.array([entry]), axis=0)
                except:
                    tracks = np.array([entry])
                entry = [
                    next_id,
                    i + 1,
                    int(matches[i][j][1][0][0]),
                    int(matches[i][j][1][0][1]),
                ]
                tracks = np.append(tracks, np.array([entry]), axis=0)
                visited[i][j] = 1
                label = matches[i][j][1][1]

                slice = i + 1
                while True:
                    if slice >= len(matches):
                        break
                    labels = []
                    for k in range(len(matches[slice])):
                        labels.append(matches[slice][k][0][1])
                        visited[slice][k] = 1

                    if not label in labels:
                        break
                    match_number = labels.index(label)
                    entry = [
                        next_id,
                        slice + 1,
                        matches[slice][match_number][1][0][0],
                        matches[slice][match_number][1][0][1],
                    ]
                    tracks = np.append(tracks, np.array([entry]), axis=0)
                    label = matches[slice][match_number][1][1]

                    slice += 1

                next_id += 1

        tracks = tracks.astype(int)
        np.save("tracks.npy", tracks)

        QApplication.restoreOverrideCursor()
        return tracks

    def _adjust_ids(self):
        """
        Replaces track ID 0. Also adjusts segmentation IDs to match track IDs
        """
        raise NotImplementedError
        print("Adjusting segmentation IDs")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        import sys

        np.set_printoptions(threshold=sys.maxsize)
        print(self.viewer.layers[self.viewer.layers.index("Tracks")].data)
        QApplication.restoreOverrideCursor()
