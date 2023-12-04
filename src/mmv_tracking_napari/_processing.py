import multiprocessing
import os
import platform
from multiprocessing import Pool, Manager

import napari
import numpy as np
from cellpose import models, core
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage, optimize, spatial
import torch

from mmv_tracking_napari import APPROX_INF, MAX_MATCHING_DIST
from ._grabber import grab_layer
from ._logger import choice_dialog, notify

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

    dock = None  # ?? ich vermute, kluge Menschen wissen, was das hier macht. Braucht keinen Kommentar, aber interessieren würde es mich trotzdem
                 # !! Attribut für selbstverweis. Wird für hotkeys benutzt
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
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Data processing")
        self.parent = parent
        self.viewer = parent.viewer
        ProcessingWindow.dock = self
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            pass

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
        self.combobox_segmentation.currentTextChanged.connect(
            self.toggle_segmentation_buttons
        )

        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())

        content.layout().addWidget(label_segmentation)
        content.layout().addWidget(self.combobox_segmentation)
        content.layout().addWidget(self.btn_preview_segment)
        content.layout().addWidget(self.btn_segment)
        content.layout().addWidget(line)
        content.layout().addWidget(label_tracking)
        content.layout().addWidget(btn_track)
        content.layout().addWidget(btn_adjust_seg_ids)

        self.layout().addWidget(content)

    def toggle_segmentation_buttons(self, text):
        """
        Toggles the segmentation buttons if a valid model is selected.

        Args:
            text (str): Selected model from the combobox.

        Returns:
            None
        """
        if text == "select model":
            self.btn_segment.setEnabled(False)
            self.btn_preview_segment.setEnabled(False)
        else:
            self.btn_segment.setEnabled(True)
            self.btn_preview_segment.setEnabled(True)

    def read_models(self):
        """
        Reads the available models from the 'models' directory and adds them to the segmentation combobox.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        for file in os.listdir(path):
            self.combobox_segmentation.addItem(file)

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
            print(f"ran into some error: {e}")
            return
        self.parent.combobox_segmentation.setCurrentText("calculated segmentation")
        print("Added segmentation to viewer")

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

        try:
            data = grab_layer(
                self.viewer, self.parent.combobox_image.currentText()
            ).data
        except AttributeError:
            print("Image layer not found in viewer")
            QApplication.restoreOverrideCursor()
            notify("No image layer found!")
            return

        if data is None:
            print("Image layer not selected")
            QApplication.restoreOverrideCursor()
            notify("Please select the image layer!")
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

        if core.use_gpu():
            print("can gpu")
            model = models.CellposeModel(gpu=True, pretrained_model=parameters["model_path"])
            mask = []
            for layer_slice in data:
                layer_mask, _, _ = model.eval(
                    layer_slice,
                    channels = [parameters["chan"], parameters["chan2"]],
                    diameter = parameters["diameter"],
                    flow_threshold = parameters["flow_threshold"],
                    cellprob_threshold = parameters["cellprob_threshold"],
                )
                mask.append(layer_mask)
            mask = np.array(mask)
        else:
            print("can't gpu")

            # set process limit
            if self.parent.rb_eco.isChecked():
                AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
            else:
                AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
            print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))


            data_with_parameters = [(layer_slice, parameters) for layer_slice in data]
            
            with Pool(AMOUNT_OF_PROCESSES) as p:
                mask = p.starmap(segment_slice_cpu, data_with_parameters)
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

    def _add_tracks_to_viewer(self, params):
        """
        Adds the tracks as a layer to the viewer with a specified name

        Parameters
        ----------
        tracks : array
            the tracks data to add to the viewer
        """
        # check if tracks are usable
        tracks, layername = params
        tracks_layer = grab_layer(
            self.viewer, self.parent.combobox_tracks.currentText()
        )
        if tracks_layer is None:
            self.viewer.add_tracks(tracks, name=layername)
        else:
            tracks_layer.data = tracks
        self.parent.combobox_tracks.setCurrentText(layername)
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
        print("Running tracking")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = self._get_segmentation_data()
        except ValueError:
            QApplication.restoreOverrideCursor()
            notify("No segmentation layer found!")
            return
        
        if data is None:
            QApplication.restoreOverrideCursor()
            notify("Please select the segmentation layer!")
            return

        tracks_name = self._check_for_tracks_layer()
        
        AMOUNT_OF_PROCESSES = self._calculate_processes_limit()
        print(f"Running on {AMOUNT_OF_PROCESSES} processes max")

        extended_centroids = self._calculate_centroids_parallel(data)
        matches = self._match_centroids_parallel(extended_centroids)

        tracks = self._process_matches(matches)

        QApplication.restoreOverrideCursor()
        return tracks, tracks_name
        
    def _get_segmentation_data(self):
        try:
            return grab_layer(self.viewer, self.parent.combobox_segmentation.currentText()).data
        except ValueError as exc:
            raise ValueError("Segmentation layer not found in viewer") from exc
        
    def _check_for_tracks_layer(self):
        tracks_name = "Tracks"
        try:
            tracks_layer = grab_layer(self.viewer, self.parent.combobox_tracks.currentText())
        except ValueError:
            pass
        else:
            if tracks_layer is not None:
                QApplication.restoreOverrideCursor()
                ret = choice_dialog(
                    "Tracks layer found. Do you want to replace it?",
                    [QMessageBox.Yes, QMessageBox.No],
                )
                QApplication.setOverrideCursor(Qt.WaitCursor)
                if ret == 65536:
                    QApplication.restoreOverrideCursor()
                    return tracks_name
                tracks_name = tracks_layer.name
        return tracks_name

    def _calculate_processes_limit(self):
        if self.parent.rb_eco.isChecked():
            return max(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            return max(1, int(multiprocessing.cpu_count() * 0.8))

    def _calculate_centroids_parallel(self, data):
        with Pool(self._calculate_processes_limit()) as p:
            return p.map(calculate_centroids, data)

    def _match_centroids_parallel(self, extended_centroids):
        slice_pairs = [(extended_centroids[i-1], extended_centroids[i]) for i in range(1, len(extended_centroids))]
        with Pool(self._calculate_processes_limit()) as p:
            return p.map(match_centroids, (slice_pairs))

    def _process_matches(self, matches):
        # Initialize variables to store tracks, unique ID and visited cells
        tracks = np.array([])
        next_id = 0
        visited = [[0] * len(matches[i]) for i in range(len(matches))]

        # Helper function to append entry to tracks
        def process_entry(entry, tracks):
            try:
                tracks = np.append(tracks, np.array([entry]), axis=0)
            except ValueError:
                tracks = np.array([entry])
            return tracks

        # Create an iterator to traverse through all slices and cells
        iterator = iter(((i, j) for i in range(len(visited)) for j in range(len(visited[i]))))

        # Iterate through slices and cells
        for slice_id, cell_id in iterator:
            if visited[slice_id][cell_id]:
                continue

            # Extract centroid information for parent and child cells
            entry = [next_id, slice_id, int(matches[slice_id][cell_id]["parent"]["centroid"][0]), int(matches[slice_id][cell_id]["parent"]["centroid"][1])]
            tracks = process_entry(entry, tracks)

            entry = [next_id, slice_id + 1, int(matches[slice_id][cell_id]["child"]["centroid"][0]), int(matches[slice_id][cell_id]["child"]["centroid"][1])]
            tracks = process_entry(entry, tracks)

            if next_id == 14 or next_id == 56:
                print(f"relevant id: {next_id}")

            visited[slice_id][cell_id] = 1
            label = matches[slice_id][cell_id]["child"]["id"]

            # Iterate through subsequent slices to complete the track
            while True:
                if slice_id + 1 >= len(matches):
                    break
                labels = [matches[slice_id + 1][matched_cell]["parent"]["id"] for matched_cell in range(len(matches[slice_id + 1]))]

                if not label in labels:
                    break

                match_number = labels.index(label)
                visited[slice_id + 1][match_number] = 1
                entry = [next_id, slice_id + 2, int(matches[slice_id + 1][match_number]["child"]["centroid"][0]),
                                                    int(matches[slice_id + 1][match_number]["child"]["centroid"][1])]
                tracks = process_entry(entry, tracks)
                label = matches[slice_id + 1][match_number]["child"]["id"]

                slice_id += 1
            
            next_id += 1

        return tracks.astype(int)

    def _adjust_ids(self):  # ?? ich weiß, ist noch zu tun, aber ich vermute stark, dass dann Kommentare hilfreich sein werden :D
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
        """
        Replaces Track ID 0 with new Track ID
        Changes Segmentation IDs to corresponding Track IDs
        """
        import multiprocessing

        if self.rb_eco.isChecked():
            self.AMOUNT_OF_THREADS = np.maximum(
                1, int(multiprocessing.cpu_count() * 0.4)
            )
        else:
            self.AMOUNT_OF_THREADS = np.maximum(
                1, int(multiprocessing.cpu_count() * 0.8)
            )
        try:
            label_layer = self.viewer.layers[
                self.viewer.layers.index("Segmentation Data")
            ]
        except ValueError:
            message("No label layer found!")
            return
        self.btn_adjust_seg_ids.setEnabled(False)
        self.rb_eco.setEnabled(False)
        self.rb_heavy.setEnabled(False)

        self.done = ThreadSafeCounter(0)
        self.new_id = 0
        self.completed = ThreadSafeCounter(0)
        next_slice = SliceCounter()

        self.threads = [QThread() for _ in range(self.AMOUNT_OF_THREADS)]

        for thread in self.threads:
            thread.finished.connect(thread.deleteLater)

        self.tracks_worker = AdjustTracksWorker(self, self.tracks)
        self.tracks_worker.moveToThread(self.threads[0])
        self.threads[0].started.connect(self.tracks_worker.run)
        self.tracks_worker.progress.connect(self._update_progress)
        self.tracks_worker.tracks_ready.connect(self._replace_tracks)
        for thread in self.threads[1 : self.AMOUNT_OF_THREADS]:
            self.tracks_worker.finished.connect(thread.start)
        self.tracks_worker.status.connect(self._set_description)
        self.tracks_worker.finished.connect(self.tracks_worker.deleteLater)

        self.seg_workers = [
            AdjustSegWorker(self, label_layer, self.tracks, next_slice, self.completed)
            for _ in range(self.AMOUNT_OF_THREADS)
        ]
        for i in range(0, self.AMOUNT_OF_THREADS):
            self.seg_workers[i].moveToThread(self.threads[i])

        self.tracks_worker.finished.connect(self.seg_workers[0].run)
        for i in range(1, self.AMOUNT_OF_THREADS):
            self.threads[i].started.connect(self.seg_workers[i].run)

        for seg_worker in self.seg_workers:
            seg_worker.update_progress.connect(self._multithread_progress)
            seg_worker.status.connect(self._multithread_description)
            seg_worker.finished.connect(seg_worker.deleteLater)

        for i in range(0, self.AMOUNT_OF_THREADS):
            self.seg_workers[i].finished.connect(self.threads[i].quit)

        for thread in self.threads:
            thread.finished.connect(self._enable_adjust_button)

        self.threads[0].start()
        print("(>'-')>")


def segment_slice_cpu(layer_slice, parameters):   # ?? Der parameter slice ist nen array oder?
                                        # Hier sollten wir nochmal schauen. Können wir mit cellpose.core.use_gpu() überprüfen, ob eine GPU verfügbar ist? (Auch wenn das nicht immer zu funktionieren schien)
                                        # Und falls ja, können wir vlt. für die GPU-Variante für model.eval über die einzelnen Slices loopen, damit wir nicht jedes mal models.Cellposemodel(...) neu aufrufen müssen?
                                        # Ich vermute, der verpflichtende models.CellposeModel Aufruf pro Slice lässt sich für die multi-processing CPU Variante aber nicht schön umgehen oder?
    """                                 # Yes, ndarray. Ja, wird in der aufrufenden funktion überprüft. gpu model wird nur einmal erstellt. cpu model lässt sich nicht so einfach übergeben, ohne es für jede ausführung neu zu erstellen
    Calculate segmentation for a single slice

    Parameters
    ----------
    layer_slice : 2darray
        the slice of raw image data to calculate segmentation for
    parameters : dict
        the parameters for the segmentation model

    Returns
    -------
    """
    model = models.CellposeModel(gpu=False, pretrained_model=parameters["model_path"])
    mask, _, _ = model.eval(
        layer_slice,
        channels=[parameters["chan"], parameters["chan2"]],
        diameter=parameters["diameter"],
        flow_threshold=parameters["flow_threshold"],
        cellprob_threshold=parameters["cellprob_threshold"],
    )
    return mask


# calculate centroids
def calculate_centroids(label_slice):
    """
    Calculate the centroids of objects in a 2D slice.

    Parameters
    ----------
    label_slice : numpy.ndarray
        A 2D numpy array representing the slice.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays: the centroids and the labels.
    """
    labels = np.unique(label_slice)[1:]
    centroids = ndimage.center_of_mass(label_slice, labels=label_slice, index=labels)

    return (centroids, labels)


def match_centroids(slice_pair):    # ?? copilot docs; Konstanten raus; Was mir generell noch auffällt: Wäre es "optisch ansprechender", wenn wir die Funktion hier vor 
                                    # _track_segmentation definieren würden, da match_centroids dort aufgerufen wird? Wäre das "best practice" oder ist sowas egal?
                                    # und slice_pair sollten wir sehr gut kommentieren, da es nicht offensichtlich ist, was das ist bzw. was da drin steckt
    """                             # !! ich definiere helfer generell eher nach den eigentlichen funktionen. wie genau willst du slice_pair kommentieren?
    Match centroids between two slices and return the matched pairs.

    Args:
        slice_pair (tuple): A tuple containing two slices, each represented by a tuple
                            containing the centroids and IDs of cells in that slice.

    Returns:
        list: A list of matched pairs, where each pair consists of the centroid and ID
              of a cell in the parent slice and the centroid and ID of the corresponding
              cell in the child slice.
    """

    parent_centroids = slice_pair[0][0]         # ?? ich hab das hier mal nach vorne gezogen und alle folgenden Aufrufe von slice_pair auf die Variablen hier umgestellt
    parent_ids = slice_pair[0][1]               # !! (y)
    child_centroids = slice_pair[1][0]
    child_ids = slice_pair[1][1]    

    num_cells_parent = len(parent_centroids)
    num_cells_child = len(child_centroids)



    # calculate distance between each pair of cells
    cost_mat = spatial.distance.cdist(parent_centroids, child_centroids)

    # if the distance is too far, change to approx. Inf.
    cost_mat[cost_mat > MAX_MATCHING_DIST] = APPROX_INF

    # add edges from cells in previous frame to auxillary vertices
    # in order to accomendate segmentation errors and leaving cells
    cost_mat_aug = (
        MAX_MATCHING_DIST
        * 1.2
        * np.ones((num_cells_parent, num_cells_child + num_cells_parent), dtype=float)
    )
    cost_mat_aug[:num_cells_parent, :num_cells_child] = cost_mat[:, :]

    # solve the optimization problem
    row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)

    matched_pairs = []



    for i in range(len(row_ind)):
        parent_centroid = np.around(parent_centroids[row_ind[i]])
        parent_id = parent_ids[row_ind[i]]
        try:
            child_centroid = np.around(child_centroids[col_ind[i]])
            child_id = child_ids[col_ind[i]]
        except:
            continue

        matched_pairs.append({"parent": {"centroid": parent_centroid, "id": parent_id}, "child": {"centroid": child_centroid, "id": child_id}})
        
    return matched_pairs
