import multiprocessing
import os
import platform
from multiprocessing import Pool, Manager
from threading import Event
import json
from pathlib import Path

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

from mmv_h4tracks import APPROX_INF, MAX_MATCHING_DIST
from ._grabber import grab_layer
from ._logger import choice_dialog, notify, handle_exception

CUSTOM_MODEL_PREFIX = "custom_"


def segment_slice_cpu(layer_slice, parameters):
    """
    Parameters
    ----------
    layer_slice : nd array
        the slice of raw image data to calculate segmentation for
    parameters : dict
        the parameters for the segmentation model

    Returns
    -------
    nd array
        the segmentation mask for the slice
    """
    model = models.CellposeModel(
        gpu=False, pretrained_model=parameters["model_path"]
    )
    eval_params = parameters
    eval_params.pop("model_path", None)
    mask, _, _ = model.eval(layer_slice, **eval_params)
    return mask


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


def match_centroids(
    slice_pair,
):
    """
    Match centroids between two slices.

    Parameters
    ----------
    slice_pair : tuple
        A tuple containing two slices, each represented by a tuple
        containing the centroids and IDs of cells in that slice.

    Returns
    -------
    list
        A list of matched pairs, where each pair consists of the centroid and ID
    """
    parent_centroids = slice_pair[0][0]
    parent_ids = slice_pair[0][1]
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

        matched_pairs.append(
            {
                "parent": {"centroid": parent_centroid, "id": parent_id},
                "child": {"centroid": child_centroid, "id": child_id},
            }
        )

    return matched_pairs

def read_custom_model_dict():
    """
    Reads the parameters of the custom models from the 'custom_models.json' file and returns them
    """
    try:
        with open(Path(__file__).parent / "custom_models.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def read_models(widget):
    """
    Reads the available models from the 'models' and 'custom models' directory and returns them
    """
    path = Path(__file__).parent / "models"

    hardcoded_models = [file.name for file in path.iterdir() if not file.is_dir()]
    custom_models = []

    p = Path(__file__).parent / "models" / "custom_models"
    custom_model_filenames = [file.name for file in p.glob("*") if file.is_file()]
    for custom_model in widget.custom_models:
        if widget.custom_models[custom_model]["filename"] in custom_model_filenames:
            custom_models.append(CUSTOM_MODEL_PREFIX + custom_model)
    return hardcoded_models, custom_models

def display_models(widget, hardcoded_models, custom_models):
    """
    Adds the passed models to the segmentation combobox.
    """
    hardcoded_models.sort()
    custom_models.sort()
    widget.combobox_segmentation.clear()
    widget.combobox_segmentation.addItems(hardcoded_models)
    widget.combobox_segmentation.addItems(custom_models)


def run_segmentation(widget):
    """
    Calls segmentation without demo flag set
    """
    worker = _segment_image(widget)
    worker.returned.connect(_add_segmentation_to_viewer)


def run_demo_segmentation(widget):
    """
    Calls segmentation with the demo flag set
    """
    worker = _segment_image(widget, True)
    worker.returned.connect(_add_segmentation_to_viewer)


def _add_segmentation_to_viewer(widget_and_mask):
    """
    Adds the segmentation as a layer to the viewer with a specified name

    Parameters
    ----------
    mask : array
        the segmentation data to add to the viewer
    """
    widget, mask = widget_and_mask
    labels = widget.viewer.add_labels(mask, name="calculated segmentation")
    widget.parent.combobox_segmentation.setCurrentText(labels.name)


@thread_worker(connect={"errored": handle_exception})
def _segment_image(widget, demo=False):
    """
    Run segmentation on the raw image data

    Parameters
    ----------
    demo : Boolean
        whether or not to do a demo of the segmentation
    Returns
    -------
    widget, mask
        the widget and the segmentation mask
    """
    QApplication.setOverrideCursor(Qt.WaitCursor)

    viewer = widget.viewer

    data = grab_layer(viewer, widget.parent.combobox_image.currentText()).data

    if demo:
        data = data[0:5]

    selected_model = widget.combobox_segmentation.currentText()
    parameters = _get_parameters(widget, selected_model)

    if core.use_gpu():
        model = models.CellposeModel(
            gpu=True, pretrained_model=parameters.pop("model_path")
        )
        mask = []
        for layer_slice in data:
            layer_mask, _, _ = model.eval(layer_slice, **parameters)
            mask.append(layer_mask)
        mask = np.array(mask)
    else:
        # set process limit
        AMOUNT_OF_PROCESSES = widget.parent.get_process_limit()

        data_with_parameters = [(layer_slice, parameters) for layer_slice in data]

        with Pool(AMOUNT_OF_PROCESSES) as p:
            mask = p.starmap(segment_slice_cpu, data_with_parameters)
            mask = np.asarray(mask)

    if not demo:
        widget.parent.initial_layers[0] = mask
    QApplication.restoreOverrideCursor()
    return widget, mask


def _get_parameters(widget, model: str):
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
    # Hardcoded models
    if model == "Neutrophil_granulocytes":
        params = {
            "model_path": str(Path(__file__).parent.absolute() / "models" / model),
            "diameter": 15,
            "channels": [0, 0],
            "flow_threshold": 0.4,
            "cellprob_threshold": 0,
        }

    model = model[len(CUSTOM_MODEL_PREFIX) :]
    # Custom models
    if model in widget.custom_models:
        params = widget.custom_models[model]["params"]
        params["model_path"] = str(
            Path(__file__).parent.absolute()
            / "models"
            / "custom_models"
            / widget.custom_models[model]["filename"]
        )

    return params


@thread_worker(connect={"errored": handle_exception})
def _track_segmentation(widget):
    """
    Run tracking on the segmentation data

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    widget, tracks, tracks_name
        the widget, the tracks and the name of the tracks layer
    """
    QApplication.setOverrideCursor(Qt.WaitCursor)
    data = _get_segmentation_data(widget)

    # check for tracks layer
    _, collision = _check_for_tracks_layer(widget)
    if collision:
        QApplication.restoreOverrideCursor()
        yield "Replace tracks layer"
        widget.choice_event.wait()
        widget.choice_event.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        ret = widget.ret
        del widget.ret
        if ret == 65536:
            QApplication.restoreOverrideCursor()
            return

    extended_centroids = _calculate_centroids_parallel(widget, data)
    matches = _match_centroids_parallel(widget, extended_centroids)

    tracks = _process_matches(matches)

    QApplication.restoreOverrideCursor()
    return tracks


def _get_segmentation_data(widget):
    """
    Get the segmentation data from the viewer

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    array
        the segmentation data
    """
    try:
        return grab_layer(
            widget.viewer, widget.parent.combobox_segmentation.currentText()
        ).data
    except ValueError as exc:
        raise ValueError("Segmentation layer not found in viewer") from exc


def _add_tracks_to_viewer(params):
    """
    Adds the tracks as a layer to the viewer with a specified name

    Parameters
    ----------
    tracks : array
        the tracks data to add to the viewer
    """
    # check if tracks are usable
    if params is None:
        return
    widget, tracks, layername = params
    try:
        tracks_layer = grab_layer(
            widget.viewer, widget.parent.combobox_tracks.currentText()
        )
    except ValueError as exc:
        if str(exc) == "Layer name can not be blank":
            widget.viewer.add_tracks(tracks, name=layername)
        else:
            handle_exception(exc)
            return
    else:
        tracks_layer.data = tracks
    widget.parent.tracks = tracks
    widget.parent.initial_layers[1] = tracks
    widget.parent.combobox_tracks.setCurrentText(layername)


def _check_for_tracks_layer(widget):
    """
    Check if there is a tracks layer in the viewer

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    tracks_name : String
        the name of the tracks layer
    collision : Boolean
        whether or not there is a tracks layer in the viewer
    """
    tracks_name = "Tracks"
    collision = True
    try:
        tracks_layer = grab_layer(
            widget.viewer, widget.parent.combobox_tracks.currentText()
        )
    except ValueError:
        collision = False
    else:
        tracks_name = tracks_layer.name
    return tracks_name, collision


def _calculate_processes_limit(widget):
    """
    Calculate the amount of processes to use for multiprocessing

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    int
        the amount of processes to use
    """
    if widget.parent.rb_eco.isChecked():
        return max(1, int(multiprocessing.cpu_count() * 0.4))
    else:
        return max(1, int(multiprocessing.cpu_count() * 0.8))


def _calculate_centroids_parallel(widget, data):
    """
    Calculate the centroids of objects in a 2D slice.
    """
    with Pool(_calculate_processes_limit(widget)) as p:
        return p.map(calculate_centroids, data)


def _match_centroids_parallel(widget, extended_centroids):
    """
    Match centroids between two slices.
    """
    slice_pairs = [
        (extended_centroids[i - 1], extended_centroids[i])
        for i in range(1, len(extended_centroids))
    ]
    with Pool(_calculate_processes_limit(widget)) as p:
        return p.map(match_centroids, (slice_pairs))


def _process_matches(matches):
    """
    Process the matches to create the tracks

    Parameters
    ----------
    matches : list
        the list of matches

    Returns
    -------
    array
        the tracks
    """
    # Initialize variables to store tracks, unique ID and visited cells
    tracks = np.array([])
    next_id = 0
    visited = [[0] * len(matches[i]) for i in range(len(matches))]

    # Helper function to append entry to tracks
    def process_entry(entry, tracks):
        """
        Process an entry to add to the tracks"""
        try:
            tracks = np.append(tracks, np.array([entry]), axis=0)
        except ValueError:
            tracks = np.array([entry])
        return tracks

    # Create an iterator to traverse through all slices and cells
    iterator = iter(
        ((i, j) for i in range(len(visited)) for j in range(len(visited[i])))
    )

    # Iterate through slices and cells
    for slice_id, cell_id in iterator:
        if visited[slice_id][cell_id]:
            continue

        # Extract centroid information for parent and child cells
        entry = [
            next_id,
            slice_id,
            int(matches[slice_id][cell_id]["parent"]["centroid"][0]),
            int(matches[slice_id][cell_id]["parent"]["centroid"][1]),
        ]
        tracks = process_entry(entry, tracks)

        entry = [
            next_id,
            slice_id + 1,
            int(matches[slice_id][cell_id]["child"]["centroid"][0]),
            int(matches[slice_id][cell_id]["child"]["centroid"][1]),
        ]
        tracks = process_entry(entry, tracks)

        visited[slice_id][cell_id] = 1
        label = matches[slice_id][cell_id]["child"]["id"]

        # Iterate through subsequent slices to complete the track
        while True:
            if slice_id + 1 >= len(matches):
                break
            labels = [
                matches[slice_id + 1][matched_cell]["parent"]["id"]
                for matched_cell in range(len(matches[slice_id + 1]))
            ]

            if not label in labels:
                break

            match_number = labels.index(label)
            visited[slice_id + 1][match_number] = 1
            entry = [
                next_id,
                slice_id + 2,
                int(matches[slice_id + 1][match_number]["child"]["centroid"][0]),
                int(matches[slice_id + 1][match_number]["child"]["centroid"][1]),
            ]
            tracks = process_entry(entry, tracks)
            label = matches[slice_id + 1][match_number]["child"]["id"]

            slice_id += 1

        next_id += 1

    return tracks.astype(int)
