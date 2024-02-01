import csv

import numpy as np
import zarr
from qtpy.QtWidgets import QFileDialog, QMessageBox

from ._logger import choice_dialog, notify


def save_dialog(parent, filetype="*.zarr", directory=""):
    """
    Opens a dialog to select a location to save a file

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog
    filetype : str
        Only files of this file type will be displayed
    directory : str
        Opens view at the specified directory

    Returns
    -------
    str
        Path of selected file
    """
    dialog = QFileDialog()
    dialog.setNameFilter(filetype)
    filetype_name = filetype[2:].capitalize()
    filepath = dialog.getSaveFileName(
        parent,
        f"Select location for {filetype_name}-File to be created",
        directory,
    )
    return filepath


def save_zarr(parent, zarr_file, layers, cached_tracks):
    """
    Saves the (changed) layers to a zarr file. Fails if required layers are missing

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog
    zarr_file : zarr
        The zarr file to which to save the layers
    layers : list of layer
        The layers raw image, segmentation, tracks in order
    cached_tracks : array
        The complete tracks layer
    """

    response = 1
    if not np.array_equal(layers[2].data, cached_tracks):
        response = choice_dialog(
            (
                "It looks like you have selected only some of the tracks from your tracks layer. "
                + "Do you want to save only the selected ones or all of them?"
            ),
            [
                ("Save Selected", QMessageBox.YesRole),  # returns 0
                ("Save All", QMessageBox.NoRole),  # returns 1
                QMessageBox.Cancel,  # returns 4194304
            ],
        )
        if response == 4194304:
            return

    tracks = cached_tracks
    if response == 0:
        tracks = layers[2]

    if zarr_file == None:
        file = save_dialog(parent, "*.zarr")
        zarr_file = zarr.open(file, mode="w")
        zarr_file.create_dataset(
            "raw_data",
            shape=layers[0].data.shape,
            dtype="f8",
            data=layers[0].data,
        )
        zarr_file.create_dataset(
            "segmentation_data",
            shape=layers[1].data.shape,
            dtype="i4",
            data=layers[1].data,
        )
        zarr_file.create_dataset(
            "tracking_data", shape=tracks.shape, dtype="i4", data=tracks
        )
    else:
        zarr_file["raw_data"][:] = layers[0].data
        zarr_file["segmentation_data"][:] = layers[1].data
        zarr_file["tracking_data"].resize(tracks.shape[0], tracks.shape[1])
        zarr_file["tracking_data"][:] = tracks
    notify("Zarr file has been saved.")


def save_csv(file, data):
    """
    Save data to a csv file

    Parameters
    ----------
    file : str
        Path of the csv file to write to
    data : list
        CSV data to write to disk
    """
    csvfile = open(file[0], "w", newline="")
    writer = csv.writer(csvfile)
    [writer.writerow(row) for row in data]
    csvfile.close()
