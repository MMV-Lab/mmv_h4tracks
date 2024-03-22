import csv
import locale

import numpy as np
import zarr
import pandas as pd
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication

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


def save_zarr( zarr_file, layers, cached_tracks):
    """
    Saves the (changed) layers to a zarr file. Fails if required layers are missing

    Parameters
    ----------
    zarr_file : zarr
        The zarr file to which to save the layers
    layers : list of layer
        The layers raw image, segmentation, tracks in order
    cached_tracks : array
        The cached tracks layer, can be None
    """

    response = 0
    if not cached_tracks is None:
    # if not np.array_equal(layers[2].data, cached_tracks):
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

    tracks = layers[2].data
    if response == 1:
        tracks = cached_tracks
    # if response == 0:
    #     tracks = layers[2]

    if not "raw_data" in zarr_file:
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
    QApplication.restoreOverrideCursor()
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
    default_locale = locale.getdefaultlocale()[0]
    if default_locale.startswith("de"):
        data = [convert_np64_to_string(sublist) for sublist in data]
        delimiter = ";"
    else:
        delimiter = ","
    csvfile = open(file[0], "w", newline="")
    writer = csv.writer(csvfile, delimiter = delimiter)
    [writer.writerow(row) for row in data]
    csvfile.close()
    print("CSV file has been saved.")

def convert_np64_to_string(sublist):
    converted_sublist = []
    for item in sublist:
        if isinstance(item, np.float64):
            converted_sublist.append(str(item).replace(".", ","))
        else:
            converted_sublist.append(item)
    return converted_sublist