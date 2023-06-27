
import zarr

import numpy as np
from qtpy.QtWidgets import QMessageBox

from ._logger import choice_dialog, notify

def save_dialog(parent, filetype = "*.zarr", directory = ""):
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
    print("Prompting user to select save location")
    dialog = QFileDialog()
    dialog.setNameFilter(filetype)
    filetype_name = filetype[2:].capitalize()
    print("Showing dialog")
    filepath = dialog.getSaveFileName(parent,
                                      "Select location for {}-File to be created".format(filetype_name),
                                      directory)
    print("Dialog has been closed")
    print("Selected {} as path".format(filepath))
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
        The layers currently open in the viewer
    cached_tracks : array
        The complete tracks layer
    """

    print("Saving to file")
    try:
        index_of_raw = layers.index("Raw Image")
    except:
        print("No layer named 'Raw Image' found")
        raise ValueError("Raw Image layer missing!")
    try:
        index_of_segmentation = layers.index("Segmentation Data")
    except:
        print("No layer named 'Segmentation Data' found")
        raise ValueError("Segmentation layer missing!")
    try:
        index_of_tracks = layers.index("Tracks")
    except:
        print("No layer named 'Tracks' found")
        raise ValueError("Tracks layer missing!")
    
    response = 1
    if not np.array_equal(layers[index_of_tracks].data, cached_tracks):
        print("Difference between displayed and full tracks detected")
        response = choice_dialog(
            ("It looks like you have selected only some of the tracks from your tracks layer. " +
             "Do you want to save only the selected ones or all of them?"),
            [("Save Selected", QMessageBox.YesRole), # returns 0
             ("Save All", QMessageBox.NoRole), # returns 1
              QMessageBox.Cancel] # returns 4194304
            )
        if response == 4194304:
            print("Saving cancelled")
            return
    
    tracks = cached_tracks
    if response == 0:
        print("Saving currently displayed tracks")
        tracks = layers[index_of_tracks]
    else:
        print("Saving complete tracks")

    if zarr_file == None:
        print("No zarr file passed, prompting for new file location")
        file = save_dialog(parent, "*.zarr")
        zarr_file = zarr.open(file, mode = 'w')
        zarr_file.create_dataset(
            'raw_data',
            shape = layers[index_of_raw].data.shape,
            dtype = 'f8',
            data = layers[index_of_raw].data
            )
        zarr_file.create_dataset(
            'segmentation_data',
            shape = layers[index_of_segmentation].data.shape,
            dtype = 'i4',
            data = layers[index_of_segmentation].data
            )
        zarr_file.create_dataset(
            'tracking_data',
            shape = tracks.shape,
            dtype = 'i4',
            data = tracks
            )
    else:
        print("Saving to loaded zarr file")
        zarr_file['raw_data'][:] = layers[index_of_raw].data
        zarr_file['segmentation_data'][:] = layers[index_of_segmentation].data
        zarr_file['tracking_data'].resize(tracks.shape[0], tracks.shape[1])
        zarr_file['tracking_data'][:] = tracks
    print("Saving to zarr successful")
    notify("Zarr file has been saved.")
        
    
        
    
    
    