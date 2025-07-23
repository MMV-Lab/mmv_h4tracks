import csv
import locale

import numpy as np
from qtpy.QtWidgets import QFileDialog
from napari.layers import Image, Labels
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels


def save_dialog(parent, filetype="*.ome.zarr", directory=""):
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

def save_ome_zarr(file, layers: list, implied_tracks: bool = True):
    """
    Save the image, segmentation and tracking data to an OME-zarr file

    path : str
        Path of the OME-zarr file to write to
    layers : list
        List of layers to save, in the order: raw image, segmentation
    implied_tracks : bool
        If True, the segmentation ids are equivalent to track ids
    """
    assert len(layers) == 2, "Only raw image and segmentation layers are supported"
    assert isinstance(layers[0], Image)
    assert isinstance(layers[1], Labels)

    if isinstance(file, str):
        # create the zarr file
        store = parse_url(file, mode="w").store
        root = zarr.group(store=store)
    else:
        # zarr file already exists, use it
        root = file

    # generate the raw image metadata
    frames = None
    unit = "unit"
    if "raw_metadata" in layers[0].metadata:
        raw_metadata = layers[0].metadata["raw_metadata"].split("\n")
        try:
            frames = [string.split("=")[1] for string in raw_metadata if string.startswith("frames")][0]
        except IndexError:
            pass
        try:
            unit = [string.split("=")[1] for string in raw_metadata if string.startswith("unit")][0]
        except IndexError:
            pass
    axes = "yx"
    if layers[0].ndim >= 4:
        z_size = layers[0].data.shape[-3]
        axes = "zyx"
    else:
        z_size = 1
    if layers[0].rgb:
        c_size = 3
        axes = "c" + axes
    else:
        c_size = 1
    axes = "t" + axes
    x_size = layers[0].data.shape[-2]
    y_size = layers[0].data.shape[-1]
    t_size = layers[0].data.shape[0]
    # dtype = layers[0].dtype
    scales = layers[0].scale
    # layer[0].multiscale might not matter explicitly
    chunk_shape = (1, layers[0].data.shape[1], layers[0].data.shape[2])

    # write the raw image data
    image_data = np.stack(layers[0].data)
    if "0" in root:
        # replace existing image data
        root["0"] = image_data
    else:
        # name omitted as it breaks the writer
        write_image(
            image = image_data,
            group = root,
            axes = axes,
            storage_options = dict(chunks=chunk_shape),
            scaler = None,
        )
    
    axes_metadata = []
    for axis in axes:
        if axis == "t":
            axes_metadata.append({"name": axis, "type": "time"})
        elif axis == "c":
            axes_metadata.append({"name": axis, "type": "space"})
        else:
            axes_metadata.append({"name": axis, "type": "space", "unit": unit})

    scale_metadata = [round(scale, 6) for scale in scales]

    # write the raw image metadata
    image_group = root["0"]
    image_group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": axes_metadata,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{
                "type": "scale",
                "scale": scale_metadata
            }]
        }]
    }]

    image_group.attrs["omero"] = {
        "channels": [{
            "active": True,
            "label": layers[0].name,
            "window": {
                "start": float(layers[0].data.min()),
                "end": float(layers[0].data.max()),
                "min": int(layers[0].data.min()),
                "max": int(layers[0].data.max())
            },
            "color": "7F7F7F" # figure out correct color for colormap gray in napari
        }],
        "rdefs": {
            "model": "color",
            "defaultZ": 0,
            "defaultT": 0
        }
    }

    # write the segmentation data
    write_labels(
        labels = layers[1].data,
        group = root, #label_group,
        name = "TrackedCells",
        axes = "tyx",
        storage_options = dict(chunks=chunk_shape),
        scaler = None,
    )

    assert "0" in root["labels"]["TrackedCells"], "Label data was not written correctly."
    # write the segmentation metadata
    label_group = root["labels"]["TrackedCells"]
    label_group.attrs["image-label"] = {
        "source": {"image": "0"},
        "version": "0.4",
        "label": layers[1].name,
    }

    label_group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": axes_metadata,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{
                "type": "scale",
                "scale": scale_metadata
            }]
        }]
    }]

    # assume passed labels are adjusted to tracking, so always true
    root["labels"]["TrackedCells"].attrs["implied_tracks"] = implied_tracks

    # generic metadata
    root.attrs["Image Dimensions"] = {
        "x": x_size,
        "y": y_size,
        "z": z_size,
        "t": t_size,
        "c": c_size
    }
    root.attrs["Frames"] = frames
    print("OME-zarr file has been saved.")

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
    writer = csv.writer(csvfile, delimiter=delimiter)
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
