import csv
import locale

import numpy as np
from qtpy.QtWidgets import QFileDialog, QApplication
from napari.layers import Image, Labels, Tracks
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from ._logger import notify


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

def save_zarr(file, layers: list):
    """
    Save the image, segmentation and tracking data to a zarr file

    Parameters
    ----------
    file : str
        Path of the zarr file to write to
    layers : list
        List of layers to save, in the order: raw image, segmentation
    """
    assert len(layers) == 3, "Only raw image segmentation and tracking layers are supported"
    assert isinstance(layers[0], Image)
    assert isinstance(layers[1], Labels)
    assert isinstance(layers[2], Tracks)

    if isinstance(file, str):
        # create the zarr file
        store = parse_url(file, mode="w").store
        root = zarr.group(store=store)
    else:
        # zarr file already exists, use it
        root = file

    if layers[0].multiscale:
        raw_image = layers[0].data[0]
    else:
        raw_image = layers[0].data

    if not "raw_data" in file:
        root.create_dataset(
            "raw_data",
            shape=raw_image.shape,
            dtype="f8",
            data=raw_image,
        )
        root.create_dataset(
            "segmentation_data",
            shape=layers[1].data.shape,
            dtype="i4",
            data=layers[1].data,
        )
        root.create_dataset(
            "tracking_data", shape=layers[2].data.shape, dtype="i4", data=layers[2].data
        )
    else:
        root["raw_data"][:] = raw_image
        root["segmentation_data"][:] = layers[1].data
        root["tracking_data"].resize((layers[2].data.shape[0], layers[2].data.shape[1]))
        root["tracking_data"][:] = layers[2].data
    QApplication.restoreOverrideCursor()
    if isinstance(file, str):
        notify(f"{file} has been saved.")
    else:
        notify("Zarr file has been saved.")

def save_ome_zarr(
    file,
    layers: list,
    implied_tracks: bool = True,
    is_multiscale: bool = False,
):
    """
    Save the image, segmentation and tracking data to an OME-zarr file

    file : str
        Path of the OME-zarr file to write to
    layers : list
        List of layers to save, in the order: raw image, segmentation
    implied_tracks : bool
        If True, the segmentation ids are equivalent to track ids
    is_multiscale : bool
        If True, the original data was multiscale and should be saved as multiscale.
        If False, save only the finest level (layers[0].data[0])
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
    
    # Check if the image layer is multiscale or single resolution
    # Multiscale layers have data as a list/tuple of arrays (one per resolution level)
    # Single resolution layers have data as a numpy array
    layer_data = layers[0].data
    # Check actual layer structure: multiscale if list/tuple with more than 1 element
    is_layer_multiscale = isinstance(layer_data, (list, tuple)) and len(layer_data) > 1
    
    # Determine how to save:
    # - If is_multiscale is True: original data was multiscale, save all levels
    # - If is_multiscale is False: save as single resolution (use first level if layer is multiscale for display)
    save_as_multiscale = is_multiscale
    
    # Get shape from layer data
    if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
        # Layer is multiscale (or list with one element) - get shape from first level
        original_shape = layer_data[0].shape
    else:
        # Layer is single resolution array
        original_shape = layer_data.shape
    
    # Get scales from layer (needed for metadata)
    scales = layers[0].scale
    
    # write the raw image data
    if save_as_multiscale:
        # Save all multiscale levels
        image_data_list = layer_data
        # Prepare first level - ensure it has time dimension
        first_level = image_data_list[0]
        if first_level.ndim == 3:
            first_level = np.stack([first_level])
        # Calculate axes based on final shape
        final_shape = first_level.shape
        axes = "yx"
        if len(final_shape) >= 4:
            z_size = final_shape[-3]
            axes = "zyx"
        else:
            z_size = 1
        if layers[0].rgb:
            c_size = 3
            axes = "c" + axes
        else:
            c_size = 1
        axes = "t" + axes
        x_size = final_shape[-2]
        y_size = final_shape[-1]
        t_size = final_shape[0]
        # chunk_shape should match spatial dimensions (excluding time)
        if len(final_shape) == 4:  # TZYX
            chunk_shape = (1, final_shape[1], final_shape[2], final_shape[3])
        elif len(final_shape) == 3:  # TYX
            chunk_shape = (1, final_shape[1], final_shape[2])
        else:
            chunk_shape = (1, final_shape[1], final_shape[2])
        
        # Remove existing multiscale data if present
        for key in list(root.keys()):
            if key.isdigit():
                del root[key]
        # Write first level with write_image to set up structure, then write others directly
        if "0" in root:
            root["0"] = first_level
        else:
            write_image(
                image=first_level,
                group=root,
                axes=axes,
                storage_options=dict(chunks=chunk_shape),
                scaler=None,
            )
        # Write remaining levels directly
        for i in range(1, len(image_data_list)):
            level_data = image_data_list[i]
            if level_data.ndim == 3:
                level_data = np.stack([level_data])
            root[str(i)] = level_data
    else:
        # Save only the finest level (single resolution)
        # Handle both cases: data as array or data as list with one element
        if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
            image_data = layer_data[0]
        else:
            image_data = layer_data
        # Ensure it has time dimension (stack only if 3D)
        # If already 4D, assume it already has time dimension
        if image_data.ndim == 3:
            image_data = np.stack([image_data])
        
        # Calculate axes based on final shape
        final_shape = image_data.shape
        ndim = len(final_shape)
        axes = "yx"
        if ndim >= 4:
            z_size = final_shape[-3]
            axes = "zyx"
        else:
            z_size = 1
        if layers[0].rgb:
            c_size = 3
            axes = "c" + axes
        else:
            c_size = 1
        axes = "t" + axes
        # Verify axes length matches dimensions
        if len(axes) != ndim:
            raise ValueError(
                f"Axes length ({len(axes)}) must match number of dimensions ({ndim}). "
                f"Shape: {final_shape}, Axes: {axes}"
            )
        x_size = final_shape[-2]
        y_size = final_shape[-1]
        t_size = final_shape[0]
        # chunk_shape should match spatial dimensions (excluding time)
        if len(final_shape) == 4:  # TZYX
            chunk_shape = (1, final_shape[1], final_shape[2], final_shape[3])
        elif len(final_shape) == 3:  # TYX
            chunk_shape = (1, final_shape[1], final_shape[2])
        else:
            chunk_shape = (1, final_shape[1], final_shape[2])
        
        if "0" in root:
            # replace existing image data
            root["0"] = image_data
        else:
            # name omitted as it breaks the writer
            write_image(
                image=image_data,
                group=root,
                axes=axes,
                storage_options=dict(chunks=chunk_shape),
                scaler=None,
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
    # For multiscale, use root group; for single, use "0" group
    if save_as_multiscale:
        image_group = root
        # Create datasets list for all multiscale levels
        datasets = []
        for i in range(len(image_data_list)):
            # Calculate scale for this level (each level is 2x downsampled from previous)
            # Level 0 (finest) has original scale, level 1 has 2x scale, level 2 has 4x scale, etc.
            level_scale = [s * (2 ** i) for s in scale_metadata]
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [round(s, 6) for s in level_scale]
                }]
            })
        # Get data min/max for omero metadata (across all levels)
        data_min = min(level.min() for level in image_data_list)
        data_max = max(level.max() for level in image_data_list)
    else:
        image_group = root["0"]
        datasets = [{
            "path": "0",
            "coordinateTransformations": [{
                "type": "scale",
                "scale": scale_metadata
            }]
        }]
        # Get data min/max for omero metadata (single level)
        data_min = image_data.min()
        data_max = image_data.max()
    
    image_group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": axes_metadata,
        "datasets": datasets
    }]
    
    image_group.attrs["omero"] = {
        "channels": [{
            "active": True,
            "label": layers[0].name,
            "window": {
                "start": float(data_min),
                "end": float(data_max),
                "min": int(data_min),
                "max": int(data_max)
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
    # Calculate chunk_shape for segmentation (3D: TYX)
    seg_shape = layers[1].data.shape
    if len(seg_shape) == 3:  # TYX
        seg_chunk_shape = (1, seg_shape[1], seg_shape[2])
    else:
        seg_chunk_shape = (1, seg_shape[1], seg_shape[2]) if len(seg_shape) >= 3 else (1, seg_shape[0], seg_shape[1])
    
    write_labels(
        labels = layers[1].data,
        group = root, #label_group,
        name = "TrackedCells",
        axes = "tyx",
        storage_options = dict(chunks=seg_chunk_shape),
        scaler = None,
    )

    if "0" not in root["labels"]["TrackedCells"]:
        raise ValueError("Label data was not written correctly. The expected key '0' is missing in 'TrackedCells'.")
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
    if isinstance(file, str):
        notify(f"{file} has been saved.")
    else:
        notify("OME-zarr file has been saved.")

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
