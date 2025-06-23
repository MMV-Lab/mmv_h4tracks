"""Module providing functionality for reading data from disk"""

from qtpy.QtWidgets import QApplication
from ome_zarr.io import parse_url
import zarr
from qtpy.QtWidgets import QFileDialog


def open_dialog(parent, filetype="*.zarr", directory=""):
    """
    Opens a dialog to select a file to open

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
        Path of the selected file
    """
    dialog = QFileDialog()
    dialog.setNameFilter(filetype)
    filetype_name = filetype[2:].capitalize()
    filepath = dialog.getExistingDirectory(
        parent, f"Select {filetype_name}-File", directory=directory
    )
    return filepath


def napari_get_reader(path):
    """
    Determines reader type for file(s) at [path]

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path list or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
        print(f"Selected {path} as path from list")

    # if we can read the file, return the reader function
    if path.endswith(".ome.zarr"):
        return ome_zarr_reader
    elif path.endswith(".zarr"):
        return zarr_reader

    # otherwise return None
    return None

def ome_zarr_reader(filename):
    """
    Take an OME-Zarr file name and read the file in.
    Try to read any metadata we can and put it in the layers
    
    Parameters
    ----------
    filename : str
        Path to a .ome.zarr file
    Returns
    -------
        zarr.Group"""
    # read file
    # calculate tracks from segmentation if applicable
    # add metadata to layers
    store = parse_url(filename, mode="a").store
    root = zarr.open_group(store=store, mode="a")
    return root, True
    array_keys = []

    def get_array_keys(group):
        for key in group.keys():
            if isinstance(group[key], zarr.core.Array):
                array_keys.append(key)
            elif isinstance(group[key], zarr.hierarchy.Group):
                get_array_keys(group[key])

    get_array_keys(root)
    print(array_keys)
    # for key in keys:
    #     if isinstance(root[key], zarr.core.Array):

    #         print(f"Group: {key}")

    raw_image = root.get("0")
    segmentation = root.get("labels/Tracked Cells/0")
    print(f"Raw image: {raw_image}")
    print(f"Segmentation: {segmentation}")
    # results = []

    # def recursive_list(group, prefix=""):
    #     for name, item in group.items():
    #         path = f"{prefix}/{name}".lstrip("/")
    #         attrs = dict(item.attrs)  # Copy to a plain dict for safe printing
    #         results.append((path, attrs))
    #         if isinstance(item, zarr.hierarchy.Group):
    #             recursive_list(item, path)

    # recursive_list(root)

    # print(results)

    QApplication.restoreOverrideCursor()
    raise NotImplementedError("Ome Zarr reading is not implemented yet.")


def zarr_reader(filename):
    """
    Take a file name and read the file in.

    Parameters
    ----------
    filename : str
        Path to a .zarr file

    Returns
    -------
        Array or Group
    """
    return zarr.open(filename, mode="a"), False
