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
    print("Loading from zarr")
    dialog = QFileDialog()
    dialog.setNameFilter(filetype)
    filetype_name = filetype[2:].capitalize()
    print("Showing dialog")
    filepath = dialog.getExistingDirectory(
        parent, "Select {}-File".format(filetype_name), directory=directory
    )
    print("Dialog has been closed")
    print("Selected {} as path".format(filepath))
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
    print("Got {} as path".format(path))
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
        print("Selected {} as path from list".format(path))

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".zarr"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return zarr_reader


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
    return zarr.open(filename, mode="a")
