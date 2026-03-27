"""Module providing functionality for reading data from disk"""

from ome_zarr.io import parse_url
import zarr
import numpy as np
from pathlib import Path
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
    store = parse_url(filename, mode="a").store
    root = zarr.open_group(store=store, mode="a")
    return root, True

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


def check_multiscale_image(zarr_file):
    """
    Check if raw image in an OME-Zarr file is multiscale.
    
    For multiscale files, metadata is on root group.
    For single resolution files, metadata is on "0" group.
    
    Parameters
    ----------
    zarr_file : zarr.Group
        The OME-Zarr file/group to check
        
    Returns
    -------
    bool
        True if image is multiscale (multiple datasets in multiscales metadata), False otherwise
    """
    # Check root attrs first (for multiscale files)
    try:
        root_metadata = dict(zarr_file.attrs)
        multiscales_metadata = root_metadata.get("multiscales", [])
        if multiscales_metadata and len(multiscales_metadata[0].get("datasets", [])) > 1:
            return True
    except (AttributeError, KeyError, IndexError):
        pass
    
    # Check "0" group attrs (for single resolution files)
    try:
        raw_image = zarr_file.get("0")
        img_metadata = dict(raw_image.attrs)
        multiscales_metadata = img_metadata.get("multiscales", [])
        if multiscales_metadata and len(multiscales_metadata[0].get("datasets", [])) > 1:
            return True
    except (AttributeError, KeyError, IndexError):
        pass
    
    return False


def build_multiscale(image: np.ndarray):
    """
    Return list of progressively downsampled versions of the given image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array
        
    Returns
    -------
    list
        List of image arrays at different resolutions (factors 1, 2, 4, 8)
    """
    levels = [image]
    current = image
    for _ in range(3):
        if any(dim <= 1 for dim in current.shape[-2:]):
            break
        current = current[..., ::2, ::2]
        levels.append(current)
    return levels


def load_zarr_data(zarr_file):
    """
    Load raw image, segmentation, and tracking data from a zarr file.
    
    Parameters
    ----------
    zarr_file : zarr.Group
        The zarr file/group to load from
        
    Returns
    -------
    tuple
        (raw_levels, segmentation, tracks, is_multiscale)
        - raw_levels: list of numpy arrays (multiscale pyramid for display)
        - segmentation: numpy array
        - tracks: numpy array (filtered to remove single-frame tracks)
        - is_multiscale: bool (always False for non-OME zarr)
    """
    # Non-OME zarr files never have multiscale raw data
    is_multiscale = False
    raw_data_source = zarr_file["raw_data"]
    # Original was single array
    raw_data = raw_data_source[:]
    raw_levels = build_multiscale(raw_data)
    
    segmentation = zarr_file["segmentation_data"][:]
    
    # Load and filter tracks
    tracks = zarr_file["tracking_data"][:]
    # Filter track ids of tracks that just occur once
    count_of_track_ids = np.unique(tracks[:, 0], return_counts=True)
    filtered_track_ids = np.delete(
        count_of_track_ids, count_of_track_ids[1] == 1, 1
    )
    
    # Remove tracks that only exist in one slice
    filtered_tracks = np.delete(
        tracks,
        np.where(np.isin(tracks[:, 0], filtered_track_ids[0, :], invert=True)),
        0,
    )
    
    return raw_levels, segmentation, filtered_tracks, is_multiscale


def load_ome_zarr_data(zarr_file, zarr_path=None):
    """
    Load raw image, segmentation, and metadata from an OME-Zarr file.
    
    Parameters
    ----------
    zarr_file : zarr.Group
        The OME-Zarr file/group to load from
    zarr_path : str, optional
        Filesystem path to the zarr file (used if store path cannot be inferred)
    
    Returns
    -------
    tuple
        (raw_levels, segmentation, metadata, is_multiscale, tracks)
        - raw_levels: list of numpy arrays (multiscale pyramid for display)
        - segmentation: numpy array
        - metadata: dict with keys 'frames', 'unit', 'implied_tracks', 'img_metadata'
        - is_multiscale: bool indicating if original data was multiscale
        - tracks: numpy array or None if tracks.npy doesn't exist
    """
    generic_metadata = dict(zarr_file.attrs)
    try:
        labels_metadata = dict(zarr_file.get("labels").attrs)
    except AttributeError:
        raise ValueError("No labels found in OME-Zarr file.")
    
    label_value = labels_metadata.get("labels", "TrackedCells")
    label_name = label_value[0] if isinstance(label_value, (list, tuple)) else label_value
    
    # Check if original was multiscale first
    is_multiscale = check_multiscale_image(zarr_file)
    
    # Read raw image metadata - for multiscale, it's on root; for single, it's on "0"
    if is_multiscale:
        # Multiscale: metadata is on root group
        try:
            img_metadata = dict(zarr_file.attrs)
            if not img_metadata:
                print(f"Warning: root attrs is empty. Available keys in root: {list(zarr_file.keys())}")
        except AttributeError:
            raise ValueError("No image metadata found in OME-Zarr file.")
        
        # Original was multiscale - read all levels
        if "multiscales" not in img_metadata or len(img_metadata["multiscales"]) == 0:
            print("Warning: multiscales metadata not found in root, trying to infer from available datasets")
            # Fallback: look for numeric keys in root
            numeric_keys = sorted([int(k) for k in zarr_file.keys() if k.isdigit()])
            if len(numeric_keys) > 1:
                raw_levels = [zarr_file.get(str(k))[:] for k in numeric_keys]
            else:
                raise ValueError("Could not determine multiscale structure")
        else:
            datasets = img_metadata["multiscales"][0]["datasets"]
            raw_levels = [zarr_file.get(ds["path"])[:] for ds in datasets]
    else:
        # Single resolution: metadata is on "0" group
        raw_image = zarr_file.get("0")
        try:
            img_metadata = dict(raw_image.attrs)
        except AttributeError:
            raise ValueError("No image metadata found in OME-Zarr file.")
        # Original was single resolution
        raw_levels = build_multiscale(raw_image[:])
    
    # Read segmentation
    segmentation = zarr_file.get(f"labels/{label_name}/0")
    try:
        seg_metadata = dict(zarr_file.get(f"labels/{label_name}").attrs)
    except AttributeError:
        seg_metadata = dict()
    
    # Extract metadata
    frames = generic_metadata.get("Frames", None)
    # Try to get unit from multiscales metadata if available
    unit = "unit"
    if "multiscales" in img_metadata and len(img_metadata["multiscales"]) > 0:
        unit = next(
            (ax.get("unit", "unit") for ax in img_metadata["multiscales"][0].get("axes", []) if ax.get("name") in ["z", "y", "x"]),
            "unit"
        )
    
    implied_tracks = seg_metadata.get("implied_tracks", False)
    
    metadata = {
        "frames": frames,
        "unit": unit,
        "implied_tracks": implied_tracks,
        "img_metadata": img_metadata,
    }
    
    # Try to infer filesystem path from zarr store
    if zarr_path is None:
        try:
            store = zarr_file.store
            # For DirectoryStore, path is usually available
            if hasattr(store, 'path'):
                zarr_path = store.path
            elif hasattr(store, 'dir_path'):
                zarr_path = store.dir_path
            # For some store types, we might need to check the root path
            elif hasattr(store, 'root'):
                zarr_path = store.root
        except (AttributeError, TypeError):
            pass
    
    # Try to load tracks from tables/tracks/tracks.npy if it exists
    tracks = None
    if zarr_path:
        tracks_path = Path(zarr_path) / "tables" / "tracks" / "tracks.npy"
        if tracks_path.exists():
            try:
                tracks = np.load(tracks_path)
            except Exception as e:
                print(f"Warning: Could not load tracks.npy: {e}")
    
    return raw_levels, segmentation, metadata, is_multiscale, tracks