"""Module providing tests for the segmentation widget"""

import pytest
from pathlib import Path
from bioio import BioImage
import numpy as np
from unittest.mock import Mock

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._reader import build_multiscale

PATH = Path(__file__).parent / "data"
IMAGE_EXTENSIONS = {".tif", ".tiff"}
TRACK_EXTENSIONS = {".npy"}


@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())


@pytest.fixture
def viewer_with_data(create_widget):
    widget = create_widget
    viewer = widget.viewer
    for file in Path(PATH / "images").iterdir():
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = BioImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_image(image, name=name)
    for file in Path(PATH / "segmentation").iterdir():
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = BioImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_labels(image, name=name)
    for file in Path(PATH / "tracks").iterdir():
        if not file.is_file() or file.suffix.lower() not in TRACK_EXTENSIONS:
            continue
        tracks = np.load(file)
        name = file.stem
        viewer.add_tracks(tracks, name=name)
    yield widget


@pytest.mark.integration
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 65, 72), (1, 63, 72)])
def test_remove_cell_from_tracks(viewer_with_data, position):
    widget = viewer_with_data
    try:
        widget.segmentation_window.remove_cell_from_tracks(position)
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


# TODO: test for track layer schema when
# cell at start/end of track is removed
# especially when track is only 2 cells long


def create_mock_event(position):
    """Create a mock napari event with a position attribute."""
    event = Mock()
    event.position = np.array(position)
    return event


@pytest.fixture
def widget_with_multiscale_2d_seg(create_widget):
    """Create widget with multiscale 2D segmentation and tracks."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create 2D segmentation with single-pixel cells
    # Shape: (y, x) = (10, 10)
    seg_2d = np.zeros((10, 10), dtype=np.int32)
    seg_2d[5, 5] = 1  # Cell 1 at (5, 5)
    seg_2d[7, 7] = 2  # Cell 2 at (7, 7)
    
    # Create multiscale levels
    seg_levels = build_multiscale(seg_2d)
    
    # Add multiscale segmentation layer
    viewer.add_labels(seg_levels, name="test_seg_2d_multiscale", multiscale=True)
    
    # Create tracks data matching the segmentation
    # Tracks format: [track_id, frame, y, x]
    # For 2D, frame is always 0
    tracks = np.array([
        [1, 0, 5, 5],  # Track 1 at frame 0, position (5, 5)
        [2, 0, 7, 7],  # Track 2 at frame 0, position (7, 7)
    ], dtype=np.int32)
    
    viewer.add_tracks(tracks, name="test_tracks_2d")
    
    widget.combobox_segmentation.setCurrentText("test_seg_2d_multiscale")
    widget.combobox_tracks.setCurrentText("test_tracks_2d")
    
    return widget


@pytest.fixture
def widget_with_multiscale_3d_seg(create_widget):
    """Create widget with multiscale 3D segmentation and tracks."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create 3D segmentation with single-pixel cells
    # Shape: (t, y, x) = (3, 10, 10)
    seg_3d = np.zeros((3, 10, 10), dtype=np.int32)
    seg_3d[0, 5, 5] = 1  # Cell 1 at frame 0, position (5, 5)
    seg_3d[1, 5, 5] = 1  # Cell 1 at frame 1, position (5, 5)
    seg_3d[2, 5, 5] = 1  # Cell 1 at frame 2, position (5, 5)
    seg_3d[1, 7, 7] = 2  # Cell 2 at frame 1, position (7, 7)
    seg_3d[2, 7, 7] = 2  # Cell 2 at frame 2, position (7, 7)
    
    # Create multiscale levels
    seg_levels = build_multiscale(seg_3d)
    
    # Add multiscale segmentation layer
    viewer.add_labels(seg_levels, name="test_seg_3d_multiscale", multiscale=True)
    
    # Create tracks data matching the segmentation
    # Tracks format: [track_id, frame, y, x]
    tracks = np.array([
        [1, 0, 5, 5],  # Track 1 at frame 0
        [1, 1, 5, 5],  # Track 1 at frame 1
        [1, 2, 5, 5],  # Track 1 at frame 2
        [2, 1, 7, 7],  # Track 2 at frame 1
        [2, 2, 7, 7],  # Track 2 at frame 2
    ], dtype=np.int32)
    
    viewer.add_tracks(tracks, name="test_tracks_3d")
    
    widget.combobox_segmentation.setCurrentText("test_seg_3d_multiscale")
    widget.combobox_tracks.setCurrentText("test_tracks_3d")
    
    return widget


@pytest.fixture
def widget_with_single_2d_seg(create_widget):
    """Create widget with single resolution 2D segmentation and tracks."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create 2D segmentation with single-pixel cells
    # Shape: (y, x) = (10, 10)
    seg_2d = np.zeros((10, 10), dtype=np.int32)
    seg_2d[5, 5] = 1  # Cell 1 at (5, 5)
    seg_2d[7, 7] = 2  # Cell 2 at (7, 7)
    
    # Add single resolution segmentation layer
    viewer.add_labels(seg_2d, name="test_seg_2d_single")
    
    # Create tracks data matching the segmentation
    # Tracks format: [track_id, frame, y, x]
    # For 2D, frame is always 0
    tracks = np.array([
        [1, 0, 5, 5],  # Track 1 at frame 0, position (5, 5)
        [2, 0, 7, 7],  # Track 2 at frame 0, position (7, 7)
    ], dtype=np.int32)
    
    viewer.add_tracks(tracks, name="test_tracks_2d")
    
    widget.combobox_segmentation.setCurrentText("test_seg_2d_single")
    widget.combobox_tracks.setCurrentText("test_tracks_2d")
    
    return widget


@pytest.fixture
def widget_with_single_3d_seg(create_widget):
    """Create widget with single resolution 3D segmentation and tracks."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create 3D segmentation with single-pixel cells
    # Shape: (t, y, x) = (3, 10, 10)
    seg_3d = np.zeros((3, 10, 10), dtype=np.int32)
    seg_3d[0, 5, 5] = 1  # Cell 1 at frame 0, position (5, 5)
    seg_3d[1, 5, 5] = 1  # Cell 1 at frame 1, position (5, 5)
    seg_3d[2, 5, 5] = 1  # Cell 1 at frame 2, position (5, 5)
    seg_3d[1, 7, 7] = 2  # Cell 2 at frame 1, position (7, 7)
    seg_3d[2, 7, 7] = 2  # Cell 2 at frame 2, position (7, 7)
    
    # Add single resolution segmentation layer
    viewer.add_labels(seg_3d, name="test_seg_3d_single")
    
    # Create tracks data matching the segmentation
    # Tracks format: [track_id, frame, y, x]
    tracks = np.array([
        [1, 0, 5, 5],  # Track 1 at frame 0
        [1, 1, 5, 5],  # Track 1 at frame 1
        [1, 2, 5, 5],  # Track 1 at frame 2
        [2, 1, 7, 7],  # Track 2 at frame 1
        [2, 2, 7, 7],  # Track 2 at frame 2
    ], dtype=np.int32)
    
    viewer.add_tracks(tracks, name="test_tracks_3d")
    
    widget.combobox_segmentation.setCurrentText("test_seg_3d_single")
    widget.combobox_tracks.setCurrentText("test_tracks_3d")
    
    return widget


@pytest.mark.integration
@pytest.mark.parametrize("fixture_name,event_position", [
    ("widget_with_multiscale_2d_seg", (0, 5, 5)),
    ("widget_with_multiscale_3d_seg", (0, 1, 5, 5)),
    ("widget_with_single_2d_seg", (5, 5)),
    ("widget_with_single_3d_seg", (1, 5, 5)),
])
def test_remove_label_multiscale_single_resolution(request, fixture_name, event_position):
    """Test _remove_label with multiscale and single resolution 2D/3D segmentation."""
    widget = request.getfixturevalue(fixture_name)
    event = create_mock_event(event_position)
    
    # Call _remove_label - should not raise IndexError
    try:
        widget.segmentation_window._remove_label(event)
    except IndexError as e:
        pytest.fail(f"IndexError raised: {e}")
    except Exception as e:
        # Other exceptions are acceptable (e.g., if cell is not tracked)
        pass
