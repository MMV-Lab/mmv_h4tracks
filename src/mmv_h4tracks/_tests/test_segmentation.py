"""Module providing tests for the segmentation widget"""

import pytest
import numpy as np
from unittest.mock import Mock
from scipy import ndimage

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._reader import build_multiscale
from mmv_h4tracks._tests.data_loading import DATA_ROOT, load_image_zyx
from mmv_h4tracks._tests.fixture_helpers import (
    clear_viewer_layers,
    reset_plugin_state,
    reset_widget,
)

PATH = DATA_ROOT
REMOVE_CELL_SEG = "test_seg"
REMOVE_CELL_TRK = "test_trk"


@pytest.fixture(scope="module")
def remove_cell_testdata():
    """Only the labels + tracks pair needed by remove_cell_from_tracks."""
    seg = load_image_zyx(PATH / "segmentation" / f"{REMOVE_CELL_SEG}.tiff")
    trk = np.load(PATH / "tracks" / f"{REMOVE_CELL_TRK}.npy")
    return seg, trk


@pytest.fixture(scope="module")
def remove_cell_widget_loaded(module_widget, remove_cell_testdata):
    """Attach the slim remove-cell layers once per module."""
    seg, trk = remove_cell_testdata
    reset_widget(module_widget)
    module_widget.viewer.add_labels(np.array(seg, copy=True), name=REMOVE_CELL_SEG)
    module_widget.viewer.add_tracks(np.array(trk, copy=True), name=REMOVE_CELL_TRK)
    yield module_widget


@pytest.fixture
def viewer_with_data(remove_cell_widget_loaded, remove_cell_testdata):
    """
    Soft-reset + restore slim layers for remove_cell_from_tracks.

    Avoids reloading every image/segmentation/tracks file each parametrized case.
    """
    widget = remove_cell_widget_loaded
    seg, trk = remove_cell_testdata

    reset_plugin_state(widget)
    present = {layer.name for layer in widget.viewer.layers}
    if {REMOVE_CELL_SEG, REMOVE_CELL_TRK} - present:
        clear_viewer_layers(widget.viewer)
        widget.viewer.add_labels(np.array(seg, copy=True), name=REMOVE_CELL_SEG)
        widget.viewer.add_tracks(np.array(trk, copy=True), name=REMOVE_CELL_TRK)
    else:
        widget.viewer.layers[REMOVE_CELL_SEG].data = np.array(seg, copy=True)
        widget.viewer.layers[REMOVE_CELL_TRK].data = np.array(trk, copy=True)

    widget.combobox_segmentation.setCurrentText(REMOVE_CELL_SEG)
    widget.combobox_tracks.setCurrentText(REMOVE_CELL_TRK)
    yield widget


@pytest.mark.integration
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 65, 72), (1, 63, 72)])
def test_remove_cell_from_tracks(viewer_with_data, position):
    widget = viewer_with_data
    try:
        widget.segmentation_window.remove_cell_from_tracks(position)
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


def _assert_tracks_schema(tracks: np.ndarray) -> None:
    """Each track ID must have unique, contiguous frames (sorted)."""
    assert tracks.ndim == 2 and tracks.shape[1] == 4
    for trk_id in np.unique(tracks[:, 0]):
        trk = tracks[tracks[:, 0] == trk_id]
        trk = trk[np.argsort(trk[:, 1])]
        frames = trk[:, 1]
        assert len(set(frames.tolist())) == len(frames)
        assert len(frames) == frames[-1] - frames[0] + 1


def _centroid_of_label(seg_frame: np.ndarray, label_id: int) -> list:
    y, x = ndimage.center_of_mass(seg_frame, labels=seg_frame, index=label_id)
    return [int(np.rint(y)), int(np.rint(x))]


@pytest.fixture
def widget_with_short_tracks(create_widget):
    """Labels + tracks for schema checks when removing start/end of short tracks."""
    widget = create_widget
    # 3 frames; label 1 is a small blob at the same place each frame.
    seg = np.zeros((3, 20, 20), dtype=np.int32)
    for z in range(3):
        seg[z, 5:8, 5:8] = 1
    cy, cx = _centroid_of_label(seg[0], 1)
    # Track length 3 and a separate length-2 track (label 2 on frames 0-1).
    seg[:, 12:15, 12:15] = 0
    for z in range(2):
        seg[z, 12:15, 12:15] = 2
    cy2, cx2 = _centroid_of_label(seg[0], 2)

    tracks = np.array(
        [
            [1, 0, cy, cx],
            [1, 1, cy, cx],
            [1, 2, cy, cx],
            [2, 0, cy2, cx2],
            [2, 1, cy2, cx2],
        ],
        dtype=np.int64,
    )
    widget.viewer.add_labels(seg, name="schema_seg")
    widget.viewer.add_tracks(tracks, name="schema_trk")
    widget.combobox_segmentation.setCurrentText("schema_seg")
    widget.combobox_tracks.setCurrentText("schema_trk")
    return widget, (cy, cx), (cy2, cx2)


@pytest.mark.integration
@pytest.mark.schema
@pytest.mark.parametrize(
    "which, end",
    [
        ("long", "start"),
        ("long", "end"),
        ("short", "start"),
        ("short", "end"),
    ],
)
def test_remove_cell_preserves_track_schema(widget_with_short_tracks, which, end):
    """Removing a cell at the start/end of a track keeps remaining tracks valid."""
    widget, (cy, cx), (cy2, cx2) = widget_with_short_tracks
    if which == "long":
        frame = 0 if end == "start" else 2
        position = (frame, cy, cx)
        removed_id = 1
    else:
        frame = 0 if end == "start" else 1
        position = (frame, cy2, cx2)
        removed_id = 2

    before = widget.viewer.layers["schema_trk"].data.copy()
    widget.segmentation_window.remove_cell_from_tracks(position)
    after = widget.viewer.layers["schema_trk"].data

    # Removed centroid should no longer appear for that track id at that frame.
    remaining_same = after[
        (after[:, 0] == removed_id) & (after[:, 1] == frame)
    ]
    assert len(remaining_same) == 0

    if len(after):
        _assert_tracks_schema(after)
    # Length-2 track: removing either end deletes the whole track (both ends).
    if which == "short":
        if len(after):
            assert removed_id not in np.unique(after[:, 0])
    else:
        # Length-3: one end removed → remaining run still present or renumbered.
        assert before.shape[0] > after.shape[0]


def create_mock_event(position):
    """Create a mock napari event with a position attribute."""
    event = Mock()
    event.position = np.array(position)
    return event


@pytest.fixture
def widget_with_multiscale_2d_seg(create_widget):
    """Create widget with multiscale 2D segmentation."""
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
    
    widget.combobox_segmentation.setCurrentText("test_seg_2d_multiscale")
    
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
    """Create widget with single resolution 2D segmentation."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create 2D segmentation with single-pixel cells
    # Shape: (y, x) = (10, 10)
    seg_2d = np.zeros((10, 10), dtype=np.int32)
    seg_2d[5, 5] = 1  # Cell 1 at (5, 5)
    seg_2d[7, 7] = 2  # Cell 2 at (7, 7)
    
    # Add single resolution segmentation layer
    viewer.add_labels(seg_2d, name="test_seg_2d_single")
    
    widget.combobox_segmentation.setCurrentText("test_seg_2d_single")
    
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
