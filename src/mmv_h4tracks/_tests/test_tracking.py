"""Module providing tests for the tracking module."""

import pytest
from pathlib import Path
from aicsimageio import AICSImage
import numpy as np

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._tracking import LINK_TEXT, UNLINK_TEXT

PATH = Path(__file__).parent / "data"

@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())

@pytest.fixture
def viewer_with_data(create_widget):
    widget = create_widget
    viewer = widget.viewer
    for file in list(Path(PATH / "images").iterdir()):
        image = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_image(image, name=name)
    for file in list(Path(PATH / "segmentation").iterdir()):
        segmentation = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_labels(segmentation, name=name)
    for file in list(Path(PATH / "tracks").iterdir()):
        tracks = np.load(file)
        name = file.stem
        viewer.add_tracks(tracks, name=name)
    yield widget

pytestmark = pytest.mark.tracking

# @pytest.mark.unit
# @pytest.mark.misc
# @pytest.mark.not_implemented
# def test_set_callback(viewer_with_data):
#     pass

@pytest.mark.unit
@pytest.mark.misc
def test_reset_button_labels(create_widget):
    window = create_widget.tracking_window
    window.btn_insert_correspondence.setText("Test")
    window.btn_remove_correspondence.setText("Test")
    window.reset_button_labels()
    assert window.btn_insert_correspondence.text() == LINK_TEXT
    assert window.btn_remove_correspondence.text() == UNLINK_TEXT

@pytest.mark.unit
@pytest.mark.misc
def test_assign_new_track_id(create_widget):
    widget = create_widget
    window = widget.tracking_window
    window.viewer = widget.viewer
    initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1]])
    tracks_layer = widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window.assign_new_track_id(tracks_layer, 0, 1)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[1,0,1,1], [1,1,1,1], [1,2,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_link_stored_cells_append(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.selected_cells = [[2,1,1], [3,1,1]]
    window.link_stored_cells()
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_link_stored_cells_no_layer(create_widget):
    widget = create_widget
    window = widget.tracking_window
    window.selected_cells = [[0,1,1], [1,1,1]]
    window.link_stored_cells()
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[1,0,1,1], [1,1,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_link_stored_cells_connect(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [1,2,1,1], [1,3,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.selected_cells = [[1,1,1], [2,1,1]]
    window.link_stored_cells()
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
    assert np.array_equal(post_tracks, expected_result)

# @pytest.mark.integration
# @pytest.mark.misc
# @pytest.mark.broken
# def test_link_stored_cells_no_selection(create_widget):
#     widget = create_widget
#     initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
#     widget.viewer.add_tracks(initial_tracks, name="tracks")
#     window = widget.tracking_window
#     window.link_stored_cells()
#     post_tracks = widget.viewer.layers["tracks"].data
#     expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
#     assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_link_stored_cells_enclosed(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,1,1,1], [0,2,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.selected_cells = [[0,1,1], [1,1,1], [2,1,1], [3,1,1]]
    window.link_stored_cells()
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[1,0,1,1], [1,1,1,1], [1,2,1,1], [1,3,1,1]])
    assert np.array_equal(post_tracks, expected_result)

# @pytest.mark.integration
# @pytest.mark.misc
# def test_link_stored_cells_existing(create_widget):
#     widget = create_widget
#     initial_tracks = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
#     widget.viewer.add_tracks(initial_tracks, name="Tracks")
#     window = widget.tracking_window
#     window.selected_cells = [[1,1,1], [2,1,1]]
#     window.link_stored_cells()
#     post_tracks = widget.viewer.layers["Tracks"].data
#     expected_result = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
#     assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_no_layer(create_widget):
    widget = create_widget
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,1,1], [4,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[1,0,1,1], [1,1,1,1], [1,2,1,1], [1,3,1,1], [1,4,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_existing_layer(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,2,2], [1,2,2], [2,2,2], [3,2,2], [4,2,2]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [1,0,2,2], [1,1,2,2], [1,2,2,2], [1,3,2,2], [1,4,2,2]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_contained_track(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,1,1], [4,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_contained_track_multiple(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1], [1,4,1,1], [1,5,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,1,1], [4,1,1], [5,1,1], [6,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1], [0,5,1,1], [0,6,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_extend_low(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,1,1], [4,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_extend_high(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,0,1,1],[0,1,1,1], [0,2,1,1], [0,3,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,1,1], [4,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1], [0,4,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_diverge(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,1,1,1], [0,2,1,1], [0,3,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,1,1], [1,1,1], [2,1,1], [3,2,2], [4,2,2]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
    assert np.array_equal(post_tracks, expected_result)

@pytest.mark.integration
@pytest.mark.misc
def test_evaluate_proposed_track_converge(create_widget):
    widget = create_widget
    initial_tracks = np.array([[0,4,1,1], [0,5,1,1], [0,6,1,1]])
    widget.viewer.add_tracks(initial_tracks, name="Tracks")
    window = widget.tracking_window
    window.viewer = widget.viewer
    proposed_track = [[0,2,2], [1,2,2], [2,2,2], [3,2,2], [4,2,2], [5,1,1]]
    window.evaluate_proposed_track(proposed_track)
    post_tracks = widget.viewer.layers["Tracks"].data
    expected_result = np.array([[0,4,1,1], [0,5,1,1], [0,6,1,1], [1,0,2,2], [1,1,2,2], [1,2,2,2], [1,3,2,2], [1,4,2,2]])
    assert np.array_equal(post_tracks, expected_result)

# @pytest.mark.integration
# @pytest.mark.misc
# @pytest.mark.broken
# def test_evaluate_proposed_track_(create_widget):
#     widget = create_widget
#     initial_tracks = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1]])
#     widget.viewer.add_tracks(initial_tracks, name="Tracks")
#     window = widget.tracking_window
#     window.viewer = widget.viewer
#     proposed_track = [[0,1,1], [1,1,1]]
#     window.evaluate_proposed_track(proposed_track)
#     post_tracks = widget.viewer.layers["Tracks"].data
#     expected_result = np.array([[0,0,1,1], [0,1,1,1], [0,2,1,1], [0,3,1,1]])
#     assert np.array_equal(post_tracks, expected_result)

### Untested Functions
## on_clicks
# coordinate_tracking_on_click
# overlap_tracking_on_click
# single_overlap_tracking_on_click
# unlink_tracks_on_click
# filter_tracks_on_click
# delete_listed_tracks_on_click
# delete_displayed_tracks_on_click
## workers
# worker_overlap_tracking
# worker_single_overlap_tracking
## callbacks
# overlap_tracking_callback
## misc
# on_yielded
# store_cell_for_link
# store_cell_for_unlink
# unlink_stored_cells
# process_new_tracks
# remove_entries_from_tracks
# display_cached_tracks
# display_selected_tracks
# add_entries_to_tracks
# add_track_to_tracks
# get_tracks_layer
# restore_callbacks
# set_callbacks
    
### Partially tested functions
# evaluate_proposed_track
# link_tracking_on_click
# link_stored_cells