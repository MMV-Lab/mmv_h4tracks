"""Module providing tests for the tracking module."""

import pytest
from unittest.mock import patch
from pathlib import Path
from aicsimageio import AICSImage
import numpy as np

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._tracking import LINK_TEXT, UNLINK_TEXT

PATH = Path(__file__).parent / "data"

@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())

# segmentation and tracking used from a changed version of
# 2022-05-11_example.zarr
@pytest.fixture
def widget_with_seg_trk(create_widget):
    widget = create_widget
    viewer = widget.viewer
    seg_path = Path(PATH / "segmentation" / "test_seg.tiff")
    seg = AICSImage(seg_path).get_image_data("ZYX")
    viewer.add_labels(seg, name="test_seg")
    trk_path = Path(PATH / "tracks" / "test_trk.npy")
    trk = np.load(trk_path)
    viewer.add_tracks(trk, name="test_trk")
    return widget

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
        if name == "test_seg":
            continue
        viewer.add_labels(segmentation, name=name)
    for file in list(Path(PATH / "tracks").iterdir()):
        tracks = np.load(file)
        name = file.stem
        if name == "test_trk":
            continue
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

def check_schema(widget):
    assert len(widget.viewer.layers) == 2
    index = widget.viewer.layers.index("test_trk")
    tracks_layer = widget.viewer.layers[index]
    assert tracks_layer.data.shape[1] == 4
    unique_ids = np.unique(tracks_layer.data[:, 0])
    for trk_id in unique_ids:
        trk = tracks_layer.data[tracks_layer.data[:, 0] == trk_id]
        frames = trk[:, 1]
        low_frame = frames[0]
        # starting with lowest frame
        assert all(frames >= low_frame)
        high_frame = frames[-1]
        # ending with highest frame
        assert all(frames <= high_frame)
        # only unique frames
        assert len(set(frames)) == len(frames)
        # continuous frames
        if len(frames) != high_frame - low_frame + 1:
            print(f"track_id: {trk_id}")
            print(f"track: {trk}")
        assert len(frames) == high_frame - low_frame + 1

@pytest.mark.schema
class TestLink:
    class TestValid:
        class TestSingle:
            # Add one connection
            def test_merge(self, widget_with_seg_trk):
                # Merge two tracks into one
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[1, 390, 390], [2, 393, 393]]
                tracking_widget.link_tracks_on_click()
                check_schema(widget)

            @pytest.mark.parametrize("cells", [
                [[3, 398, 397], [4, 387, 399]],
                [[5, 389, 383], [6, 383, 391]],
            ])
            def test_extend(self, widget_with_seg_trk, cells):
                # Extend a track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.link_tracks_on_click()
                check_schema(widget)
                
            def test_create(self, widget_with_seg_trk):
                # Create a new track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[2, 390, 377], [3, 398, 397]]
                tracking_widget.link_tracks_on_click()
                check_schema(widget)
                
        class TestMultiple:
            # Add multiple (consecutive) connections
            def test_merge(self, widget_with_seg_trk):
                # Merge two tracks into one
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[1, 390, 390], [2, 390, 377], [3, 398, 397], [4, 387, 399]]
                tracking_widget.link_tracks_on_click()
                check_schema(widget)

            @pytest.mark.parametrize("cells", [
                [[2, 390, 377], [3, 398, 397], [4, 387, 399]],
                [[5, 389, 383], [6, 383, 391], [7, 383, 395]],
            ])
            
            def test_extend(self, widget_with_seg_trk, cells):
                # Extend a track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.link_tracks_on_click()
                check_schema(widget)
                
            def test_create(self, widget_with_seg_trk):
                # Create a new track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_insert_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[2, 390, 377], [3, 398, 397], [4, 401, 385]]
                tracking_widget.link_tracks_on_click()
                check_schema(widget)

    class TestInvalid:
        @patch("mmv_h4tracks._tracking.notify")
        def test_same_frame(self, mock_notify, widget_with_seg_trk):
            # Attempt to link two cells in the same frame
            widget = widget_with_seg_trk
            tracking_widget = widget.tracking_window
            tracking_widget.btn_insert_correspondence.setText("Confirm")
            tracking_widget.selected_cells = [[4, 401, 385], [4, 401, 387]]
            tracking_widget.link_tracks_on_click()
            check_schema(widget)
            mock_notify.assert_called_once_with("Looks like you selected multiple cells in slice 4. You can only connect cells from different slices.")
            
        @pytest.mark.parametrize("cell1", [
            [2, 390, 377],
            [2, 393, 393],
        ])
        @pytest.mark.parametrize("cell2", [
            [4, 387, 399],
            [4, 401, 385],
        ])
        @patch("mmv_h4tracks._tracking.notify")
        def test_non_consecutive(self, mock_notify, widget_with_seg_trk, cell1, cell2):
            # Attempt to link two cells that are not consecutive
            widget = widget_with_seg_trk
            tracking_widget = widget.tracking_window
            tracking_widget.btn_insert_correspondence.setText("Confirm")
            tracking_widget.selected_cells = [cell1, cell2]
            tracking_widget.link_tracks_on_click()
            check_schema(widget)
            mock_notify.assert_called_once_with("Gaps in the tracks are not supported yet. Please also select cells in frames [3].")
            
        @pytest.mark.parametrize("cells", [
            [[5, 119, 14], [6, 383, 391]],
            [[5, 392, 275], [6, 383, 391]],
            [[3, 398, 397], [4, 383, 521]],
            [[4, 401, 385], [5, 389, 383]],
        ])
        @patch("mmv_h4tracks._tracking.notify")
        def test_split(self, mock_notify, widget_with_seg_trk, cells):
            # Attempt to add a cell to a track that already has
            # a cell in the same frame
            widget = widget_with_seg_trk
            tracking_widget = widget.tracking_window
            tracking_widget.btn_insert_correspondence.setText("Confirm")
            tracking_widget.selected_cells = cells
            tracking_widget.link_tracks_on_click()
            check_schema(widget)

            valid_call_args = [f"You selected a cell in frame {frame}, but track {track_id} already contains a cell in this frame." for frame, track_id in [[6, 9], [6, 15], [3, 12], [4, 61]]]
            assert mock_notify.call_count == 1
            assert mock_notify.call_args[0][0] in valid_call_args

@pytest.mark.schema
class TestUnlink:
    class TestValid:
        class TestSingle:
            # Remove one connection

            def test_split(self, widget_with_seg_trk):
                # Split a track into two
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[4, 310, 724], [5, 293, 716]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)

            @pytest.mark.parametrize("cells", [
                [[0, 557, 348], [1, 557, 347]],
                [[8, 458, 885], [9, 442, 877]],
                ])
            def test_trim(self, widget_with_seg_trk, cells):
                # Remove a cell from one end of a track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)

            def test_remove(self, widget_with_seg_trk):
                # Remove the entire track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[0, 507, 803], [1, 507, 803]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
        class TestMultiple:
            # Remove multiple (consecutive) connections

            @pytest.mark.parametrize("cells", [
                [[2, 255, 365], [5, 256, 361]],
                [[3, 68, 234], [4, 67, 234], [5, 66, 234]],
            ])
            def test_split(self, widget_with_seg_trk, cells):
                # Split a track into two
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)

            @pytest.mark.parametrize("cells", [
                [[6, 176, 154], [9, 176, 153]],
                [[7, 175, 179], [8, 175, 179], [9, 174, 178]],
                [[0, 210, 192], [3, 210, 188]],
                [[0, 394, 1217], [1, 394, 1216], [2, 393, 1217]],
            ])
            def test_trim(self, widget_with_seg_trk, cells):
                # Remove cells from one end of a track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)

            @pytest.mark.parametrize("cells", [
                [[0, 753, 1068], [1, 754, 1069], [2, 754, 1068], [3, 754, 1068], [4, 754, 1067], [5, 754, 1067], [6, 754, 1067], [7, 754, 1066], [8, 753, 1065], [9, 752, 1065]],
                [[0, 704, 1228], [9, 700, 1227]],
            ])
            def test_remove(self, widget_with_seg_trk, cells):
                # Remove the entire track
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = cells
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
    class TestInvalid:
        class TestNone:
            # Attempt to remove without selecting enough cells

            @patch("mmv_h4tracks._tracking.notify")
            def test_zero(self, mock_notify, widget_with_seg_trk):
                # No cells selected
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = []
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("Please select at least two cells to disconnect!")

            @patch("mmv_h4tracks._tracking.notify")
            def test_one(self, mock_notify, widget_with_seg_trk):
                # Only one cell selected
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[6, 211, 884]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("Please select at least two cells to disconnect!")
        class TestDifferent:
            # Attempt to remove a connection between two cells
            # from different tracks

            @patch("mmv_h4tracks._tracking.notify")
            def test_same_frame(self, mock_notify, widget_with_seg_trk):
                # Both cells in the same frame
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[6, 211, 884], [6, 224, 854]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("Please select cells from the same track to disconnect.")

            @patch("mmv_h4tracks._tracking.notify")
            def test_different_frame(self, mock_notify, widget_with_seg_trk):
                # Cells in different frames
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[4, 177, 155], [5, 175, 180]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("Please select cells from the same track to disconnect.")
        class TestUntracked:
            # Attempt to remove a connection involving
            # untracked cells
            @patch("mmv_h4tracks._tracking.notify")
            def test_one(self, mock_notify, widget_with_seg_trk):
                # One cell is untracked
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[3, 557, 345], [4, 401, 385]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("All selected cells must be tracked.")

            @patch("mmv_h4tracks._tracking.notify")
            def test_both(self, mock_notify, widget_with_seg_trk):
                # Both cells are untracked
                widget = widget_with_seg_trk
                tracking_widget = widget.tracking_window
                tracking_widget.btn_remove_correspondence.setText("Confirm")
                tracking_widget.selected_cells = [[3, 398, 397], [4, 401, 385]]
                tracking_widget.unlink_tracks_on_click()
                check_schema(widget)
                mock_notify.assert_called_once_with("Please select cells from the same track to disconnect.")

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