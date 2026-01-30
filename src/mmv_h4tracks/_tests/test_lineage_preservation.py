"""Module providing tests for lineage graph preservation through track operations."""

import pytest
import numpy as np
from unittest.mock import patch, Mock
from qtpy.QtWidgets import QMessageBox

from mmv_h4tracks import MMVH4TRACKS


@pytest.fixture
def create_widget(make_napari_viewer):
    """Fixture to create a widget instance."""
    yield MMVH4TRACKS(make_napari_viewer())


@pytest.fixture
def widget_with_tracks_and_lineage(create_widget):
    """Fixture to create a widget with tracks layer containing lineage graph.
    
    Creates tracks with IDs 1, 2, 3, 4 where:
    - Track 2 has parent 1
    - Track 3 has parent 1
    - Track 4 has parents 2 and 3
    """
    widget = create_widget
    viewer = widget.viewer
    
    # Create tracks data with track_ids 1, 2, 3, 4
    # Format: [track_id, time, y, x]
    tracks_data = np.array([
        [1, 0, 10, 10],
        [1, 1, 11, 11],
        [1, 2, 12, 12],
        [2, 1, 20, 20],
        [2, 2, 21, 21],
        [2, 3, 22, 22],
        [3, 1, 30, 30],
        [3, 2, 31, 31],
        [3, 3, 32, 32],
        [4, 2, 40, 40],
        [4, 3, 41, 41],
        [4, 4, 42, 42],
    ], dtype=np.int64)
    
    tracks_layer = viewer.add_tracks(tracks_data, name="test_tracks")
    
    # Set up lineage graph
    # Track 2 and 3 have parent 1
    # Track 4 has parents 2 and 3
    tracks_layer.graph = {
        2: [1],
        3: [1],
        4: [2, 3]
    }
    
    widget.combobox_tracks.setCurrentText("test_tracks")
    
    return widget


@pytest.mark.unit
def test_display_selected_tracks_preserves_lineage(widget_with_tracks_and_lineage):
    """Test that display_selected_tracks caches the full graph and sets it to empty for filtered view."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    original_graph = dict(tracks_layer.graph)
    
    # Filter to show only tracks 2 and 3
    tracking_window.display_selected_tracks([2, 3])
    
    # Check that cached_tracks and cached_graph are set
    assert tracking_window.cached_tracks is not None
    assert tracking_window.cached_graph is not None
    assert tracking_window.cached_graph == original_graph
    
    # Check that displayed tracks are filtered
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert displayed_track_ids == {2, 3}
    
    # Check that graph is empty for filtered view (to avoid napari validation errors)
    assert tracks_layer.graph == {}


@pytest.mark.unit
def test_display_cached_tracks_restores_lineage(widget_with_tracks_and_lineage):
    """Test that display_cached_tracks restores the full cached graph."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Normalize graph keys to int for consistent comparison
    original_graph = {int(k): [int(p) for p in v] for k, v in tracks_layer.graph.items()}
    
    # First filter tracks (this caches the graph)
    tracking_window.display_selected_tracks([2, 3])
    
    # Verify graph is cached (normalize for comparison)
    cached_graph_normalized = {int(k): [int(p) for p in v] for k, v in tracking_window.cached_graph.items()}
    assert cached_graph_normalized == original_graph
    
    # Restore cached tracks
    tracking_window.display_cached_tracks()
    
    # Check that all tracks are restored
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert displayed_track_ids == {1, 2, 3, 4}
    
    # Check that full graph is restored (normalize for comparison)
    restored_graph_normalized = {int(k): [int(p) for p in v] for k, v in tracks_layer.graph.items()}
    assert restored_graph_normalized == original_graph
    
    # Check that cache is cleared
    assert tracking_window.cached_tracks is None
    assert tracking_window.cached_graph is None


@pytest.mark.unit
def test_display_cached_tracks_creates_new_layer_with_graph(widget_with_tracks_and_lineage):
    """Test that display_cached_tracks creates new layer with graph when no layer exists."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    original_graph = dict(tracks_layer.graph)
    
    # Filter tracks (this caches the graph)
    tracking_window.display_selected_tracks([2, 3])
    
    # Remove the tracks layer
    widget.viewer.layers.remove(tracks_layer.name)
    
    # Restore cached tracks (should create new layer)
    tracking_window.display_cached_tracks()
    
    # Get the new layer (display_cached_tracks creates layer named "Tracks")
    new_tracks_layer = widget.viewer.layers["Tracks"]
    
    # Check that all tracks are restored
    displayed_track_ids = set(np.unique(new_tracks_layer.data[:, 0]).astype(int))
    assert displayed_track_ids == {1, 2, 3, 4}
    
    # Check that full graph is restored
    assert new_tracks_layer.graph == original_graph


@pytest.mark.unit
def test_create_implicit_tracks_preserves_lineage(widget_with_tracks_and_lineage):
    """Test that create_implicit_tracks_wrapper preserves lineage when creating new layer."""
    widget = widget_with_tracks_and_lineage
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    original_graph = dict(tracks_layer.graph)
    
    # Create minimal segmentation data for create_implicit_tracks
    # Format: (time, y, x) - create a simple 2-frame segmentation
    seg_data = np.zeros((2, 50, 50), dtype=np.int32)
    # Add some labels in frame 0
    seg_data[0, 10:15, 10:15] = 1
    seg_data[0, 20:25, 20:25] = 2
    # Add same labels in frame 1 (so they become tracks)
    seg_data[1, 11:16, 11:16] = 1
    seg_data[1, 21:26, 21:26] = 2
    
    widget.viewer.add_labels(seg_data, name="test_seg")
    widget.combobox_segmentation.setCurrentText("test_seg")
    
    # Mock the _clear_layers to avoid user interaction
    with patch.object(widget, '_clear_layers', return_value=True):
        # Create implicit tracks (this will clear and recreate the layer)
        widget.create_implicit_tracks_wrapper(None)
    
    # Get the new tracks layer
    new_tracks_layer = widget.viewer.layers["Tracks"]
    
    # Check that graph is preserved (filtered to only include valid track IDs)
    # Since create_implicit_tracks creates new tracks from segmentation,
    # the graph should be filtered to only include entries for tracks that still exist
    new_track_ids = set(np.unique(new_tracks_layer.data[:, 0]).astype(int))
    
    # Graph should only contain entries for tracks that still exist
    if new_tracks_layer.graph:
        for track_id in new_tracks_layer.graph:
            assert track_id in new_track_ids
            for parent_id in new_tracks_layer.graph[track_id]:
                assert parent_id in new_track_ids


@pytest.mark.unit
def test_process_new_tracks_preserves_lineage(widget_with_tracks_and_lineage):
    """Test that process_new_tracks preserves lineage when updating tracks."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    original_graph = dict(tracks_layer.graph)
    
    # Create new tracks that are a subset of original (simulating filtered tracks)
    # Keep tracks 1, 2, 3 (remove track 4)
    new_tracks = tracks_layer.data[tracks_layer.data[:, 0] != 4]
    
    tracking_window.process_new_tracks(new_tracks)
    
    # Check that graph is filtered to only include valid track IDs
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert displayed_track_ids == {1, 2, 3}
    
    # Graph should only contain entries for tracks 1, 2, 3
    # Track 2 and 3 have parent 1, so those should be preserved
    assert 2 in tracks_layer.graph
    assert 3 in tracks_layer.graph
    assert tracks_layer.graph[2] == [1]
    assert tracks_layer.graph[3] == [1]
    # Track 4 should not be in graph (it was removed)
    assert 4 not in tracks_layer.graph


@pytest.mark.unit
def test_process_new_tracks_creates_new_layer_without_graph(widget_with_tracks_and_lineage):
    """Test that process_new_tracks creates new layer when none exists, without graph."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Remove the tracks layer
    widget.viewer.layers.remove(tracks_layer.name)
    
    # Create new tracks
    new_tracks = np.array([
        [1, 0, 10, 10],
        [1, 1, 11, 11],
        [2, 1, 20, 20],
        [2, 2, 21, 21],
    ], dtype=np.int64)
    
    tracking_window.process_new_tracks(new_tracks)
    
    # Get the new layer
    new_tracks_layer = widget.viewer.layers["Tracks"]
    
    # New layer should have no graph (since there was no previous layer)
    assert not new_tracks_layer.graph or new_tracks_layer.graph == {}


@pytest.mark.unit
def test_remove_entries_from_tracks_filters_lineage(widget_with_tracks_and_lineage):
    """Test that remove_entries_from_tracks filters lineage when removing entries."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Remove all entries from track 4
    cells_to_remove = [
        [4, 2, 40, 40],
        [4, 3, 41, 41],
        [4, 4, 42, 42],
    ]
    
    tracking_window.remove_entries_from_tracks(cells_to_remove)
    
    # Track 4 should be removed from tracks
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert 4 not in displayed_track_ids
    
    # Track 4 should not be in graph
    assert 4 not in tracks_layer.graph
    
    # Other tracks should still have their lineage
    assert 2 in tracks_layer.graph
    assert 3 in tracks_layer.graph
    # Normalize for comparison (handle potential type differences)
    graph_2_parents = [int(p) for p in tracks_layer.graph[2]]
    graph_3_parents = [int(p) for p in tracks_layer.graph[3]]
    assert graph_2_parents == [1]
    assert graph_3_parents == [1]


@pytest.mark.unit
def test_remove_entries_from_tracks_removes_layer_when_empty(widget_with_tracks_and_lineage):
    """Test that remove_entries_from_tracks removes layer when all tracks are removed."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Remove all entries
    all_cells = [
        [1, 0, 10, 10],
        [1, 1, 11, 11],
        [1, 2, 12, 12],
        [2, 1, 20, 20],
        [2, 2, 21, 21],
        [2, 3, 22, 22],
        [3, 1, 30, 30],
        [3, 2, 31, 31],
        [3, 3, 32, 32],
        [4, 2, 40, 40],
        [4, 3, 41, 41],
        [4, 4, 42, 42],
    ]
    
    tracking_window.remove_entries_from_tracks(all_cells)
    
    # Layer should be removed
    assert "test_tracks" not in widget.viewer.layers
    
    # Cache should be cleared
    assert tracking_window.cached_tracks is None
    assert tracking_window.cached_graph is None


@pytest.mark.unit
def test_assign_new_track_id_updates_lineage(widget_with_tracks_and_lineage):
    """Test that assign_new_track_id updates lineage graph keys when track ID changes."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Change track ID from 2 to 5
    tracking_window.assign_new_track_id(tracks_layer, old_id=2, new_id=5)
    
    # Check that track ID 2 is now 5 in data
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert 2 not in displayed_track_ids
    assert 5 in displayed_track_ids
    
    # Check that graph key is updated from 2 to 5
    assert 2 not in tracks_layer.graph
    assert 5 in tracks_layer.graph
    # Normalize for comparison (handle potential type differences)
    graph_5_parents = [int(p) for p in tracks_layer.graph[5]]
    assert graph_5_parents == [1]  # Parent relationship preserved
    
    # Note: assign_new_track_id only updates the graph key, not parent references in other tracks
    # So track 4 will still reference parent 2 (which no longer exists)
    # preserve_and_filter_graph will filter out invalid parent references
    # Since track 4 had parents [2, 3] and 2 is now invalid, it will be filtered to [3]
    if 4 in tracks_layer.graph:
        # Normalize for comparison
        graph_4_parents = [int(p) for p in tracks_layer.graph[4]]
        # Track 4 should have parent 3 (2 was filtered out since it no longer exists)
        assert 3 in graph_4_parents
        assert 2 not in graph_4_parents


@pytest.mark.unit
def test_add_entries_to_tracks_preserves_lineage(widget_with_tracks_and_lineage):
    """Test that add_entries_to_tracks preserves lineage when adding entries."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    original_graph = dict(tracks_layer.graph)
    
    # Add new entries to existing track 1
    # Format: [time, y, x] - track_id will be inserted at position 0
    new_cells = [
        [3, 13, 13],  # time, y, x (track_id will be added at position 0)
    ]
    
    tracking_window.add_entries_to_tracks(new_cells, track_id=1)
    
    # Check that graph is preserved
    assert tracks_layer.graph == original_graph
    
    # Check that new entries are added
    track_1_entries = tracks_layer.data[tracks_layer.data[:, 0] == 1]
    assert len(track_1_entries) > 3  # Should have more than original 3 entries


@pytest.mark.unit
def test_add_entries_to_tracks_creates_new_layer_without_graph(widget_with_tracks_and_lineage):
    """Test that add_entries_to_tracks creates new layer when none exists, without graph."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Remove the tracks layer
    widget.viewer.layers.remove(tracks_layer.name)
    
    # Add entries (should create new layer)
    # Format: [time, y, x] - track_id will be inserted at position 0
    new_cells = [
        [0, 10, 10],  # time, y, x
    ]
    
    tracking_window.add_entries_to_tracks(new_cells, track_id=1)
    
    # Get the new layer
    new_tracks_layer = widget.viewer.layers["Tracks"]
    
    # New layer should have no graph (since there was no previous layer)
    assert not new_tracks_layer.graph or new_tracks_layer.graph == {}


@pytest.mark.unit
def test_delete_displayed_tracks_filters_lineage(widget_with_tracks_and_lineage):
    """Test that delete_displayed_tracks_on_click filters lineage when removing displayed tracks."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # First filter to show only tracks 2 and 3 (this caches the full graph)
    tracking_window.display_selected_tracks([2, 3])
    
    # Verify cached tracks contains all tracks
    cached_track_ids = set(np.unique(tracking_window.cached_tracks[:, 0]).astype(int))
    assert cached_track_ids == {1, 2, 3, 4}
    
    # Now delete displayed tracks (should remove tracks 2 and 3 from cached tracks)
    # Note: delete_displayed_tracks_on_click doesn't show a message box when cache exists
    # It directly removes the displayed tracks from cached tracks
    tracking_window.delete_displayed_tracks_on_click()
    
    # Check that only tracks 1 and 4 remain (the diff between cached and displayed)
    displayed_track_ids = set(np.unique(tracks_layer.data[:, 0]).astype(int))
    assert displayed_track_ids == {1, 4}
    
    # Graph should be filtered to remove tracks 2 and 3
    assert 2 not in tracks_layer.graph
    assert 3 not in tracks_layer.graph
    
    # Track 4 had parents [2, 3], both of which are now invalid
    # preserve_and_filter_graph will remove track 4 from graph since all its parents are invalid
    assert 4 not in tracks_layer.graph
    
    # Cache should be cleared
    assert tracking_window.cached_tracks is None
    assert tracking_window.cached_graph is None


@pytest.mark.unit
def test_delete_displayed_tracks_removes_layer_when_no_cache(widget_with_tracks_and_lineage):
    """Test that delete_displayed_tracks_on_click removes layer when no cache exists and user confirms."""
    widget = widget_with_tracks_and_lineage
    tracking_window = widget.tracking_window
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Ensure no cache exists
    tracking_window.cached_tracks = None
    tracking_window.cached_graph = None
    
    # Mock the message box to return "Yes"
    with patch('mmv_h4tracks._tracking.QMessageBox') as mock_msg:
        mock_msg_instance = Mock()
        mock_msg_instance.exec_.return_value = QMessageBox.Yes
        mock_msg.return_value = mock_msg_instance
        
        tracking_window.delete_displayed_tracks_on_click()
    
    # Layer should be removed
    assert "test_tracks" not in widget.viewer.layers
