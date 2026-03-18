"""Module providing tests for lineage file loading."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import numpy as np

from mmv_h4tracks import MMVH4TRACKS


@pytest.fixture
def create_widget(make_napari_viewer):
    """Fixture to create a widget instance."""
    yield MMVH4TRACKS(make_napari_viewer())


@pytest.fixture
def widget_with_tracks(create_widget):
    """Fixture to create a widget with tracks layer containing known track_ids (1, 2, 3, 4)."""
    widget = create_widget
    viewer = widget.viewer
    
    # Create tracks data with track_ids 1, 2, 3, 4
    # Format: [track_id, time, y, x]
    tracks_data = np.array([
        [1, 0, 10, 10],
        [1, 1, 11, 11],
        [2, 0, 20, 20],
        [2, 1, 21, 21],
        [3, 0, 30, 30],
        [3, 1, 31, 31],
        [4, 0, 40, 40],
        [4, 1, 41, 41],
    ], dtype=np.int64)
    
    viewer.add_tracks(tracks_data, name="test_tracks")
    widget.combobox_tracks.setCurrentText("test_tracks")
    
    return widget


@pytest.fixture
def create_temp_lineage_file():
    """Context manager for temporary lineage files."""
    class TempLineageFile:
        def __init__(self, content):
            self.content = content
            self.path = None
        
        def __enter__(self):
            # Create a temporary file
            fd, self.path = tempfile.mkstemp(suffix='.txt', text=True)
            with os.fdopen(fd, 'w') as f:
                f.write(self.content)
            return self.path
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.path and os.path.exists(self.path):
                os.unlink(self.path)
    
    return TempLineageFile


# Success cases

@pytest.mark.unit
def test_load_lineage_plain_integer_parent_id(widget_with_tracks, create_temp_lineage_file):
    """Test plain integer format (e.g., "1 0 1 2")."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file: track_id, start_t, end_t, parent_id (plain integer)
    content = "2 0 1 1\n3 0 1 1\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was updated correctly
            assert hasattr(tracks_layer, 'graph')
            assert tracks_layer.graph[2] == [1]
            assert tracks_layer.graph[3] == [1]


@pytest.mark.unit
def test_load_lineage_bracketed_single_parent_id(widget_with_tracks, create_temp_lineage_file):
    """Test single integer in brackets (e.g., "1 0 1 [2]")."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file: track_id, start_t, end_t, parent_id (bracketed single)
    content = "2 0 1 [1]\n3 0 1 [1]\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was updated correctly
            assert hasattr(tracks_layer, 'graph')
            assert tracks_layer.graph[2] == [1]
            assert tracks_layer.graph[3] == [1]


@pytest.mark.unit
def test_load_lineage_bracketed_multiple_parent_ids(widget_with_tracks, create_temp_lineage_file):
    """Test multiple comma-separated parent_ids (e.g., "1 0 1 [2,3]")."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file: track_id, start_t, end_t, parent_id (bracketed multiple)
    content = "4 0 1 [2,3]\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was updated correctly
            assert hasattr(tracks_layer, 'graph')
            assert tracks_layer.graph[4] == [2, 3]


@pytest.mark.unit
def test_load_lineage_mixed_formats(widget_with_tracks, create_temp_lineage_file):
    """Test file with all three formats mixed."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with mixed formats
    content = "2 0 1 1\n3 0 1 [1]\n4 0 1 [2,3]\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was updated correctly
            assert hasattr(tracks_layer, 'graph')
            assert tracks_layer.graph[2] == [1]
            assert tracks_layer.graph[3] == [1]
            assert tracks_layer.graph[4] == [2, 3]


@pytest.mark.unit
def test_load_lineage_initializes_graph_if_none(widget_with_tracks, create_temp_lineage_file):
    """Test graph initialization when it doesn't exist."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Set graph to empty dict (can't delete property, but can set to empty)
    # The widget code handles None/empty by using: getattr(tracks_layer, 'graph', {}) or {}
    tracks_layer.graph = {}
    
    content = "2 0 1 1\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was initialized
            assert hasattr(tracks_layer, 'graph')
            assert isinstance(tracks_layer.graph, dict)
            assert tracks_layer.graph[2] == [1]


@pytest.mark.unit
def test_load_lineage_preserves_existing_graph(widget_with_tracks, create_temp_lineage_file):
    """Test that existing graph entries are preserved."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Initialize graph with existing entry (use track_id 3 which exists in tracks data)
    # Track 3 has parent 1 (both exist in tracks: 1, 2, 3, 4)
    tracks_layer.graph = {3: [1]}
    
    content = "2 0 1 1\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that existing entry is preserved and new entry is added
            assert tracks_layer.graph[3] == [1]
            assert tracks_layer.graph[2] == [1]


@pytest.mark.unit
def test_load_lineage_empty_lines_skipped(widget_with_tracks, create_temp_lineage_file):
    """Test that empty lines are ignored."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with empty lines
    content = "2 0 1 1\n\n3 0 1 1\n   \n4 0 1 [2,3]\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that graph was updated correctly (empty lines should be skipped)
            assert hasattr(tracks_layer, 'graph')
            assert tracks_layer.graph[2] == [1]
            assert tracks_layer.graph[3] == [1]
            assert tracks_layer.graph[4] == [2, 3]


# Failure cases

@pytest.mark.unit
def test_load_lineage_no_tracks_layer_selected(create_widget):
    """No tracks layer selected (should return early)."""
    widget = create_widget
    widget.combobox_tracks.setCurrentText("")  # No selection
    
    with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
        widget.hotkey_load_lineage_file(None)
        
        # File dialog should not be called
        mock_dialog.return_value.getOpenFileName.assert_not_called()


@pytest.mark.unit
def test_load_lineage_invalid_tracks_layer(create_widget):
    """Tracks layer doesn't exist (should return early)."""
    widget = create_widget
    widget.combobox_tracks.setCurrentText("nonexistent_tracks")
    
    with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
        widget.hotkey_load_lineage_file(None)
        
        # File dialog should not be called
        mock_dialog.return_value.getOpenFileName.assert_not_called()


@pytest.mark.unit
def test_load_lineage_user_cancels_dialog(widget_with_tracks):
    """User cancels file dialog."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Initialize graph to check it's not modified
    if not hasattr(tracks_layer, 'graph'):
        tracks_layer.graph = {}
    original_graph = dict(tracks_layer.graph)
    
    with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
        # Return empty filepath (user canceled)
        mock_dialog.return_value.getOpenFileName.return_value = ("", "Text files (*.txt)")
        
        widget.hotkey_load_lineage_file(None)
        
        # Graph should not be modified
        assert tracks_layer.graph == original_graph


@pytest.mark.unit
def test_load_lineage_file_not_found(widget_with_tracks):
    """File doesn't exist."""
    widget = widget_with_tracks
    
    with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
        # Return non-existent filepath
        mock_dialog.return_value.getOpenFileName.return_value = ("/nonexistent/file.txt", "Text files (*.txt)")
        
        # Should not raise exception, just print error
        widget.hotkey_load_lineage_file(None)


@pytest.mark.unit
def test_load_lineage_invalid_track_id(widget_with_tracks, create_temp_lineage_file):
    """Track_id doesn't exist in tracks layer (should skip)."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with invalid track_id (99 doesn't exist)
    content = "2 0 1 1\n99 0 1 1\n3 0 1 1\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that only valid track_ids are in graph
            assert hasattr(tracks_layer, 'graph')
            assert 2 in tracks_layer.graph
            assert 3 in tracks_layer.graph
            assert 99 not in tracks_layer.graph


@pytest.mark.unit
def test_load_lineage_insufficient_values(widget_with_tracks, create_temp_lineage_file):
    """Lines with < 4 values (should skip)."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with insufficient values
    content = "2 0 1\n3 0 1 1\n4 0\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that only valid line is in graph
            assert hasattr(tracks_layer, 'graph')
            assert 3 in tracks_layer.graph
            assert 2 not in tracks_layer.graph
            assert 4 not in tracks_layer.graph


@pytest.mark.unit
def test_load_lineage_non_integer_values(widget_with_tracks, create_temp_lineage_file):
    """Lines with non-integer values (should skip)."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with non-integer values
    content = "2 0 1 abc\n3 0 1 1\n4 x y z\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Check that only valid line is in graph
            assert hasattr(tracks_layer, 'graph')
            assert 3 in tracks_layer.graph
            assert 2 not in tracks_layer.graph
            assert 4 not in tracks_layer.graph


@pytest.mark.unit
def test_load_lineage_empty_file(widget_with_tracks, create_temp_lineage_file):
    """Empty file handling."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Empty file
    content = ""
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Graph should exist but be empty (or unchanged if it existed)
            assert hasattr(tracks_layer, 'graph')


@pytest.mark.unit
def test_load_lineage_all_track_ids_invalid(widget_with_tracks, create_temp_lineage_file):
    """All track_ids invalid."""
    widget = widget_with_tracks
    tracks_layer = widget.viewer.layers["test_tracks"]
    
    # Lineage file with all invalid track_ids
    content = "99 0 1 1\n100 0 1 1\n"
    
    with create_temp_lineage_file(content) as filepath:
        with patch('mmv_h4tracks._widget.QFileDialog') as mock_dialog:
            mock_dialog.return_value.getOpenFileName.return_value = (filepath, "Text files (*.txt)")
            
            widget.hotkey_load_lineage_file(None)
            
            # Graph should exist but contain no entries (or unchanged if it existed)
            assert hasattr(tracks_layer, 'graph')
            assert 99 not in tracks_layer.graph
            assert 100 not in tracks_layer.graph
