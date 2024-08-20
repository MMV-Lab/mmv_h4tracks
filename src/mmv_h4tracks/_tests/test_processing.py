"""Module providing tests for the processing module."""
import numpy as np
import pytest
from pathlib import Path

import napari
from unittest.mock import patch
from aicsimageio import AICSImage

from mmv_h4tracks import _processing as processing, MMVH4TRACKS
from mmv_h4tracks._segmentation import SegmentationWindow
from mmv_h4tracks._tracking import TrackingWindow

@pytest.fixture
def create_widget(make_napari_viewer):
    return MMVH4TRACKS(make_napari_viewer())

pytestmark = pytest.mark.processing

@pytest.fixture
def widget_with_segmentation(create_widget):
    widget = create_widget
    path = Path(__file__).parent / "data" / "segmentation" / "test_seg.tiff"
    segmentation = AICSImage(path).get_image_data("ZYX")
    widget.viewer.add_labels(segmentation, name="segmentation")
    return widget

@pytest.mark.format
@pytest.mark.unit
def test_segment_slice_cpu():
    layer_slice = np.zeros((100, 100), dtype=np.int8)
    parameters = {
            "model_path": str(Path(__file__).parent.parent.absolute() / "models" / "Neutrophil granulocytes"),
            "diameter": 15,
            "channels": [0, 0],
            "flow_threshold": 0.4,
            "cellprob_threshold": 0,
    }
    assert processing.segment_slice_cpu(layer_slice, parameters).shape == (100, 100)

@pytest.mark.format
@pytest.mark.unit
def test_calculate_centroid():
    layer_slice = np.zeros((100, 100), dtype=np.int8)
    centroids, labels = processing.calculate_centroids(layer_slice)
    assert centroids == []
    assert labels.shape == (0,)
    assert labels.dtype == np.int8



@pytest.mark.unit
def test_read_custom_model_dict(create_widget):
    widget = create_widget
    model_dict = processing.read_custom_model_dict()
    assert model_dict == {}

@pytest.mark.unit
def test_read_models(create_widget):
    widget = create_widget
    segmentation_widget = SegmentationWindow(widget)
    hardcoded_models, custom_models = processing.read_models(segmentation_widget)
    
    assert hardcoded_models == ["Neutrophil_granulocytes"]
    assert custom_models == []

@pytest.mark.unit
def test_display_models(create_widget):
    widget = create_widget
    segmentation_widget = SegmentationWindow(widget)
    hardcoded_models, custom_models = processing.read_models(segmentation_widget)
    processing.display_models(segmentation_widget, hardcoded_models, custom_models)
    assert segmentation_widget.combobox_segmentation.count() == 1
    assert segmentation_widget.combobox_segmentation.currentText() == "Neutrophil_granulocytes"

@pytest.mark.integration
@pytest.mark.schema
def test_track_segmentation_schema(widget_with_segmentation, qtbot):
    # TODO:
    # test:
    # - _get_segmentation_data
    # - _check_for_tracks_layer
    # - _calculate_centroids_parallel
    # - _match_centroids_parallel
    # - _process_matches
    widget = widget_with_segmentation
    # add segmentation layer to widget
    tracking_widget = TrackingWindow(widget)
    # check:
    # - segmentation layer exists
    # - tracking layer does not exist
    viewer = widget.viewer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "segmentation"
    assert type(layer) == napari.layers.Labels
    worker = processing._track_segmentation(tracking_widget)
    trk_data = None
    def capture_result(result):
        nonlocal trk_data
        trk_data = result
    worker.returned.connect(capture_result)
    with qtbot.waitSignal(worker.returned, timeout=60000) as blocker:
        blocker.wait()
    # check:
    # - tracking layer data exists
    # - tracking layer data is nx4
    # - tracking layer data only has continuous tracks
    # - z_0 of track n <= z_0 of track n+1
    assert trk_data.shape[1] == 4
    unique_ids, indices = np.unique(trk_data[:, 0], return_index=True)
    ordered_unique_ids = unique_ids[np.argsort(indices)]
    # ids are in correct order
    assert np.array_equal(unique_ids, ordered_unique_ids)
    # ids are continuous
    assert unique_ids[-1] - unique_ids[0] + 1 == len(unique_ids)
    lower_bound_frame = -1
    for trk_id in ordered_unique_ids:
        trk = trk_data[trk_data[:, 0] == trk_id]
        frames = trk[:, 1]
        low_frame = frames[0]
        # starting with lowest frame
        assert all(frames >= low_frame)
        # starting frame is not lower than previous track
        assert low_frame >= lower_bound_frame
        lower_bound_frame = low_frame
        high_frame = frames[-1]
        # ending with highest frame
        assert all(frames <= high_frame)
        # only unique frames
        assert len(set(frames)) == len(frames)
        # continuous frames
        assert len(frames) == high_frame - low_frame + 1

    # assert calls:
    # - _get_segmentation_data x1 with widget
    # - _check_for_tracks_layer x1 with widget
    # - _calculate_centroids_parallel x1 with (widget, data)
    # - _match_centroids_parallel x1
    # - _process_matches x1

    # schema for tracking layer after link unlink relaxed to:
    # - tracking layer exists (only for link?) 
    # - tracking layer is nx4
    # - tracking layer only has continuous tracks
