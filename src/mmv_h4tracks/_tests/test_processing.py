"""Module providing tests for the processing module."""
import numpy as np
import pytest
from pathlib import Path

from mmv_h4tracks import _processing as processing, MMVH4TRACKS
from mmv_h4tracks._segmentation import SegmentationWindow

@pytest.fixture
def create_widget(make_napari_viewer):
    return MMVH4TRACKS(make_napari_viewer())

pytestmark = pytest.mark.processing

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