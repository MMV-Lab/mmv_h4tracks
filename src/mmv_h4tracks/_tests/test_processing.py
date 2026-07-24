"""Module providing tests for the processing module."""

import numpy as np
import pytest
from pathlib import Path

import napari

from mmv_h4tracks import _processing as processing
from mmv_h4tracks._segmentation import SegmentationWindow


pytestmark = pytest.mark.processing


@pytest.fixture
def widget_with_segmentation(create_widget):
    """Tiny synthetic labels stack for fast proximity-tracking schema checks."""
    widget = create_widget
    # 5 frames, two stationary blobs well within MAX_MATCHING_DIST.
    seg = np.zeros((5, 40, 40), dtype=np.int32)
    for z in range(seg.shape[0]):
        seg[z, 8:12, 8:12] = 1
        seg[z, 25:29, 25:29] = 2
    widget.viewer.add_labels(seg, name="segmentation")
    widget.combobox_segmentation.setCurrentText("segmentation")
    return widget


@pytest.mark.format
@pytest.mark.unit
def test_segment_slice_cpu():
    layer_slice = np.zeros((100, 100), dtype=np.int8)
    parameters = {
        "model_path": str(
            Path(__file__).parent.parent.absolute()
            / "models"
            / "Neutrophil granulocytes"
        ),
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
def test_read_custom_model_dict(tmp_path):
    from mmv_h4tracks._custom_models import CustomModelStore, set_custom_model_store

    set_custom_model_store(CustomModelStore(tmp_path / "empty_store"))
    try:
        model_dict = processing.read_custom_model_dict()
        assert model_dict == {}
    finally:
        set_custom_model_store(None)


@pytest.mark.unit
def test_get_parameters_hardcoded(create_widget):
    widget = SegmentationWindow(create_widget)
    params = processing._get_parameters(widget, "Neutrophil_granulocytes")
    assert params["diameter"] == 15
    assert params["model_path"].endswith("Neutrophil_granulocytes")


@pytest.mark.unit
def test_get_parameters_custom(create_widget):
    widget = SegmentationWindow(create_widget)
    widget.custom_models = {
        "demo": {
            "filename": "demo_weights",
            "params": {"diameter": 20, "flow_threshold": 0.3, "cellprob_threshold": 0},
        }
    }
    params = processing._get_parameters(widget, "custom_demo")
    assert params["diameter"] == 20
    assert params["model_path"].endswith("demo_weights")
    # Stored dict must not be mutated by path injection.
    assert "model_path" not in widget.custom_models["demo"]["params"]


@pytest.mark.unit
def test_get_parameters_unknown_raises(create_widget):
    widget = SegmentationWindow(create_widget)
    with pytest.raises(ValueError, match="Unknown model"):
        processing._get_parameters(widget, "not_a_real_model")


@pytest.mark.unit
def test_read_models(create_widget):
    widget = create_widget
    segmentation_widget = SegmentationWindow(widget)
    hardcoded_models, _custom_models = processing.read_models(segmentation_widget)

    assert "Neutrophil_granulocytes" in hardcoded_models


@pytest.mark.unit
def test_display_models(create_widget):
    widget = create_widget
    segmentation_widget = SegmentationWindow(widget)
    hardcoded_models, custom_models = processing.read_models(segmentation_widget)
    assert "Neutrophil_granulocytes" in hardcoded_models
    processing.display_models(segmentation_widget, hardcoded_models, custom_models)
    combo = segmentation_widget.combobox_cellpose_model
    assert combo.findText("Neutrophil_granulocytes") >= 0
    assert combo.count() == len(hardcoded_models) + len(custom_models)


@pytest.mark.integration
@pytest.mark.schema
def test_track_segmentation_schema(widget_with_segmentation, qtbot, monkeypatch):
    """Schema-check proximity tracking on a tiny synthetic stack."""
    # Force serial map/starmap (no Pool spawn) via the shared concurrency policy.
    monkeypatch.setattr(
        widget_with_segmentation,
        "get_process_limit",
        lambda: 1,
    )

    widget = widget_with_segmentation
    tracking_widget = widget.tracking_window
    viewer = widget.viewer
    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "segmentation"
    assert isinstance(layer, napari.layers.Labels)

    worker = processing._track_segmentation(tracking_widget)
    trk_data = None

    def capture_result(result):
        nonlocal trk_data
        trk_data = result

    worker.returned.connect(capture_result)
    with qtbot.waitSignal(worker.returned, timeout=30000):
        pass

    assert trk_data is not None
    assert trk_data.shape[1] == 4
    unique_ids, indices = np.unique(trk_data[:, 0], return_index=True)
    ordered_unique_ids = unique_ids[np.argsort(indices)]
    assert np.array_equal(unique_ids, ordered_unique_ids)
    assert unique_ids[-1] - unique_ids[0] + 1 == len(unique_ids)
    lower_bound_frame = -1
    for trk_id in ordered_unique_ids:
        trk = trk_data[trk_data[:, 0] == trk_id]
        frames = trk[:, 1]
        low_frame = frames[0]
        assert all(frames >= low_frame)
        assert low_frame >= lower_bound_frame
        lower_bound_frame = low_frame
        high_frame = frames[-1]
        assert all(frames <= high_frame)
        assert len(set(frames)) == len(frames)
        assert len(frames) == high_frame - low_frame + 1
