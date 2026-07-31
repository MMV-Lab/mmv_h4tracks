"""Module providing tests for the analysis widget"""

import numpy as np
import pytest

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._evaluation import (
    round_half_up,
    get_false_positives,
    get_false_negatives,
    get_split_cells,
)
from mmv_h4tracks._tests.data_loading import (
    DATA_ROOT,
    load_named_tracks,
    load_named_volumes,
)
from mmv_h4tracks._tests.fixture_helpers import (
    clear_viewer_layers,
    reset_plugin_state,
    reset_widget,
)

# this tests if the analysis returns the proper values
PATH = DATA_ROOT
SEGMENTATION_GT = "GT"


@pytest.fixture(autouse=True)
def _evaluation_tests_use_single_process(monkeypatch):
    """Keep evaluation helpers off multiprocessing.Pool in this module."""
    monkeypatch.setattr(
        "mmv_h4tracks._widget.MMVH4TRACKS.get_process_limit",
        lambda self: 1,
    )


@pytest.fixture(scope="module")
def evaluation_testdata():
    """Load evaluation arrays once per module from sibling ``.npy`` dumps."""
    segmentations = load_named_volumes(PATH / "segmentation")
    tracks = load_named_tracks(PATH / "tracks")
    return segmentations, tracks


def _ensure_evaluation_layers(widget, segmentations, tracks) -> None:
    """Add evaluation layers if missing (e.g. after get_widget cleared the viewer)."""
    viewer = widget.viewer
    expected = set(segmentations) | set(tracks)
    present = {layer.name for layer in viewer.layers}
    if expected.issubset(present):
        return

    clear_viewer_layers(viewer)
    for name, data in segmentations.items():
        viewer.add_labels(np.array(data, copy=True), name=name)
    for name, data in tracks.items():
        viewer.add_tracks(np.array(data, copy=True), name=name)


def _restore_evaluation_layer_data(widget, segmentations, tracks) -> None:
    """Reset layer.data in place from pristine module caches."""
    for name, data in segmentations.items():
        widget.viewer.layers[name].data = np.array(data, copy=True)
    for name, data in tracks.items():
        widget.viewer.layers[name].data = np.array(data, copy=True)


@pytest.fixture(scope="module")
def evaluation_widget_loaded(module_widget, evaluation_testdata):
    """Attach evaluation layers once per module."""
    segmentations, tracks = evaluation_testdata
    reset_widget(module_widget)
    _ensure_evaluation_layers(module_widget, segmentations, tracks)
    yield module_widget


@pytest.fixture
def set_widget_up(evaluation_widget_loaded, evaluation_testdata):
    """
    Soft-reset plugin state and restore evaluation layer data between tests.

    Layers stay in the viewer across evaluation tests; only mutated ``.data``
    (e.g. from ``adjust_centroids``) is restored from the module cache.
    """
    my_widget = evaluation_widget_loaded
    segmentations, tracks = evaluation_testdata

    reset_plugin_state(my_widget)
    _ensure_evaluation_layers(my_widget, segmentations, tracks)
    _restore_evaluation_layer_data(my_widget, segmentations, tracks)
    my_widget.combobox_segmentation.setCurrentIndex(
        my_widget.combobox_segmentation.findText(SEGMENTATION_GT)
    )
    yield my_widget


@pytest.fixture
def get_widget(module_widget):
    """
    Tiny synthetic labels for unit-style evaluation helpers.

    Clears the shared viewer (including module evaluation layers); the next
    ``set_widget_up`` rebuilds them via ``_ensure_evaluation_layers``.
    """
    reset_widget(module_widget)
    add_layers(module_widget.viewer)
    yield module_widget


def add_layers(viewer):
    """
    Adds sample data to the viewer

    Parameters
    ----------
    viewer : Viewer
        Napari viewer instance
    """
    gt = np.asarray([[[1, 2], [3, 0]], [[5, 0], [7, 0]], [[9, 10], [11, 12]]])
    more = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    less = np.asarray([[[0, 0], [0, 4]], [[5, 0], [0, 0]], [[0, 0], [0, 0]]])
    viewer.add_labels(less, name="less")
    viewer.add_labels(gt, name="gt")
    viewer.add_labels(more, name="more")


# split in unit & integration tests


# test if rounding works correctly
@pytest.mark.unit
@pytest.mark.parametrize("value", *[np.linspace(0, 1, 11)])
def test_round_half_up(value):
    if value < 0.5:
        assert round_half_up(value) == 0
    else:
        assert round_half_up(value) == 1


# test if iou, dice and ap50 are caluculated right for single frame, multiple frames and all frames
@pytest.mark.eval
@pytest.mark.eval_seg
@pytest.mark.unit
@pytest.mark.parametrize("score", ["iou", "dice", "ap50"])
@pytest.mark.parametrize("area", ["unchanged", "decreased", "increased"])
@pytest.mark.parametrize("frames", ["range", "all"])
def test_segmentation_evaluation(get_widget, score, area, frames):
    """
    Test if segmentation evaluation produces correct values

    Parameters
    ----------
    get_widget : MMVTracking
        Instance of the main widget
    score : str
        Name of the metric
    area : str
        Area which is used in the comparison
    frames : str
        Amount of frames to be analyzed
    """
    widget = get_widget
    viewer = widget.viewer
    window = widget.evaluation_window
    if frames == "range":
        gt = viewer.layers[1].data[0:2]
        if area == "unchanged":
            if score == "iou":
                assert window._calculate_iou(gt, gt) == 1
            elif score == "dice":
                assert window._calculate_dice(gt, gt) == 1
            elif score == "ap50":
                assert window._calculate_ap50(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data[0:2]
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 1 / 6
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 2 / 7
            elif score == "ap50":
                assert window._calculate_ap50(gt, seg) == 1 / 6
        elif area == "increased":
            seg = viewer.layers[2].data[0:2]
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.625
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 10 / 13
            elif score == "ap50":
                assert window._calculate_ap50(gt, seg) == 0.625
    elif frames == "all":
        gt = viewer.layers[1].data
        if area == "unchanged":
            if score == "iou":
                assert window._calculate_iou(gt, gt) == 1
            elif score == "dice":
                assert window._calculate_dice(gt, gt) == 1
            elif score == "ap50":
                assert window._calculate_ap50(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.1
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 2 / 11
            elif score == "ap50":
                assert window._calculate_ap50(gt, seg) == 0.1
        elif area == "increased":
            seg = viewer.layers[2].data
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.75
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 6 / 7
            elif score == "ap50":
                assert window._calculate_ap50(gt, seg) == 0.75

# False-positive / FN / split-cell metrics (serial path under test fixtures)
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value",
    [("false positive", 2), ("false positive_1", 3)],
)
def test_false_positives(set_widget_up, layername, expected_value):
    """
    Test if false positives are calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : int
        Expected fault value for false positives
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fp = window.get_segmentation_fault(gt_seg, eval_seg, get_false_positives)
    assert fp == expected_value

@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value, gt",
    [
        ("false_negative", 1, "GT"),
        ("false_negative_1", 5, "GT"),
        ("false_negative_2", 2, "GT_false_negative_2"),
    ],
)
def test_false_negatives(set_widget_up, layername, expected_value, gt):
    """
    Test if false negatives are calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : int
        Expected fault value for false negatives
    gt : str
        Name of the ground truth segmentation layer
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index(gt)].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fn = window.get_segmentation_fault(gt_seg, eval_seg, get_false_negatives)
    assert fn == expected_value

@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize("layername, expected_value", [("falsely_merged", 3)])
def test_split_cells(set_widget_up, layername, expected_value):
    """
    Test if split cells are calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : int
        Expected fault value for split cells
    """
    # test if split cells are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    sc = window.get_segmentation_fault(gt_seg, eval_seg, get_split_cells)
    assert sc == expected_value

@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value",
    [
        ("added_edge", 0),
        ("deleted_edge", 4),
        ("centroid_outside", 2),
        ("falsely_cut_tracks", 0),
        ("switch", 4),
    ],
)
def test_added_edges(set_widget_up, layername, expected_value):
    """
    Test if added edges are calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : float
        Expected fault value for added edges
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks_layer = viewer.layers[viewer.layers.index(layername)]
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    bounds = (0, gt_seg.shape[0] - 1)
    window.adjust_centroids(gt_seg, eval_tracks_layer, bounds)
    eval_tracks = eval_tracks_layer.data
    _, ae = window.get_track_fault(gt_seg, gt_tracks, eval_seg, eval_tracks)
    assert ae == expected_value

@pytest.mark.new
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value",
    [
        ("added_edge", 0),
        ("deleted_edge", 4),
        ("centroid_outside", 2),
        ("falsely_cut_tracks", 0),
        ("switch", 4),
    ],
)
def test_added_edges_changed_seg(set_widget_up, layername, expected_value):
    """
    Test if added edges are calculated correctly when segmentation is changed

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : float
        Expected fault value for added edges
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index("seg_changed")].data
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks_layer = viewer.layers[viewer.layers.index(layername)]
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    bounds = (0, gt_seg.shape[0] - 1)
    window.adjust_centroids(eval_seg, eval_tracks_layer, bounds)
    eval_tracks = eval_tracks_layer.data
    _, ae = window.get_track_fault(gt_seg, gt_tracks, eval_seg, eval_tracks)
    assert ae == expected_value

@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value",
    [
        ("deleted_edge", 0),
        ("added_edge", 5),
        ("centroid_outside", 2),
        ("falsely_cut_tracks", 0),
        ("switch", 4),
    ],
)
def test_deleted_edges_identical_segmentation(set_widget_up, layername, expected_value):
    """
    Test if deleted edges are calculated correctly when eval seg matches GT seg.

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : int
        Expected fault value for deleted edges
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks_layer = viewer.layers[viewer.layers.index(layername)]
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    bounds = (0, gt_seg.shape[0] - 1)
    window.adjust_centroids(gt_seg, eval_tracks_layer, bounds)
    eval_tracks = eval_tracks_layer.data
    de, _ = window.get_track_fault(gt_seg, gt_tracks, eval_seg, eval_tracks)
    assert de == expected_value

@pytest.mark.new
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value",
    [
        ("deleted_edge", 0),
        ("added_edge", 5),
        ("centroid_outside", 2),
        ("falsely_cut_tracks", 0),
        ("switch", 4),
    ],
)
def test_deleted_edges(set_widget_up, layername, expected_value):
    """
    Test if deleted edges are calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername : str
        Name of the label layer to evaluate
    expected_vale : int
        Expected fault value for deleted edges
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index("seg_changed")].data
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks_layer = viewer.layers[viewer.layers.index(layername)]
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    bounds = (0, gt_seg.shape[0] - 1)
    window.adjust_centroids(gt_seg, eval_tracks_layer, bounds)
    eval_tracks = eval_tracks_layer.data
    de, _ = window.get_track_fault(gt_seg, gt_tracks, eval_seg, eval_tracks)
    assert de == expected_value


@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.integration
@pytest.mark.parametrize(
    "layername_seg, layername_tracks, expected_value",
    [("false positive", "added_edge", 7)],
)
def test_fault_value(set_widget_up, layername_seg, layername_tracks, expected_value):
    """
    Test if fault value for tracking evaluation is calculated correctly.

    Calls ``evaluate_curated_tracking`` synchronously (same work as the
    worker path, without racing the UI thread).
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    eval_seg = viewer.layers[layername_seg].data
    eval_tracks = viewer.layers[layername_tracks].data
    gt_seg = viewer.layers["GT"].data
    gt_tracks_layer = viewer.layers["GT_tracks"]
    widget.combobox_segmentation.setCurrentIndex(
        widget.combobox_segmentation.findText("GT")
    )
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText("GT_tracks"))

    window.evaluate_curated_tracking(gt_tracks_layer, gt_seg, eval_tracks, eval_seg)

    fault_value = float(window.tracking_table.item(4, 1).text())
    assert fault_value == expected_value
