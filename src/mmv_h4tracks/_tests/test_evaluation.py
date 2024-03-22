"""Module providing tests for the analysis widget"""
from pathlib import Path

import numpy as np
import pytest
from aicsimageio import (
    AICSImage,
)

from mmv_h4tracks import MMVH4TRACKS

# this tests if the analysis returns the proper values
PATH = Path(__file__).parent / "data"

@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())


@pytest.fixture
def set_widget_up(create_widget):
    """
    Creates an instance of the plugin and adds all layers of testdata to the viewer

    Parameters
    ----------
    make_napari_viewer : fixture
        Pytest fixture that creates a napari viewer

    Yields
    ------
    my_widget
        Instance of the main widget
    """
    SEGMENTATION_GT = "GT"
    my_widget = create_widget
    viewer = my_widget.viewer
    for file in list(Path(PATH / "segmentation").iterdir()):
        print(file.stem)
        segmentation = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_labels(segmentation, name=name)
    for file in list(Path(PATH / "tracks").iterdir()):
        print(file.stem)
        tracks = np.load(file)
        name = file.stem
        viewer.add_tracks(tracks, name=name)
    my_widget.combobox_segmentation.setCurrentIndex(
        my_widget.combobox_segmentation.findText(SEGMENTATION_GT)
    )
    yield my_widget


@pytest.fixture
def get_widget(create_widget):
    """
    Creates an instance of the plugin and adds sample layers

    Parameters
    ----------
    make_napari_viewer : fixture
        Pytest fixture that creates a napari viewer

    Yields
    ------
    my_widget
        Instance of the main widget
    """
    my_widget = create_widget
    viewer = my_widget.viewer
    add_layers(viewer)
    yield my_widget


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
def test_round_half_up(set_widget_up, value):
    from mmv_h4tracks._evaluation import round_half_up

    if value < 0.5:
        assert round_half_up(value) == 0
    else:
        assert round_half_up(value) == 1


# test if iou, dice and f1 are caluculated right for single frame, multiple frames and all frames
@pytest.mark.eval
@pytest.mark.eval_seg
@pytest.mark.unit
@pytest.mark.parametrize("score", ["iou", "dice", "f1"])
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
            elif score == "f1":
                assert window._calculate_f1(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data[0:2]
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 1 / 6
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 2 / 7
            elif score == "f1":
                assert window._calculate_f1(gt, seg) == 2 / 7
        elif area == "increased":
            seg = viewer.layers[2].data[0:2]
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.625
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 10 / 13
            elif score == "f1":
                assert window._calculate_f1(gt, seg) == 10 / 13
    elif frames == "all":
        print("in all")
        gt = viewer.layers[1].data
        if area == "unchanged":
            if score == "iou":
                assert window._calculate_iou(gt, gt) == 1
            elif score == "dice":
                assert window._calculate_dice(gt, gt) == 1
            elif score == "f1":
                assert window._calculate_f1(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.1
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 2 / 11
            elif score == "f1":
                assert window._calculate_f1(gt, seg) == 2 / 11
        elif area == "increased":
            print("in increased")
            seg = viewer.layers[2].data
            if score == "iou":
                assert window._calculate_iou(gt, seg) == 0.75
            elif score == "dice":
                assert window._calculate_dice(gt, seg) == 6 / 7
            elif score == "f1":
                assert window._calculate_f1(gt, seg) == 6 / 7


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
    from mmv_h4tracks._evaluation import get_false_positives as func

    fp = window.get_segmentation_fault(gt_seg, eval_seg, func)
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
    from mmv_h4tracks._evaluation import get_false_negatives as func

    fn = window.get_segmentation_fault(gt_seg, eval_seg, func)
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
    from mmv_h4tracks._evaluation import get_split_cells as func

    sc = window.get_segmentation_fault(gt_seg, eval_seg, func)
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
    bounds = (0, gt_seg.shape[0])
    window.adjust_centroids(gt_seg, eval_tracks_layer, bounds)
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
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks_layer = viewer.layers[viewer.layers.index(layername)]
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    bounds = (0, gt_seg.shape[0])
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
    Test if fault value for tracking evaluation is calculated correctly

    Parameters
    ----------
    set_widget_up : MMVTracking
        Instance of the main widget
    layername_seg : str
        Name of the label layer to evaluate
    layername_tracks : str
        Name of the tracks layer to evaluate
    expected_vale : float
        Expected fault value for tracking evaluation
    """
    widget = set_widget_up
    viewer = widget.viewer
    window = widget.evaluation_window
    eval_seg = viewer.layers[viewer.layers.index(layername_seg)].data
    eval_tracks = viewer.layers[viewer.layers.index(layername_tracks)].data
    widget.initial_layers = [eval_seg, eval_tracks]
    widget.combobox_segmentation.setCurrentIndex(
        widget.combobox_segmentation.findText("GT")
    )
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText("GT_tracks"))
    window.evaluate_tracking()
    fault_value = float(
        window.tracking_results.layout().itemAt(1).widget().item(4, 1).text()
    )
    assert fault_value == expected_value
