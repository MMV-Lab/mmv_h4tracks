"""Module providing tests for the analysis widget"""
# import os
from pathlib import Path

import numpy as np
import pytest

# import tifffile
from aicsimageio import (
    AICSImage,
)  # !! imports hinzufügen ist okay, aber dann müssen die auch in die

# !! requirements
from mmv_tracking_napari import MMVTracking

# this tests if the analysis returns the proper values
PATH = Path(__file__).parent / "data"
# PATH = f"{os.path.dirname(__file__)}/data"


@pytest.fixture
def set_widget_up(make_napari_viewer):
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
    viewer = make_napari_viewer()
    my_widget = MMVTracking(viewer)
    for file in list(Path(PATH / "segmentation").iterdir()):
        print(file.stem)
        segmentation = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_labels(segmentation, name=name)
    """for file in os.listdir(f"{PATH}/segmentation"):
        segmentation = AICSImage(f"{PATH}/segmentation/{file}").get_image_data("ZYX")
        name = Path(file).stem
        #name = os.path.basename(file)               # ?? lass und pathlib verwenden
        viewer.add_labels(segmentation, name=name)"""
    for file in list(Path(PATH / "tracks").iterdir()):
        print(file.stem)
        tracks = np.load(file)
        name = file.stem
        viewer.add_tracks(tracks, name=name)
    """for file in os.listdir(f"{PATH}/tracks"):
        tracks = np.load(PATH + "/tracks/" + file)
        name = Path(file).stem
        #name = os.path.basename(file)
        viewer.add_tracks(tracks, name=name)"""
    my_widget.combobox_segmentation.setCurrentIndex(
        my_widget.combobox_segmentation.findText(SEGMENTATION_GT)
    )
    yield my_widget


@pytest.fixture
def get_widget(make_napari_viewer):
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
    viewer = make_napari_viewer()
    my_widget = MMVTracking(viewer)
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


# test if iou, dice and f1 are caluculated right for single frame, multiple frames and all frames
@pytest.mark.eval
@pytest.mark.eval_seg
@pytest.mark.unit
@pytest.mark.parametrize("score", ["iou", "dice", "f1"])
@pytest.mark.parametrize("area", ["unchanged", "decreased", "increased"])
@pytest.mark.parametrize("frames", ["single", "range", "all"])
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
    # ?? können wir den Test hier übersichtlicher gestalten?
    # !! vielleicht, aber ich habe spontan keine schlaue idee wie, ohne die parametrisierung weg zu schmeißen
    widget = get_widget
    viewer = widget.viewer
    widget._analysis(hide=True)
    window = widget.analysis_window
    if frames == "single":
        gt = viewer.layers[1].data[1]
        if area == "unchanged":
            if score == "iou":
                assert window.get_iou(gt, gt) == 1
            elif score == "dice":
                assert window.get_dice(gt, gt) == 1
            elif score == "f1":
                assert window.get_f1(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data[1]
            if score == "iou":
                assert window.get_iou(gt, seg) == 0.5
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2 / 3
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2 / 3
        elif area == "increased":
            seg = viewer.layers[2].data[1]
            if score == "iou":
                assert window.get_iou(gt, seg) == 0.5
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2 / 3
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2 / 3
    elif frames == "range":
        gt = viewer.layers[1].data[0:2]
        if area == "unchanged":
            if score == "iou":
                assert window.get_iou(gt, gt) == 1
            elif score == "dice":
                assert window.get_dice(gt, gt) == 1
            elif score == "f1":
                assert window.get_f1(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data[0:2]
            if score == "iou":
                assert window.get_iou(gt, seg) == 1 / 6
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2 / 7
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2 / 7
        elif area == "increased":
            seg = viewer.layers[2].data[0:2]
            if score == "iou":
                assert window.get_iou(gt, seg) == 0.625
            elif score == "dice":
                assert window.get_dice(gt, seg) == 10 / 13
            elif score == "f1":
                assert window.get_f1(gt, seg) == 10 / 13
    elif frames == "all":
        print("in all")
        gt = viewer.layers[1].data
        if area == "unchanged":
            if score == "iou":
                assert window.get_iou(gt, gt) == 1
            elif score == "dice":
                assert window.get_dice(gt, gt) == 1
            elif score == "f1":
                assert window.get_f1(gt, gt) == 1
        elif area == "decreased":
            seg = viewer.layers[0].data
            if score == "iou":
                assert window.get_iou(gt, seg) == 0.1
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2 / 11
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2 / 11
        elif area == "increased":
            print("in increased")
            seg = viewer.layers[2].data
            if score == "iou":
                assert window.get_iou(gt, seg) == 0.75
            elif score == "dice":
                assert window.get_dice(gt, seg) == 6 / 7
            elif score == "f1":
                assert window.get_f1(gt, seg) == 6 / 7


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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fp = window.get_false_positives(gt_seg, eval_seg)
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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index(gt)].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fn = window.get_false_negatives(gt_seg, eval_seg)
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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    sc = window.get_split_cells(gt_seg, eval_seg)
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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername)].data
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    ae = window.get_added_edges(gt_seg, eval_seg, gt_tracks, eval_tracks)
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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername)].data
    widget.combobox_tracks.setCurrentIndex(widget.combobox_tracks.findText(layername))
    de = window.get_removed_edges(gt_seg, eval_seg, gt_tracks, eval_tracks)
    assert de == expected_value


# test if tracking evaluation is calculated right
# -> false positives
# -> false negatives
# -> split cells
# -> added edges
# -> removed edges


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
    widget._analysis(hide=True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT")].data
    eval_seg = viewer.layers[viewer.layers.index(layername_seg)].data
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername_tracks)].data
    widget.combobox_tracks.setCurrentIndex(
        widget.combobox_tracks.findText(layername_tracks)
    )
    fault_value = window.evaluate_tracking(gt_seg, eval_seg, gt_tracks, eval_tracks)
    assert fault_value == expected_value
