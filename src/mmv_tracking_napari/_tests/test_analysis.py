from mmv_tracking_napari import MMVTracking
import numpy as np
import pytest
import tifffile
import os

# this tests if the analysis returns the proper values
PATH = f"{os.path.dirname(__file__)}/data"

@pytest.fixture
def set_widget_up(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = MMVTracking(viewer)
    """im = tifffile.imread(f"{PATH}/images/Raw Image.tif")
    viewer.add_image(im, name = "Raw Image")"""
    for file in os.listdir(f"{PATH}/segmentation"):
        im = tifffile.imread(PATH + "/segmentation/" + file)
        image = np.array(im).astype(int)
        name = os.path.basename(file)
        viewer.add_labels(
            image,
            name = name 
        )
    for file in os.listdir(f"{PATH}/tracks"):
        tracks = np.load(PATH + "/tracks/" + file)
        name = os.path.basename(file)
        viewer.add_tracks(
            tracks,
            name = name
        )
    yield my_widget

@pytest.fixture
def get_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = MMVTracking(viewer)
    add_layers(viewer)
    yield my_widget

def add_layers(viewer):
    gt = np.asarray([[[1,2],[3,0]],[[5,0],[7,0]],[[9,10],[11,12]]])
    more = np.asarray([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
    less = np.asarray([[[0,0],[0,4]],[[5,0],[0,0]],[[0,0],[0,0]]])
    viewer.add_labels(
        less,
        name = "less"
    )
    viewer.add_labels(
        gt,
        name = "gt"
    )
    viewer.add_labels(
        more,
        name = "more"
    )

# split in unit & integration tests

# test if iou, dice and f1 are caluculated right for single frame, multiple frames and all frames
@pytest.mark.eval
@pytest.mark.eval_seg
@pytest.mark.unit
@pytest.mark.parametrize(
    "score", ["iou", "dice", "f1"]
)
@pytest.mark.parametrize(
    "area", ["unchanged", "decreased", "increased"]
)
@pytest.mark.parametrize(
    "frames", ["single", "range", "all"]
)
def test_segmentation_evaluation(get_widget, score, area, frames):
    # 
    widget = get_widget
    viewer = widget.viewer
    widget._analysis(hide = True)
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
                assert window.get_iou(gt, seg) == .5
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2/3
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2/3
        elif area == "increased":
            seg = viewer.layers[2].data[1]
            if score == "iou":
                assert window.get_iou(gt, seg) == .5
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2/3
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2/3
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
                assert window.get_iou(gt, seg) == 1/6
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2/7
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2/7
        elif area == "increased":
            seg = viewer.layers[2].data[0:2]
            if score == "iou":
                assert window.get_iou(gt, seg) == .625
            elif score == "dice":
                assert window.get_dice(gt, seg) == 10/13
            elif score == "f1":
                assert window.get_f1(gt, seg) == 10/13
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
                assert window.get_iou(gt, seg) == .1
            elif score == "dice":
                assert window.get_dice(gt, seg) == 2/11
            elif score == "f1":
                assert window.get_f1(gt, seg) == 2/11
        elif area == "increased":
            print("in increased")
            seg = viewer.layers[2].data
            if score == "iou":
                assert window.get_iou(gt, seg) == .75
            elif score == "dice":
                assert window.get_dice(gt, seg) == 6/7
            elif score == "f1":
                assert window.get_f1(gt, seg) == 6/7
    
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.new
@pytest.mark.parametrize(
    "layername, expected_value", [("false positive.tif", 2), ("false positive_1.tif", 3)]
)
def test_false_positives(set_widget_up, layername, expected_value):
    # test if false positives are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fp = window.get_false_positives(gt_seg, eval_seg)
    assert fp == expected_value
    
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.new
@pytest.mark.parametrize(
    "layername, expected_value", [("false_negative.tif", 1), ("false_negative_1.tif", 5)]
)
def test_false_negatives(set_widget_up, layername, expected_value):
    # test if false negatives are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    fn = window.get_false_negatives(gt_seg, eval_seg)
    assert fn == expected_value
    
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value", [("falsely_merged.tif", 1)]
)
def test_split_cells(set_widget_up, layername, expected_value):
    # test if split cells are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = viewer.layers[viewer.layers.index(layername)].data
    sc = window.get_split_cells(gt_seg, eval_seg)
    assert sc == expected_value
    
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value", [("deleted_edge.npy", 2)]
)
def test_added_edges(set_widget_up, layername, expected_value):
    # test if added edges are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks.npy")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername)].data
    ae = window.get_added_edges(gt_seg, eval_seg, gt_tracks, eval_tracks)
    assert ae == expected_value
    
@pytest.mark.eval
@pytest.mark.eval_tracking
@pytest.mark.unit
@pytest.mark.parametrize(
    "layername, expected_value", [("added_edge.npy", 1)]
)
def test_deleted_edges(set_widget_up, layername, expected_value):
    # test if deleted edges are calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = gt_seg
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks.npy")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername)].data
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
    "layername_seg, layername_tracks, expected_value", [("false positive.tif", "added_edge.npy", 3)]
)
def test_fault_value(set_widget_up, layername_seg, layername_tracks, expected_value):
    # test if tracking error is calculated correctly
    widget = set_widget_up
    viewer = widget.viewer
    widget._analysis(hide = True)
    window = widget.analysis_window
    gt_seg = viewer.layers[viewer.layers.index("GT.tif")].data
    eval_seg = viewer.layers[viewer.layers.index(layername_seg)].data
    gt_tracks = viewer.layers[viewer.layers.index("GT_tracks.npy")].data
    eval_tracks = viewer.layers[viewer.layers.index(layername_tracks)].data
    fault_value = window.evaluate_tracking(gt_seg, eval_seg, gt_tracks, eval_tracks)
    assert fault_value == expected_value

# do by removing part of cell/ adding more to cell

# test if centroids are adjusted correctly
