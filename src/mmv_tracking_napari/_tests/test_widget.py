from mmv_tracking_napari import MMVTracking
import numpy as np

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_segmentation_evaluation(make_napari_viewer, capsys):
    # make viewer and add a label layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_labels(np.random.randint(2, size = (1, 100, 100), dtype = int), name = "Segmentation Data")

    # create our widget, passing in the viewer
    my_widget = MMVTracking(viewer)

    # call our widget method
    my_widget._store_segmentation()
    my_widget._evaluate_segmentation()

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "IoU score for frame 0: 1.0\nDICE score for frame 0: 1.0\nF1 score for frame 0: 1.0\nIoU score for whole movie: 1.0\nDICE score for whole movie: 1.0\nF1 score for whole movie: 1.0\n"