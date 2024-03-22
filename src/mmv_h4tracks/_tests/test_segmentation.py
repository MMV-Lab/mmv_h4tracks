"""Module providing tests for the segmentation widget"""

import pytest
from pathlib import Path
from aicsimageio import AICSImage
import numpy as np

from mmv_h4tracks import MMVH4TRACKS

PATH = Path(__file__).parent / "data"

@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())

@pytest.fixture
def viewer_with_data(create_widget):
    widget = create_widget
    viewer = widget.viewer
    for file in list(Path(PATH / "images").iterdir()):
        image = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_image(image, name=name)
    for file in list(Path(PATH / "segmentation").iterdir()):
        image = AICSImage(file).get_image_data("ZYX")
        name = file.stem
        viewer.add_labels(image, name=name)
    for file in list(Path(PATH / "tracks").iterdir()):
        tracks = np.load(file)
        name = file.stem
        viewer.add_tracks(tracks, name=name)
    yield widget

@pytest.mark.integration
@pytest.mark.parametrize("position", [(0, 0, 0), (0, 65, 72), (1, 63, 72)])
def test_remove_cell_from_tracks(viewer_with_data, position):
    widget = viewer_with_data
    try:
        widget.segmentation_window.remove_cell_from_tracks(position)
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")