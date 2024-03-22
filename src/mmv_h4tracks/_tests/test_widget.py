"""Module providing tests for the main widget"""
import numpy as np
import pytest

from mmv_h4tracks import MMVH4TRACKS


AMOUNT_OF_COMBOBOXES = 3

# make_napari_viewer is a pytest fixture that returns a napari viewer object

@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())

@pytest.fixture
def viewer_with_widget(create_widget):
    """
    Creates a viewer with the plugin and 10 label layers

    Parameters
    ----------
    create_widget : MMVH4TRACKS
        Instance of the main widget

    Yields
    ------
    my_widget
        Instance of the main widget
    """
    my_widget = create_widget
    viewer = my_widget.viewer
    add_layers(viewer, 10)
    yield my_widget


def add_layers(viewer, amount, names=None):
    """
    Adds an amount of randomly generated label layers to the passed viewer

    Parameters
    ----------
    viewer : Viewer
        Napari viewer instance
    amount : int
        Amount of layers to be added
    names : list
        Names of the added layers
    """
    if names is None:
        names = range(1, 3 * amount + 1)
    for i in range(amount):
        viewer.add_image(
            np.random.randint(256, size=(1, 100, 100), dtype=int), name=names[i]
        )
    for i in range(0, amount):
        viewer.add_labels(
            np.random.randint(5, size=(1, 100, 100), dtype=int), name=names[amount + i]
        )
    for i in range(0, amount):
        viewer.add_tracks(
            np.random.randint(5, size=(20, 4), dtype=int), name=names[amount * 2 + i]
        )

@pytest.mark.widget
@pytest.mark.system
def test_widget_creation(create_widget):
    """
    Test if the widget is created correctly
    Widget should be an instance of MMVH4TRACKS

    Parameters
    ----------
    create_widget : MMVH4TRACKS
        Instance of the main widget
    """
    assert isinstance(create_widget, MMVH4TRACKS)

@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_combobox_add_layer(viewer_with_widget, index):
    """
    Test if an added layer is put into the correct place in the comboboxes
    Entry should be added at the bottom

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]

    names = ["IMAGE", "LABELS", "TRACKS"]
    add_layers(widget.viewer, 1, names)
    assert combobox.findText(names[index]) == combobox.count() - 1


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_index_consistent_on_layer_add(viewer_with_widget, index):
    """
    Test if adding a layer keeps the correct layer selected in a combobox
    Selected layer should remain unchanged

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_index = combobox.currentIndex()
    add_layers(widget.viewer, 1)
    assert combobox.currentIndex() == old_index


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_combobox_length_add_layer(viewer_with_widget, index):
    """
    Test if removing a layer removes it from the combobox
    Amount of elements in the combobox should decrease by one

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    add_layers(widget.viewer, 1)
    assert combobox.count() == old_length + 1


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_combobox_length_delete_layer(viewer_with_widget, index):
    """
    Test if removing a layer removes it's entry from the combobox

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    widget.viewer.layers.pop(index * 10)
    assert combobox.count() == old_length - 1


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
@pytest.mark.parametrize("removal_index", [3, 4, 5])
def test_index_consistent_on_layer_remove(viewer_with_widget, index, removal_index):
    """
    Test if the correct layer is selected after removing a layer
    Index should decrease by one if index of selected layer decreases
    Index should remain unchanged if index of selected layer is unchanged
    Index should remain unchanged if selected layer is removed, UNLESS:
    Index should decrease by one if last layer is selected and removed

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    widget.viewer.layers.pop(removal_index + 1 + index * 10)
    if removal_index < 4:
        assert combobox.currentIndex() == 4
    else:
        assert combobox.currentIndex() == 5


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_moved_layer_length(viewer_with_widget, index):
    """
    Test if moving a layer leaves amount of entries unchanged
    Moving layers should never change the amount of entries in the combobox

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    widget.viewer.layers.move(1, 0)
    assert combobox.count() == old_length


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
@pytest.mark.parametrize("from_index, to_index", [(4, 2), (6, 4), (5, 3)])
def test_moved_layer_order(viewer_with_widget, index, from_index, to_index):
    """
    Test if moving a layer updates the order of entries correctly
    Index of layer in layerlist and entry in combobox should match (with offset)

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    layername = combobox.currentText()

    widget.viewer.layers.move(from_index + index * 10, to_index + index * 10)
    assert widget.viewer.layers.index(layername) - index * 10 == combobox.findText(
        layername
    )


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_moved_layer_index_moved(viewer_with_widget, index):
    """
    Test if moving the selected layer updates the entry of the combobox correctly
    Current index should be updated to match the new index of the moved layer (with offset)

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    # test if moving a layer updates the index of the comboboxes correctly
    # current index should be updated to new index of moved item
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(4)
    widget.viewer.layers.move(index * 10 + 4, index * 10 + 2)
    assert combobox.currentIndex() == 2


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_moved_layer_index_not_moved(viewer_with_widget, index):
    """
    Test if the current index of the combobox is unchanged if a layer is moved, but the
    selected layer is not moved
    Index should remain unchanged

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    widget.viewer.layers.move(1, 0)
    assert combobox.currentIndex() == 5


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
def test_renamed_layer(viewer_with_widget, index):
    """
    Test if renaming a layer updates the entry in the combobox correctly
    Entry in the combobox should be changed to the new name of the layer

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    widget.viewer.layers[index * 10 + 4].name = "New"
    assert combobox.findText("New") == 5
