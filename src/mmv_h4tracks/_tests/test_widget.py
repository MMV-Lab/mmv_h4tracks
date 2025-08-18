"""Module providing tests for the main widget"""

import numpy as np
import pytest
import time

from mmv_h4tracks import MMVH4TRACKS


AMOUNT_OF_COMBOBOXES = 3

# make_napari_viewer is a pytest fixture that returns a napari viewer object


@pytest.fixture
def create_widget(make_napari_viewer):
    yield MMVH4TRACKS(make_napari_viewer())


@pytest.fixture
def viewer_with_widget(create_widget):
    """
    Creates a viewer with the plugin and 3 label layers

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
    add_layers(viewer, 3)
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
            np.random.randint(256, size=(1, 10, 10), dtype=int), name=names[i]
        )
    for i in range(0, amount):
        viewer.add_labels(
            np.random.randint(5, size=(1, 10, 10), dtype=int), name=names[amount + i]
        )
    for i in range(0, amount):
        viewer.add_tracks(
            np.random.randint(5, size=(6, 4), dtype=int), name=names[amount * 2 + i]
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
    Test if adding a layer adds it to the combobox
    Amount of elements in the combobox should increase by one

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
    widget.viewer.layers.pop(index * 3)
    assert combobox.count() == old_length - 1


@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize("index", range(AMOUNT_OF_COMBOBOXES))
@pytest.mark.parametrize("removal_index", [0, 1, 2])
@pytest.mark.parametrize("selected_index", [1, 2])
def test_combobox_index_on_layer_remove(viewer_with_widget, index, removal_index, selected_index):
    """
    Test if the correct layer is selected after removing a layer
    Index should decrease by one if index of selected layer decreases (case 1)
    Index should remain unchanged if index of selected layer is unchanged (case 2)
    Index should remain unchanged if selected layer is removed (case 3), UNLESS:
    Index should decrease by one if last layer is selected and removed (case 4)

    Parameters
    ----------
    viewer_with_widget : MMVTracking
        Instance of the main widget
    index : int
        index of the combobox to test
    """
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(selected_index)
    widget.viewer.layers.pop(removal_index + index * 3)
    if removal_index < selected_index:
        # case 1
        assert combobox.currentIndex() == selected_index - 1, f"Expected {selected_index - 1}, got {combobox.currentIndex()}"
    elif removal_index == selected_index:
        if selected_index == 2:
            # case 4
            assert combobox.currentIndex() == 1, f"Expected 1, got {combobox.currentIndex()}"
        else:
            # case 3
            assert combobox.currentIndex() == selected_index, f"Expected {selected_index}, got {combobox.currentIndex()}"
    else:
        # case 2
        assert combobox.currentIndex() == selected_index, f"Expected {selected_index}, got {combobox.currentIndex()}"


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
@pytest.mark.parametrize("from_index, to_index", [(0, 2), (2, 0), (2, 1)])
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
    combobox.setCurrentIndex(1)
    layername = combobox.currentText()

    widget.viewer.layers.move(from_index + index * 3, to_index + index * 3)
    assert widget.viewer.layers.index(layername) - index * 3 == combobox.findText(
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
    combobox.setCurrentIndex(1)
    widget.viewer.layers.move(index * 3 + 1, index * 3)
    assert combobox.currentIndex() == 0


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
    combobox.setCurrentIndex(2)
    widget.viewer.layers.move(1, 0)
    assert combobox.currentIndex() == 2


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
    widget.viewer.layers[index * 3 + 1].name = "New"
    assert combobox.findText("New") == 2
