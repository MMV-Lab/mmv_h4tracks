from mmv_tracking_napari import MMVTracking
import numpy as np
import pytest


AMOUNT_OF_COMBOBOXES = 3

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams

@pytest.fixture
def viewer_with_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    my_widget = MMVTracking(viewer)
    add_layers(viewer, 10)
    yield my_widget
    
def add_layers(viewer, amount, names = None):
    if names is None:
        names = range(amount+1)
    for i in range(0, amount):
        viewer.add_labels(
            np.random.randint(2, size = (1,100,100), dtype = int),
            name = names[i]
        )

@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_combobox_add_layer(viewer_with_widget, index):
    # test if an added layer is put into the correct place in all comboboxes
    # layer should be added at the bottom
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    add_layers(widget.viewer, 1, ["New"])
    assert combobox.findText("New") == combobox.count() - 1
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_index_consistent_on_layer_add(viewer_with_widget, index):
    # test if adding a layer keeps the selected layer right
    # index should stay the same
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_index = combobox.currentIndex()
    add_layers(widget.viewer, 1)
    assert combobox.currentIndex() == old_index
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_combobox_length_add_layer(viewer_with_widget, index):
    # test if a removed layer is removed from all comboboxes properly
    # amount of items should be higher by one
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    add_layers(widget.viewer, 1)
    assert combobox.count() == old_length + 1    
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_combobox_length_delete_layer(viewer_with_widget, index):
    # test if a removed layer is removed from all comboboxes properly
    # amount of items should be lower by one
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    widget.viewer.layers.pop(0)
    assert combobox.count() == old_length - 1
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
@pytest.mark.parametrize(
    "removal_index", [3,4,5]
)
def test_index_consistent_on_layer_remove(viewer_with_widget, index, removal_index):
    # test if removing a layer keeps the selected layer right
    # index should be reduced by one if layer lower than selected is removed
    # index should should be unchanged if selected layer is removed (if not the top layer)
    # index should be unchanged if layer higher than selected one is removed
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    widget.viewer.layers.pop(removal_index)
    if removal_index < 4:
        assert combobox.currentIndex() == 4
    else:
        assert combobox.currentIndex() == 5
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_moved_layer_length(viewer_with_widget, index):
    # test if moving a layer doesn't change the amount of entries in the comboboxes
    # amount should always stay the same
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    old_length = combobox.count()
    widget.viewer.layers.move(1,0)
    assert combobox.count() == old_length
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
@pytest.mark.parametrize(
    "from_index, to_index", [(4,2),(6,4),(5,3)]
)
def test_moved_layer_order(viewer_with_widget, index, from_index, to_index):
    # test if moving a layer updates the order of the comboboxes correctly
    # index should be updated if selected layer is moved
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    layername = combobox.currentText()
    widget.viewer.layers.move(
        from_index,
        to_index
    )
    assert widget.viewer.layers.index(layername) + 1 == combobox.findText(layername)
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_moved_layer_index_moved(viewer_with_widget, index):
    # test if moving a layer updates the index of the comboboxes correctly
    # current index should be updated to new index of moved item
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    widget.viewer.layers.move(4,2)
    assert combobox.currentIndex() == 3
    
@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_moved_layer_index_not_moved(viewer_with_widget, index):
    # test if moving a layer keeps the index of the comboboxes correctly
    # index should remain unchanged if layer is not moved
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    combobox.setCurrentIndex(5)
    widget.viewer.layers.move(1,0)
    assert combobox.currentIndex() == 5

@pytest.mark.combobox
@pytest.mark.unit
@pytest.mark.parametrize(
    "index", range(AMOUNT_OF_COMBOBOXES)
)
def test_renamed_layer(viewer_with_widget, index):
    # test if renaming a layer updates the entry in the combobox
    # name at the changed index should be updated
    widget = viewer_with_widget
    combobox = widget.layer_comboboxes[index]
    widget.viewer.layers[4].name = "New"
    assert combobox.findText("New") == 5
<<<<<<< HEAD

=======

>>>>>>> ca9d00ac72ec2ed8da80f9c789640adade88c432
