"""Module providing a function getting a layer from a viewer"""


def grab_layer(viewer, layer_name):
    """
    Function to grab a layer with a given name

    Attributes
    ----------
    viewer : Viewer
        napari viewer instance
    layer_name : str
        name of the layer

    Returns
    -------
    layer
        returns layer if it is found, throws ValueError otherwise
    """
    if layer_name == "":
        raise ValueError("Layer name can not be blank")
    try:
        return viewer.layers[viewer.layers.index(layer_name)]
    except ValueError as exc:
        raise ValueError(f"Layer named '{layer_name}' does not exist") from exc
