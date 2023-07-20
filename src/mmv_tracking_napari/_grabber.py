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
    try:
        return viewer.layers[viewer.layers.index(layer_name)]
    except ValueError:
        raise ValueError("Layer named {} does not exist".format(layer_name))
