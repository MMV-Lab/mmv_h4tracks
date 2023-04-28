
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout

from ._logger import notify
from ._grabber import grab_layer

class SegmentationWindow(QWidget):
    """
    A (QWidget) window to correct the segmentation of the data.
    
    Attributes
    ----------
    viewer : Viewer
        The Napari viewer instance
    
    Methods
    -------
    """
    
    def __init__(self, viewer):
        """
        Parameters
        ----------
        viewer : Viewer
            The Napari viewer instance
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Segmentation correction")
        self.viewer = viewer
        
        ### QObjects
        
        # Labels
        label_false_positive = QLabel("Remove false positive for ID:")
        label_next_free = QLabel("Next free label:")
        label_false_merge = QLabel("Separate falsely merged ID:")
        label_false_cut = QLabel("Merge falsely cut ID and second ID:")
        label_grab_label = QLabel("Select label:")
        
        # Buttons
        btn_false_positive = QPushButton("Remove")
        btn_false_positive.setToolTip(
            "Remove label from segmentation"
        )
        btn_false_positive.clicked.connect(self._print_on_clicks)
        
        btn_free_label = QPushButton("Load Label")
        btn_free_label.setToolTip(
            "Load next free segmentation label"
        )
        
        btn_false_merge = QPushButton("Separate")
        btn_false_merge.setToolTip(
            "Split two separate parts of the same label into two"
        )
        
        btn_false_cut = QPushButton("Merge")
        btn_false_cut.setToolTip(
            "Merge two separate labels into one"
        )
        
        btn_grab_label = QPushButton("Select")
        btn_grab_label.setToolTip(
            "Load selected segmentation label"
        )
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())

        content.layout().addWidget(label_false_positive, 0, 0)
        content.layout().addWidget(btn_false_positive, 0, 1)
        content.layout().addWidget(label_next_free, 1, 0)
        content.layout().addWidget(btn_free_label, 1, 1)
        content.layout().addWidget(label_false_merge, 2, 0)
        content.layout().addWidget(btn_false_merge, 2, 1)
        content.layout().addWidget(label_false_cut, 3, 0)
        content.layout().addWidget(btn_false_cut, 3, 1)
        content.layout().addWidget(label_grab_label, 4, 0)
        content.layout().addWidget(btn_grab_label, 4, 1)
        
        self.layout().addWidget(content)
        
    def _remove_label(self, position):
        """
        Removes the cell at the given position from the segmentation layer
        
        Parameters
        ----------
        position : list
            list of float values describing the position the user clicked on the layer (z,y,x)
        """
        # replace label with 0 to make it background
        self._replace_label(position, 0)
    
    def _set_label_id(self, id):
        """
        Sets the given id as the current id in the label layer
        
        Parameters
        ----------
        id : int
            the ID to set as the currently selected one in the napari viewer
        """
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            print("no label layer found")
            raise ValueError("Trying to set label id on missing label layer")
        
        # set the new id
        label_layer.selected_label = self._get_free_label_id(label_layer)
        
        # set the label layer as currently selected layer
        self.viewer.current_viewer().layers.select_all()
        self.viewer.current_viewer().layers.selection.select_only(label_layer)
        
    def _get_free_label_id(self, label_layer):
        """
        Calculates the next free id in the passed layer
        (always returns maximum value + 1 for now, could be changed later)
        
        Parameters
        ----------
        label_layer : layer
            label layer to calculate the free id for
        
        Returns
        -------
        int
            a free id
        """
        return np.amax(layer.data) + 1
    
    def _replace_label(self, position, id):
        """
        Replaces the label at the given position with the given ID
        
        Parameters
        ----------
        position : list
            list of float values describing the position the user clicked on the layer (z,y,x)
        id : int
            the id to set for the given position
        """
        try:
            label_layer = grab_layer(self.viewer, "Segmentation Data")
        except ValueError:
            print("Tried to replace label but no label layer found")
            notify("Please make sure the label layer exists!")
            return
        
        x = int(event.position[2])
        y = int(event.position[1])
        z = int(event.position[0])
        
        # Replace the ID with the new id
        old_id = label_layer.data[z, y, x]
        np.place(label_layer.data[z], label_layer.data[z] == old_id, id)
        
        # set the label layer as currently selected layer
        self.viewer.current_viewer().layers.select_all()
        self.viewer.current_viewer().layers.selection.select_only(label_layer)
        
    def _read_label_id(self, position):
        """
        Reads the label id at the given position
        
        Parameters
        ----------
        position : list
            position of the mouse event
            
        Returns
        -------
        int
            id at the given position in the segmentation layer
        """
        try:
            label_layer = grab_layer(self.viewer, "Segmentation Data")
        except ValueError:
            print("Tried to replace label but no label layer found")
            notify("Please make sure the label layer exists!")
            return
        
        x = int(event.position[2])
        y = int(event.position[1])
        z = int(event.position[0])
        
        return label_layer.data[z,y,x]
        
    def _print_on_clicks(self):
        for layer in self.viewer.layers:
            print('layer: {}'.format(layer))
            for callback in layer.mouse_drag_callbacks:
                print(callback)
        
        
        
        
        
        
        