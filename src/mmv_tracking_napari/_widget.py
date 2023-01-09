
import napari.layers.labels.labels
import numpy as np
import pandas as pd
import zarr
import copy
from qtpy.QtCore import QThread#, pyqtSignal <- DOESN'T WORK FOR SOME REASON, THUS EXPLICITLY IMPORTED FROM PYQT5
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                            QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
                            QScrollArea, QToolBox, QVBoxLayout, QWidget, QRadioButton)
from scipy import ndimage

from ._ressources import State, Window, SelectFromCollection, AdjustSegWorker, AdjustTracksWorker, SliceCounter, ThreadSafeCounter, SizeWorker, CPUSegWorker, GPUSegWorker
from ._functions import *
from ._reader import load_zarr
from ._writer import save_zarr

        
class MMVTracking(QWidget):
    dock = None
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        MMVTracking.dock = self

        # Variable to hold zarr file
        self.z1 = None

        # Variables to store clicked centroids for Tracking
        self.to_track = [] #np.empty((1,3),dtype=np.int8)
        self.to_cut = np.empty((1,3),dtype=np.int8)

        # Variable to hold complete (corrected) tracks layer
        self.tracks = np.empty((1,4),dtype=np.int8)
        
        # Variables to hold data for plot metrics
        self.speed = [] # example
        self.size = []
        self.direction = []
        self.euclidean_distance = [] 
        self.accumulated_distance = []
        
        # Variables to hold state of layers from when metrics were last run
        self.speed_tracks = [] # example
        self.size_seg = []
        self.direction_tracks = []
        self.euclidean_distance_tracks = [] 
        self.accumulated_distance_tracks = []
        
        # Variable to hold state of layers before human correction for evaluation
        self.segmentation_old = []
        self.tracks_old = []

        # Labels
        title = QLabel("<font color='green'>HITL4Trk</font>")
        next_free = QLabel("Next free label:")
        trajectory = QLabel("Select ID for trajectory:")
        load_save = QLabel("Load/Save .zarr file:")
        false_positive = QLabel("Remove false positive for ID:")
        false_merge = QLabel("Separate falsely merged ID:")
        false_cut = QLabel("Merge falsely cut ID and second ID:")
        remove_correspondence = QLabel("Remove tracking for later Slices for ID:")
        insert_correspondence = QLabel("ID should be tracked with second ID:")
        metric = QLabel("Evaluation metrics:")
        grab_label = QLabel("Select label:")
        self.progress_description = QLabel("Descriptive Description")
        self.progress_state = QLabel("")
        self.progress_info = QLabel("")
        min_movement = QLabel("Movement Minimum")
        min_duration = QLabel("Minimum Track Length")

        # Tooltips for Labels
        load_save.setToolTip(
            "Loading: Select the .zarr directory to open the file.<br><br>\n\n"
            "Saving: Overwrites the file selected at the time of loading, or creates a new one if none was loaded"
        )
        

        # Buttons
        btn_load = QPushButton("Load")
        btn_false_positive = QPushButton("Remove")
        btn_false_merge = QPushButton("Separate")
        btn_false_cut = QPushButton("Merge")
        self.btn_remove_correspondence = QPushButton("Unlink")
        btn_insert_correspondence = QPushButton("Link")
        btn_save = QPushButton("Save")
        btn_plot = QPushButton("Plot")
        btn_segment = QPushButton("Run instance segmentation")
        btn_preview_segment = QPushButton("Preview Segmentation")
        btn_track = QPushButton("Run tracking")
        btn_free_label = QPushButton("Load Label")
        btn_grab_label = QPushButton("Select")
        btn_export = QPushButton("Export")
        self.btn_adjust_seg_ids = QPushButton("Adjust Segmentation IDs")
        btn_auto_track = QPushButton("Automatic tracking")
        btn_delete_displayed_tracks = QPushButton("Delete displayed tracks")
        btn_evaluate_segmentation = QPushButton("Evaluate segmentation")
        btn_evaluate_tracking = QPushButton("Evaluate tracking")
        btn_store_eval_data = QPushButton("Store segmentation/tracking")
        btn_auto_track_all = QPushButton("Automatic-er Tracking")
        
        # Tooltips for Buttons
        self.btn_adjust_seg_ids.setToolTip(
            "WARNING: This will take a while"
            )
        self.btn_remove_correspondence.setToolTip(
            "HOTKEY: I<br><br>\n\n"
            "Delete cells from their tracks"
        )
        btn_insert_correspondence.setToolTip(
            "HOTKEY: U<br><br>\n\n"
            "Add cells to new track"
        )
        btn_load.setToolTip(
            "HOTKEY: Q<br><br>\n\n"
            "Load a zarr file"
        )
        btn_false_positive.setToolTip(
            "HOTKEY: R<br><br>\n\n"
            "Remove label from segmentation"
        )
        btn_false_merge.setToolTip(
            "HOTKEY: T<br><br>\n\n"
            "Split two separate parts of the same label into two"
        )
        btn_false_cut.setToolTip(
            "HOTKEY: Z<br><br>\n\n"
            "Merge two separate labels into one"
        )
        btn_save.setToolTip(
            "HOTKEY: W<br><br>\n\n"
            "Save zarr file"            
        )
        btn_free_label.setToolTip(
            "HOTKEY: E<br><br>\n\n"
            "Load next free segmentation label"
        )
        btn_grab_label.setToolTip(
            "HOTKEY: A<br><br>\n\n"
            "Load selected segmentation label"
        )

        # Linking buttons to functions
        btn_load.clicked.connect(self._load_zarr)
        btn_plot.clicked.connect(self._plot)
        btn_save.clicked.connect(self._save_zarr)
        btn_false_positive.clicked.connect(self._remove_fp)
        btn_segment.clicked.connect(self._run_segmentation)
        btn_preview_segment.clicked.connect(self._run_demo_segmentation)
        btn_track.clicked.connect(self._run_tracking)
        btn_false_merge.clicked.connect(self._false_merge)
        btn_free_label.clicked.connect(self._set_free_id)
        btn_false_cut.clicked.connect(self._false_cut)
        btn_grab_label.clicked.connect(self._grab_label)
        self.btn_remove_correspondence.clicked.connect(self._unlink)
        btn_insert_correspondence.clicked.connect(self._link)
        btn_export.clicked.connect(self._export)
        self.btn_adjust_seg_ids.clicked.connect(self._adjust_ids)
        btn_auto_track.clicked.connect(self._auto_track)
        btn_delete_displayed_tracks.clicked.connect(self._remove_displayed_tracks)
        btn_evaluate_segmentation.clicked.connect(self._evaluate_segmentation)
        btn_evaluate_tracking.clicked.connect(self._evaluate_tracking)
        btn_store_eval_data.clicked.connect(self._store_segmentation)
        btn_store_eval_data.clicked.connect(self._store_tracks)
        btn_auto_track_all.clicked.connect(self._auto_track_all)
       
        # Combo Boxes
        self.c_segmentation = QComboBox()
        self.c_plots = QComboBox()

        # Adding entries to Combo Boxes
        self.c_segmentation.addItem("select model")
        self.c_segmentation.addItem("model 1")
        self.c_segmentation.addItem("model 2")
        self.c_segmentation.addItem("model 3")
        self.c_segmentation.addItem("model 4")
        self.c_plots.addItem("speed")
        self.c_plots.addItem("size")
        self.c_plots.addItem("direction")
        self.c_plots.addItem("euclidean distance")  # example
        self.c_plots.addItem("accumulated distance")

        # Line Edits
        self.le_trajectory = QLineEdit("")
        self.le_movement = QLineEdit("")
        self.le_track_duration = QLineEdit("")
        self.le_limit_evaluation = QLineEdit("0")

        # Link functions to line edits
        self.le_trajectory.editingFinished.connect(self._select_track)
        
        # Checkboxes: off -> 0, on -> 2 if not tristate
        self.ch_speed = QCheckBox("Speed")
        self.ch_size = QCheckBox("Size")
        self.ch_direction = QCheckBox("Direction") # example
        self.ch_euclidean_distance = QCheckBox("Euclidean distance") 
        self.ch_accumulated_distance = QCheckBox("Accumulated distance")
        
        # Progressbar
        #self.progress = Progress()
        self.pb_global_progress = QProgressBar()
        
        # Tool Box
        self.toolbox = QToolBox()
        
        # Radio Buttons
        self.rb_eco = QRadioButton("Eco")
        self.rb_eco.toggle()
        self.rb_heavy = QRadioButton("Heavy")

        # Running segmentation/tracking UI
        q_seg_track = QWidget()
        q_help = QWidget()
        q_help.setLayout(QVBoxLayout())
        q_help.layout().addWidget(btn_segment)
        q_help.layout().addWidget(btn_preview_segment)
        q_seg_track.setLayout(QGridLayout())
        q_seg_track.layout().addWidget(q_help,0,0)
        q_seg_track.layout().addWidget(btn_track,0,1)
        q_seg_track.layout().addWidget(self.c_segmentation,2,0)
        q_seg_track.layout().addWidget(self.btn_adjust_seg_ids,3,0)

        # Loading/Saving .zarr file UI
        q_load = QWidget()
        q_load.setLayout(QHBoxLayout())
        q_load.layout().addWidget(load_save)
        q_load.layout().addWidget(btn_load)
        q_load.layout().addWidget(btn_save)

        # Correcting segmentation UI
        help_false_positive = QWidget()
        help_false_positive.setLayout(QHBoxLayout())
        help_false_positive.layout().addWidget(false_positive)
        help_false_positive.layout().addWidget(btn_false_positive)
        help_false_negative = QWidget()
        help_false_negative.setLayout(QHBoxLayout())
        help_false_negative.layout().addWidget(next_free)
        help_false_negative.layout().addWidget(btn_free_label)
        help_false_merge = QWidget()
        help_false_merge.setLayout(QHBoxLayout())
        help_false_merge.layout().addWidget(false_merge)
        help_false_merge.layout().addWidget(btn_false_merge)
        help_false_cut = QWidget()
        help_false_cut.setLayout(QHBoxLayout())
        help_false_cut.layout().addWidget(false_cut)
        help_false_cut.layout().addWidget(btn_false_cut)
        help_grab_layer = QWidget()
        help_grab_layer.setLayout(QHBoxLayout())
        help_grab_layer.layout().addWidget(grab_label)
        help_grab_layer.layout().addWidget(btn_grab_label)
        q_segmentation = QWidget()
        q_segmentation.setLayout(QVBoxLayout())
        q_segmentation.layout().addWidget(help_grab_layer)
        q_segmentation.layout().addWidget(help_false_negative)
        q_segmentation.layout().addWidget(help_false_positive)
        q_segmentation.layout().addWidget(help_false_merge)
        q_segmentation.layout().addWidget(help_false_cut)

        # Postprocessing tracking UI
        help_trajectory = QWidget()
        help_trajectory.setLayout(QHBoxLayout())
        help_trajectory.layout().addWidget(trajectory)
        help_trajectory.layout().addWidget(self.le_trajectory)
        help_remove_correspondence = QWidget()
        help_remove_correspondence.setLayout(QHBoxLayout())
        help_remove_correspondence.layout().addWidget(remove_correspondence)
        help_remove_correspondence.layout().addWidget(self.btn_remove_correspondence)
        help_insert_correspondence = QWidget()
        help_insert_correspondence.setLayout(QHBoxLayout())
        help_insert_correspondence.layout().addWidget(insert_correspondence)
        help_insert_correspondence.layout().addWidget(btn_insert_correspondence)
        q_tracking = QWidget()
        q_tracking.setLayout(QVBoxLayout())
        q_tracking.layout().addWidget(help_trajectory)
        q_tracking.layout().addWidget(btn_delete_displayed_tracks)
        q_tracking.layout().addWidget(help_remove_correspondence)
        q_tracking.layout().addWidget(help_insert_correspondence)
        q_tracking.layout().addWidget(btn_auto_track)
        q_tracking.layout().addWidget(btn_auto_track_all)

        # Evaluation UI
        help_movement = QWidget()
        help_movement.setLayout(QHBoxLayout())
        help_movement.layout().addWidget(min_movement)
        help_movement.layout().addWidget(self.le_movement)
        help_track_length = QWidget()
        help_track_length.setLayout(QHBoxLayout())
        help_track_length.layout().addWidget(min_duration)
        help_track_length.layout().addWidget(self.le_track_duration)
        help_plot = QWidget()
        help_plot.setLayout(QHBoxLayout())
        help_plot.layout().addWidget(metric)
        help_plot.layout().addWidget(self.c_plots)
        help_plot.layout().addWidget(btn_plot)
        q_eval = QWidget()
        q_eval.setLayout(QVBoxLayout())
        q_eval.layout().addWidget(help_movement)
        q_eval.layout().addWidget(help_track_length)
        q_eval.layout().addWidget(help_plot)
        q_eval.layout().addWidget(self.ch_speed)
        q_eval.layout().addWidget(self.ch_size)
        q_eval.layout().addWidget(self.ch_direction) # example
        q_eval.layout().addWidget(self.ch_euclidean_distance) 
        q_eval.layout().addWidget(self.ch_accumulated_distance) 
        q_eval.layout().addWidget(btn_export)
        q_eval.layout().addWidget(btn_evaluate_segmentation)
        q_eval.layout().addWidget(self.le_limit_evaluation)
        q_eval.layout().addWidget(btn_evaluate_tracking)

        # Add zones to self.toolbox
        self.toolbox.addItem(q_seg_track, "Data Processing")
        self.toolbox.addItem(q_segmentation, "Segmentation correction")
        self.toolbox.addItem(q_tracking, "Tracking correction")
        self.toolbox.addItem(q_eval, "Analysis")
        
        # Progress UI
        help_progress = QWidget()
        help_progress.setLayout(QVBoxLayout())
        help_progress.layout().addWidget(self.progress_description)
        help_progress.layout().addWidget(self.pb_global_progress)
        self.pb_global_progress.setValue(100)
        help_progress.layout().addWidget(self.progress_state)
        help_progress.layout().addWidget(self.progress_info)
        
        # Computation Mode selection
        help_mode = QWidget()
        help_mode.setLayout(QHBoxLayout())
        help_mode.layout().addWidget(self.rb_eco)
        help_mode.layout().addWidget(self.rb_heavy)

        # Assemble UI elements in ScrollArea
        scroll_area = QScrollArea()
        scroll_area.setLayout(QVBoxLayout())
        scroll_area.layout().addWidget(title)
        scroll_area.layout().addWidget(help_mode)
        scroll_area.layout().addWidget(q_load)
        scroll_area.layout().addWidget(btn_store_eval_data)
        scroll_area.layout().addWidget(self.toolbox)
        scroll_area.layout().addWidget(help_progress)

        # Set ScrollArea as content of plugin
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)
        self.setMinimumWidth(400)

        self._mouse(State.default)

    # Functions

    def _mouse(self,mode,seg_id = 0, paint = False):
        """
        hub for adding functionality to mouseclicks
        
        :param mode: used to discern which function to call on mouseclick
        :param seg_id: Segmentation ID to change selected cell to
        :param paint: Sets mode of label layer to paint if True 
        """
        
        for layer in self.viewer.layers: # Functions get applied to every layer
            if len(layer.mouse_drag_callbacks):
                if layer.mouse_drag_callbacks[0].__name__ == "no_op": # no_op is a function set by napari itself, and it is always the first in the list
                    layer.mouse_drag_callbacks.pop(-1)
                else:
                    layer.mouse_drag_callbacks.clear()

            if mode == State.default:
                try:
                    self.viewer.layers.selection.active.help = "(0)"
                except AttributeError:
                    pass
            elif mode == State.test: # Unused at the moment
                self.viewer.layers.selection.active.help = "(-1)"
            elif mode == State.remove: # False Positive -- Delete cell from label layer
                self.viewer.layers.selection.active.help = "(1)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    """
                    Removes cell from segmentation
                    
                    :param event: Mouseclick event
                    """
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    # Replace the ID with 0 (id of background)
                    false_id = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    np.place(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0])],self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0])]==false_id,0)
                    napari.viewer.current_viewer().layers.select_all()
                    napari.viewer.current_viewer().layers.selection.select_only(label_layer)
                    label_layer.refresh()
                    self._mouse(State.default)
            elif mode == State.recolour: # False Merge -- Two separate cells have the same label, relabel one
                self.viewer.layers.selection.active.help = "(2)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    """
                    Changes ID of cell from selection
                    
                    :param event: Mouseclick event
                    """
                    try:
                        self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    # Selected cell gets new label
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),self._get_free_id(self.viewer.layers[self.viewer.layers.index("Segmentation Data")]))
                    self._mouse(State.default)
            elif mode == State.merge_from: # False Cut 1 -- Two cells should be one
                self.viewer.layers.selection.active.help = "(3)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    """
                    Selects cell ID from segmentation
                    
                    :param event: Mouseclick event
                    """
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    if label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])] == 0:
                        message("Can't merge background!")
                        self._mouse(State.default)
                        return
                    # Select cell, pass ID on
                    self._mouse(State.merge_to, label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])])
            elif mode == State.merge_to: # False Cut 2 -- Two cells should be one
                self.viewer.layers.selection.active.help = "(4)"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    """
                    Changes cell ID in segmentation
                    
                    :param event: Mouseclick event
                    """
                    if self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[int(event.position[0]),int(event.position[1]),int(event.position[2])] == 0:
                        message("Can't merge background!")
                        self._mouse(State.default)
                        return
                    # Label layer can't be missing as this is only called from False Cut 1
                    # Change ID of selected cell to the ID passed on
                    self.viewer.layers[self.viewer.layers.index("Segmentation Data")].fill((int(event.position[0]),int(event.position[1]),int(event.position[2])),seg_id)
                    self._mouse(State.default)
            elif mode == State.select: # Correct Segmentation -- Cell needs to be redrawn. Loads ID of clicked cell and switches to painting mode if selected
                self.viewer.layers.selection.active.help = "(5)"
                @layer.mouse_drag_callbacks.append
                def _handle(layer,event):
                    """
                    Load ID of cell to label layer
                    
                    :param event: Mouseclick event
                    """
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        _message("Missing label layer")
                        self._mouse(State.default)
                        return
                    label_layer.selected_label = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    napari.viewer.current_viewer().layers.select_all()
                    napari.viewer.current_viewer().layers.selection.select_only(label_layer)
                    if paint:
                        import keyboard
                        keyboard.press_and_release("2")
                    self._mouse(State.default)
            elif mode == State.link: # Creates Track -- Creates a new track or extends an existing one 
                try:
                    self.viewer.layers.selection.active.help = "(6)"
                except AttributeError:
                    pass
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _record(layer,event):
                    """
                    Records centroids of selected cells
                    
                    :param event: Mouseclick event
                    """
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        try:
                            self.viewer.layers.selection.active.help = "YOU MISSED THE CELL, PRESS THE BUTTON AGAIN AND CONTINUE FROM THE LAST VALID INPUT!"
                        except AttributeError:
                            message("You missed the cell. Press the button again and continue from the last valid input")
                        self._link()
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self.to_track.append([int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                    if self.progress_info.text():
                        self.test.append(int(event.position[0]))
                        self.test.sort()
                        self.progress_info.setText(str(self.test))
                        #self.progress_info.setText(self.progress_info.text() + ", " + str(int(event.position[0])))
                    else:
                        self.test = [int(event.position[0])]
                        self.progress_info.setText(str(self.test))
                        #self.progress_info.setText(str(int(event.position[0])))

            elif mode == State.unlink: # Removes Track -- Removes cells from track
                self.viewer.layers.selection.active.help = "(7)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _cut(layer,event):
                    """
                    Records centroids of selected cells
                    
                    :param event: Mouseclick event
                    """
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        self.viewer.layers.selection.active.help = "NO CELL SELECTED, DO BETTER NEXT TIME!"
                        self._mouse(State.default)
                        return
                    centroid = ndimage.center_of_mass(label_layer.data[int(event.position[0])], labels = label_layer.data[int(event.position[0])], index = selected_cell)
                    self.to_cut.append([int(event.position[0]),int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                    if self.progress_info.text():
                        self.progress_info.setText(self.progress_info.text() + ", "+ str(int(event.position[0])))
                    else:
                        self.progress_info.setText(str(int(event.position[0])))
                    
            elif mode == State.auto_track: # Automatically track cell if condition is met
                self.viewer.layers.selection.active.help = "(8)"
                if isinstance(layer,napari.layers.labels.labels.Labels):
                    layer.mode = "pan_zoom"
                @layer.mouse_drag_callbacks.append
                def _track(layer,event):
                    """
                    Tracks selected cell automatically until overlap is not sufficiently large anymore
                    
                    :param event: Mouseclick event
                    """
                    try:
                        self.viewer.layers[self.viewer.layers.index("Tracks")]
                    except ValueError:
                        self.tracks = np.empty((1,4),dtype=np.int8)
                    try:
                        label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
                    except ValueError:
                        message("Missing label layer")
                        self._mouse(State.default)
                        return
                    selected_cell = label_layer.data[int(event.position[0]),int(event.position[1]),int(event.position[2])]
                    if selected_cell == 0: # Make sure a cell has been selected
                        self.viewer.layers.selection.active.help = "NO CELL SELECTED, DO BETTER NEXT TIME!"
                        self._mouse(State.default)
                        return
                    cell = np.where(label_layer.data[int(event.position[0])]== selected_cell)
                    my_slice = int(event.position[0])
                    while my_slice + 1 < len(label_layer.data):
                        matching = label_layer.data[my_slice + 1][cell]
                        matches = np.unique(matching,return_counts = True)
                        maximum = np.argmax(matches[1])
                        if matches[1][maximum] <= 0.7 * np.sum(matches[1]):
                            print("ABORTING1")
                            self._mouse(State.default)
                            if len(self.to_track) < 5:
                                self.to_track = []
                                self._mouse(State.default)
                                return
                            self._link(auto = True)
                            return
                        if matches[0][maximum] == 0:
                            print("ABORTING2")
                            self._mouse(State.default)
                            if len(self.to_track) < 5:
                                self.to_track = []
                                self._mouse(State.default)
                                return
                            self._link(auto = True)
                            return
                        if my_slice == int(event.position[0]):
                            centroid = ndimage.center_of_mass(label_layer.data[my_slice], labels = label_layer.data[my_slice], index = selected_cell)
                            self.to_track.append([my_slice,int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                        centroid = ndimage.center_of_mass(label_layer.data[my_slice + 1], labels = label_layer.data[my_slice + 1], index = matches[0][maximum])
                        if [my_slice+1,int(np.rint(centroid[0])),int(np.rint(centroid[1]))] in self.tracks[:,1:4].tolist():
                            print("Found tracked cell, aborting")
                            if (len(self.to_track)) < 5:
                                self.to_track = []
                                self._mouse(State.default)
                                return
                            else:
                                self._link(auto = True)
                                self._mouse(State.default)
                                return
                            
                        self.to_track.append([my_slice+1,int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                        
                        selected_cell = matches[0][maximum]
                        my_slice = my_slice + 1
                        cell = np.where(label_layer.data[my_slice] == selected_cell)
                    if len(self.to_track) < 5:
                        self.to_track = []
                        self._mouse(State.default)
                        return
                    self._link(auto = True)
                    self._mouse(State.default)

    @napari.Viewer.bind_key('q')
    def _hotkey_load_zarr(self):
        MMVTracking.dock._load_zarr()
        
    def _load_zarr(self):
        """
        Opens a dialog to select a zarr file.
        Loads the zarr file's content as layers into the viewer
        """
        #load_zarr()
        dialog = QFileDialog()
        dialog.setNameFilter('*.zarr')
        self.file = dialog.getExistingDirectory(self, "Select Zarr-File")
        if(self.file == ""):
            print("No file selected")
            return
        self.z1 = zarr.open(self.file,mode='a')

        # check if "Raw Image", "Segmentation Data" or "Track" exist in self.viewer.layers
        if "Raw Image" in self.viewer.layers or "Segmentation Data" in self.viewer.layers or "Tracks" in self.viewer.layers:
            ret = message(title="Layer name blocked", text="Found layer name", informative_text="One or more layers with the names \"Raw Image\", \"Segmentation Data\" or \"Tracks\" exists already. Continuing will delete those layers. Are you sure?", buttons=[("Continue", QMessageBox.AcceptRole),QMessageBox.Cancel])
            # ret = 0 means Continue was selected, ret = 4194304 means Cancel was selected
            if ret == 4194304:
                return
            try:
                self.viewer.layers.remove("Raw Image")
            except ValueError: # only one or two layers may exist, so not all can be deleted
                pass
            try:
                self.viewer.layers.remove("Segmentation Data")
            except ValueError: # see above
                pass
            try:
                self.viewer.layers.remove("Tracks")
            except ValueError: # see above
                pass
        try:
            self.viewer.add_image(self.z1['raw_data'][:], name = 'Raw Image')
            self.viewer.add_labels(self.z1['segmentation_data'][:], name = 'Segmentation Data')
            #self.viewer.add_tracks(self.z1['tracking_data'][:], name = 'Tracks') # Use graph argument for inheritance (https://napari.org/howtos/layers/tracks.html)
            self.tracks = self.z1['tracking_data'][:] # Cache data of tracks layer
        except:
            print("File is either no Zarr file or does not adhere to required structure")
        else:
            tmp = np.unique(self.tracks[:,0],return_counts = True) # Count occurrences of each id
            tmp = np.delete(tmp,tmp[1] == 1,1)
            self.tracks = np.delete(self.tracks,np.where(np.isin(self.tracks[:,0],tmp[0,:],invert=True)),0) # Remove tracks of length <2
            self.viewer.add_tracks(self.tracks, name='Tracks')
        self._mouse(State.default)
        
        
        """import sys
        print(sys.getsizeof(self.viewer.layers[self.viewer.layers.index("Raw Image")].data)) # <- huge for whatever reason
        print(sys.getsizeof(self.viewer.layers))
        print(sys.getsizeof(self.tracks))"""
    
    @napari.Viewer.bind_key('w')
    def _hotkey_save_zarr(self):
        MMVTracking.dock._save_zarr()

    def _save_zarr(self):
        """
        Saves the (changed) layers to a zarr file
        """
        #save_zarr()
        # Useful if we later want to allow saving to new file
        try:
            raw = self.viewer.layers.index("Raw Image")
        except ValueError:
            message("No Raw Data layer found!")
            return
        try: # Check if segmentation layer exists
            seg = self.viewer.layers.index("Segmentation Data")
        except ValueError:
            message("No Segmentation Data layer found!")
            return
        try: # Check if tracks layer exists
            track = self.viewer.layers.index("Tracks")
        except ValueError:
            message("No Tracks layer found!")
            return

        ret = 1
        if self.le_trajectory.text() != "": # Some tracks are potentially left out
            ret = message(title="Tracks", text="Limited Tracks layer", informative_text="It looks like you have selected only some of the tracks from your tracks layer. Do you want to save only the selected ones or all of them?", buttons=[("Save Selected",QMessageBox.YesRole),("Save All",QMessageBox.NoRole),QMessageBox.Cancel])
            # Save Selected -> ret = 0, Save All -> ret = 1, Cancel -> ret = 4194304
            if ret == 4194304:
                return
        if ret == 0: # save current tracks layer
            if self.z1 == None:
                dialog = QFileDialog()
                dialog.setNameFilter('*.zarr')
                file = dialog.getSaveFileName(self, "Select location for zarr to be created")
                self.file = file #+ ".zarr"
                self.z1 = zarr.open(self.file,mode='w')
                self.z1.create_dataset('raw_data', shape = self.viewer.layers[raw].data.shape, dtype = 'f8', data = self.viewer.layers[raw].data)
                self.z1.create_dataset('segmentation_data', shape = self.viewer.layers[seg].data.shape, dtype = 'i4', data = self.viewer.layers[seg].data)
                self.z1.create_dataset('tracking_data', shape = self.viewer.layers[track].data.shape, dtype = 'i4', data = self.viewer.layers[track].data)
            else:
                self.z1['raw_data'][:] = self.viewer.layers[raw].data
                self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
                self.z1['tracking_data'].resize(self.viewer.layers[track].data.shape[0],self.viewer.layers[track].data.shape[1])
                self.z1['tracking_data'][:] = self.viewer.layers[track].data
                #self.z1.create_dataset('tracking_data', shape = self.viewer.layers[track].data.shape, dtype = 'i4', data = self.viewer.layers[track].data)
        else: # save complete tracks layer
            if self.z1 == None:
                dialog = QFileDialog()
                dialog.setNameFilter('*.zarr')
                file = dialog.getSaveFileName(self, "Select location for zarr to be created")
                self.file = file[0] + ".zarr"
                self.z1 = zarr.open(self.file,mode='w')
                self.z1.create_dataset('raw_data', shape = self.viewer.layers[raw].data.shape, dtype = 'f8', data = self.viewer.layers[raw].data)
                self.z1.create_dataset('segmentation_data', shape = self.viewer.layers[seg].data.shape, dtype = 'i4', data = self.viewer.layers[seg].data)
                self.z1.create_dataset('tracking_data', shape = self.tracks, dtype = 'i4', data = self.tracks)
            else:
                self.z1['raw_data'][:] = self.viewer.layers[raw].data
                self.z1['segmentation_data'][:] = self.viewer.layers[seg].data
                self.z1['tracking_data'].resize(self.tracks.shape[0],self.tracks.shape[1])
                self.z1['tracking_data'][:] = self.tracks
                #self.z1.create_dataset('tracking_data', shape = self.tracks.shape, dtype = 'i4', data = self.tracks)
        message("Zarr file has been saved.")

    def _plot(self): # TODO: test running thread from thread, move to separate thread to allow multithreading for size calculation
        """
        Plots the data for the selected metric
        """
        # Throw warning message if plot is generated and cached tracks are different from current tracks
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        fig = Figure(figsize=(6,7))
        fig.patch.set_facecolor("#262930")
        axes = fig.add_subplot(111)
        axes.set_facecolor("#262930")
        axes.spines["bottom"].set_color("white")
        axes.spines["top"].set_color("white")
        axes.spines["right"].set_color("white")
        axes.spines["left"].set_color("white")
        axes.xaxis.label.set_color("white")
        axes.yaxis.label.set_color("white")
        axes.tick_params(axis="x", colors="white")
        axes.tick_params(axis="y", colors="white")
        canvas = FigureCanvas(fig)
        self.window = Window()
        self.window.setLayout(QVBoxLayout())
        
        if self.c_plots.currentIndex() == 0: # Speed metric
            if not (type(self.speed) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.speed_tracks)):
                self._calculate_speed()
            speed = self.speed
            unique_ids = np.unique(speed[:,0]).astype(int)
            axes.set_title("Speed",{"fontsize": 18,"color": "white"})
            axes.set_xlabel("Average")
            axes.set_ylabel("Standard Deviation")
            data = axes.scatter(speed[:,1],speed[:,2],c = np.array([[0,0.240802676,0.70703125,1]]))
            self.window.layout().addWidget(QLabel("Scatterplot Standard Deviation vs Average: Speed"))
            self.speed_tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data
        elif self.c_plots.currentIndex() == 1: # Size metric
            if not (type(self.size) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data,self.size_seg)):
                self._calculate_size()
            size = self.size
            unique_ids = np.unique(size[:,0]).astype(int)
            axes.set_title("Size",{"fontsize": 18,"color": "white"})
            axes.set_xlabel("Average")
            axes.set_ylabel("Standard Deviation")
            data = axes.scatter(size[:,1],size[:,2],c = np.array([[0,0.240802676,0.70703125,1]]))
            self.window.layout().addWidget(QLabel("Scatterplot Standard Deviation vs Average: Size"))
            self.size_seg = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        elif self.c_plots.currentIndex() == 2: # Direction metric
            if not (type(self.direction) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.direction_tracks)): # example
                self._calculate_travel()
            direction = self.direction
            unique_ids = np.unique(direction[:,0]).astype(int)
            axes.set_title("Direction",{"fontsize": 18,"color": "white"})
            axes.axvline(color='white')
            axes.axhline(color='white')
            data = axes.scatter(direction[:,1],direction[:,2],c = np.array([[0,0.240802676,0.70703125,1]]))
            self.window.layout().addWidget(QLabel("Scatterplot: Travel direction & Distance"))
            self.direction_tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data
        elif self.c_plots.currentIndex() == 3: # Euclidean distance metric
            if not (type(self.euclidean_distance) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.euclidean_distance_tracks)): 
                self._calculate_euclidean_distance()   
            euclidean_distance = self.euclidean_distance  
            unique_ids = np.unique(euclidean_distance[:,0]).astype(int) 
            axes.set_title("Euclidean distance",{"fontsize": 18,"color": "white"})
            axes.set_xlabel("x")
            axes.set_ylabel("y")     
            data = axes.scatter(euclidean_distance[:,1],euclidean_distance[:,2],c = np.array([[0,0.240802676,0.70703125,1]]))    
            self.window.layout().addWidget(QLabel("Scatterplot x vs y"))
            self.euclidean_distance_tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data   
        elif self.c_plots.currentIndex() == 4: # Accumulated distance metric
            if not (type(self.accumulated_distance) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.accumulated_distance_tracks)): 
                self._calculate_accumulated_distance()   
            accumulated_distance = self.accumulated_distance  
            unique_ids = np.unique(accumulated_distance[:,0]).astype(int) 
            axes.set_title("Accumulated distance",{"fontsize": 18,"color": "white"})
            axes.set_xlabel("x")
            axes.set_ylabel("y")     
            data = axes.scatter(accumulated_distance[:,1],accumulated_distance[:,2],c = np.array([[0,0.240802676,0.70703125,1]]))    
            self.window.layout().addWidget(QLabel("Scatterplot x vs y"))
            self.euclidean_distance_tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data               
            
                                    
        selector = SelectFromCollection(self, axes, data, unique_ids)
        
        def accept(event):
            """
            this is somehow important, TBD why
            """
            if event.key == "enter":
                print("Selected points:")
                print(selector.xys[selector.ind])
                selector.disconnect()
                axes.set_title("")
                fig.canvas.draw()
                
        fig.canvas.mpl_connect("key_press_event",accept)
        self.window.layout().addWidget(canvas)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(selector.apply)
        self.window.layout().addWidget(btn_apply)
        self.window.show()
        
    def _export(self):
        """
        Exports a CSV with selected metrics 
        """
        if not (self.ch_speed.checkState() or self.ch_size.checkState() or self.ch_direction.checkState() or self.ch_euclidean_distance.checkState() or self.ch_accumulated_distance.checkState()):     # example
            ret = message(title="No metric selected", text="You selected no metrics", informative_text="Are you sure you want to export just the amount of cells?", buttons=[QMessageBox.Yes,QMessageBox.Cancel])
             # Yes -> ret = 16384, Cancel -> ret = 4194304
            if ret == 4194304:
                return
        import csv
        dialog = QFileDialog()
        #dialog.setDefaultSuffix("csv") Doesn't work for some reason
        file = dialog.getSaveFileName(filter = "*.csv")
        if file[0] == "":
            # No file selected
            return
        """if not file[0].endswith(".csv"):
            print("ADD CSV")
        print(file)"""
        csvfile = open(file[0],'w', newline='')
        writer = csv.writer(csvfile)
        
        if not (type(self.size) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data,self.size_seg)):
            self._calculate_travel()
            
        if self.le_movement.text() == "":
            min_movement = 0
            movement_mask = np.unique(self.tracks[:,0])
        else:
            try:
                min_movement = int(self.le_movement.text())
            except ValueError:
                message(title="Wrong type", text="String detected", informative_text="Please use integer instead of text")
                return
            else:
                if min_movement != float(self.le_movement.text()):
                    message(title="Wrong type", text="Float detected", informative_text="Please use integer instead of float")
                    return
                movement_mask = self.direction[np.where(self.direction[:,4] >= min_movement)[0],0]
             
        if self.le_track_duration.text() == "":
            min_duration = 0
            duration_mask = np.unique(self.tracks[:,0])
        else:
            try:
                min_duration = int(self.le_track_duration.text())
            except ValueError:
                message(title="Wrong type", text="String detected", informative_text="Please use integer instead of text")
                err.exec()
                return
            else:
                if min_duration != float(self.le_track_duration.text()):
                    message(title="Wrong type", text="Float detected", informative_text="Please use integer instead of float")
                    return
                duration_mask = np.unique(self.tracks[:,0])[np.where(np.unique(self.tracks[:,0],return_counts=True)[1]>= min_duration)]
            
        combined_mask = np.intersect1d(movement_mask,duration_mask)
        #print(combined_mask)
        #print(np.unique(self.tracks[:,0]))
        
        duration = np.asarray([ [i,np.count_nonzero(self.tracks[:,0] == i)] for i in np.unique(self.tracks[:,0])])
        
        # Stats for all cells combined
        metrics = [""]
        metrics.append("Number of cells")
        metrics.append("Average track duration")
        metrics.append("Standard deviation of track duration")
        individual_metrics = ["ID","Track duration"]
        all_values = ["all"]
        all_values.append(len(np.unique(self.tracks[:,0]))) # example
        #all_values.append(len(self.tracks[:,0])/len(np.unique(self.tracks[:,0])))
        all_values.append(np.around(np.average(duration[:,1]),3))
        all_values.append(np.around(np.std(duration[:,1]),3))
        valid_values = ["valid"]
        valid_values.append(len(combined_mask))
        valid_values.append(np.around(np.average( [duration[i,1] for i in range(0,len(duration)) if duration[i,0] in combined_mask]),3))
        valid_values.append(np.around(np.std( [duration[i,1] for i in range(0,len(duration)) if duration[i,0] in combined_mask]),3))
        if self.ch_speed.checkState() == 2:
            if not (type(self.speed) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.speed_tracks)):
                self._calculate_speed()
            metrics.append("Average speed")
            #metrics.append("Standard deviation of speed")
            individual_metrics.append("Average speed")
            individual_metrics.append("Standard deviation of speed")
            #print(self.speed)
            all_values.append(np.around(np.average(self.speed[:,1]),3))
            #all_values.append(np.around(np.std(self.speed[:,1]),3))
            valid_values.append(np.around(np.average( [self.speed[i,1] for i in range(0,len(self.speed)) if self.speed[i,0] in combined_mask]),3))
            #valid_values.append(np.around(np.std( [self.speed[i,1] for i in range(0,len(self.speed)) if self.speed[i,0] in combined_mask]),3))
        if self.ch_size.checkState() == 2:
            if not (type(self.size) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data,self.size_seg)):
                self._calculate_size()
            metrics.append("Average size")
            metrics.append("Standard deviation of size")
            individual_metrics.append("Average size")
            individual_metrics.append("Standard deviation of size")
            all_values.append(np.around(np.average(self.size[:,1]),3))
            all_values.append(np.around(np.std(self.size[:,1]),3))
            valid_values.append(np.around(np.average( [self.size[i,1] for i in range(0,len(self.size)) if self.size[i,0] in combined_mask]),3))
            valid_values.append(np.around(np.std( [self.size[i,1] for i in range(0,len(self.size)) if self.size[i,0] in combined_mask]),3))
        if self.ch_direction.checkState() == 2:
            if not (type(self.direction) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.direction_tracks)):
                self._calculate_travel()
            metrics.append("Average direction")
            metrics.append("Standard deviation of direction")
            metrics.append("Average distance")
            #metrics.append("Standard deviation of distance")
            individual_metrics.append("Direction")
            individual_metrics.append("Distance")
            all_values.append(np.around(np.average(self.direction[:,3]),3))
            all_values.append(np.around(np.std(self.direction[:,3]),3))
            all_values.append(np.around(np.average(self.direction[:,4]),3))
            #all_values.append(np.around(np.std(self.direction[:,4]),3))
            valid_values.append(np.around(np.average( [self.direction[i,3] for i in range(0,len(self.direction)) if self.direction[i,0] in combined_mask]),3))
            valid_values.append(np.around(np.std( [self.direction[i,3] for i in range(0,len(self.direction)) if self.direction[i,0] in combined_mask]),3))
            valid_values.append(np.around(np.average( [self.direction[i,4] for i in range(0,len(self.direction)) if self.direction[i,0] in combined_mask]),3))
            #valid_values.append(np.around(np.std( [self.direction[i,4] for i in range(0,len(self.direction)) if self.direction[i,0] in combined_mask]),3))
        if self.ch_euclidean_distance.checkState() == 2:
            if not (type(self.euclidean_distance) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.euclidean_distance_tracks)):
                self._calculate_euclidean_distance()
            metrics.append("Average euclidean distance")
            #metrics.append("Standard deviation of euclidean distance")
            metrics.append("Average directed speed")
            #metrics.append("Standard deviation of directed speed")
            individual_metrics.append("Euclidean distance")
            individual_metrics.append("Directed speed")
            all_values.append(np.around(np.average(self.euclidean_distance[:,1]),3))
            #all_values.append(np.around(np.std(self.euclidean_distance[:,1]),3))
            all_values.append(np.around(np.average(self.euclidean_distance[:,1]/len(np.unique(self.tracks[:,0]))),3))
            #all_values.append(np.around(np.std(self.euclidean_distance[:,1]/len(np.unique(self.tracks[:,0]))),3))
            valid_values.append(np.around(np.average( [self.euclidean_distance[i,1] for i in range(0,len(self.euclidean_distance)) if self.euclidean_distance[i,0] in combined_mask]),3))
            #valid_values.append(np.around(np.std( [self.euclidean_distance[i,1] for i in range(0,len(self.euclidean_distance)) if self.euclidean_distance[i,0] in combined_mask]),3))
            valid_values.append(np.around(np.average( [self.euclidean_distance[i,1]/duration[i,1] for i in range(0,len(self.euclidean_distance)) if self.euclidean_distance[i,0] in combined_mask]),3))
            #valid_values.append(np.around(np.std( [self.euclidean_distance[i,1]/duration[i,1] for i in range(0,len(self.euclidean_distance)) if self.euclidean_distance[i,0] in combined_mask]),3))
        if self.ch_accumulated_distance.checkState() == 2:
            if not (type(self.accumulated_distance) == np.ndarray and np.array_equal(self.viewer.layers[self.viewer.layers.index("Tracks")].data,self.accumulated_distance_tracks)):
                self._calculate_accumulated_distance()
            metrics.append("Average accumulated distance")
            #metrics.append("Standard deviation of accumulated distance")
            individual_metrics.append("Accumulated distance")
            all_values.append(np.around(np.average(self.accumulated_distance[:,1]),3))
            #all_values.append(np.around(np.std(self.accumulated_distance[:,1]),3))
            valid_values.append(np.around(np.average( [self.accumulated_distance[i,1] for i in range(0,len(self.accumulated_distance)) if self.accumulated_distance[i,0] in combined_mask]),3))
            #valid_values.append(np.around(np.std( [self.accumulated_distance[i,1] for i in range(0,len(self.accumulated_distance)) if self.accumulated_distance[i,0] in combined_mask]),3))
            if self.ch_euclidean_distance.checkState() == 2:
                metrics.append("Average directness")
                #metrics.append("Standard deviation of directness")
                individual_metrics.append("Directness")
                directness = np.asarray([ [self.euclidean_distance[i,0],self.euclidean_distance[i,1]/self.accumulated_distance[i,1] if self.accumulated_distance[i,1] > 0 else 0] for i in range(0,len(np.unique(self.tracks[:,0])))])
                all_values.append(np.around(np.average(directness[:,1]),3))
                #all_values.append(np.around(np.std(directness[:,1]),3))
                valid_values.append(np.around(np.average( [directness[i,1] for i in range(0,len(directness)) if directness[i,0] in combined_mask]),3))
                #valid_values.append(np.around(np.std( [directness[i,1] for i in range(0,len(directness)) if directness[i,0] in combined_mask]),3))
        writer.writerow(metrics)
        writer.writerow(all_values)
        writer.writerow(valid_values)
        writer.writerow(["Movement Threshold: " + str(min_movement),"Duration Threshold: " + str(min_duration)])
        writer.writerow([None])
        writer.writerow([None])
        
        # Stats for each individual cell
        if not (self.ch_speed.checkState() or self.ch_size.checkState() or self.ch_direction.checkState() or self.ch_euclidean_distance.checkState() or self.ch_accumulated_distance.checkState()):
            csvfile.close()
            message("Export complete")
            return
        writer.writerow(individual_metrics)
        for track in combined_mask:
            value = [track]
            value.append(np.count_nonzero(self.tracks[:,0] == track))
            if self.ch_speed.checkState() == 2: # example
                value.append(self.speed[np.where(self.speed[:,0] == track)[0],1][0])
                value.append(self.speed[np.where(self.speed[:,0] == track)[0],2][0])
            if self.ch_size.checkState() == 2:
                value.append(self.size[np.where(self.size[:,0] == track)[0],1][0])
                value.append(self.size[np.where(self.size[:,0] == track)[0],2][0])
            if self.ch_direction.checkState() == 2:
                value.append(self.direction[np.where(self.direction[:,0] == track)[0],3][0])
                value.append(self.direction[np.where(self.direction[:,0] == track)[0],4][0])
            if self.ch_euclidean_distance.checkState() == 2:
                value.append(self.euclidean_distance[np.where(self.euclidean_distance[:,0] == track)[0],1][0])
                value.append(np.around(self.euclidean_distance[np.where(self.euclidean_distance[:,0] == track)[0],1][0]/np.count_nonzero(self.tracks[:,0] == track),3))
            if self.ch_accumulated_distance.checkState() == 2:
                value.append(self.accumulated_distance[np.where(self.accumulated_distance[:,0] == track)[0],1][0])
                if self.ch_accumulated_distance.checkState() == 2:
                    value.append(np.around(directness[np.where(directness[:,0] == track)[0],1][0],3))      
            writer.writerow(value)
            
        if not np.array_equal(np.unique(self.tracks[:,0]),combined_mask):
            writer.writerow([None])
            writer.writerow([None])
            writer.writerow(["invalid cells"])
            for track in np.unique(self.tracks[:,0]):
                if track in combined_mask:
                    continue
                value = [track]
                value.append(np.count_nonzero(self.tracks[:,0] == track))
                if self.ch_speed.checkState() == 2: # example
                    value.append(self.speed[np.where(self.speed[:,0] == track)[0],1][0])
                    value.append(self.speed[np.where(self.speed[:,0] == track)[0],2][0])
                if self.ch_size.checkState() == 2:
                    value.append(self.size[np.where(self.size[:,0] == track)[0],1][0])
                    value.append(self.size[np.where(self.size[:,0] == track)[0],2][0])
                if self.ch_direction.checkState() == 2:
                    value.append(self.direction[np.where(self.direction[:,0] == track)[0],3][0])
                    value.append(self.direction[np.where(self.direction[:,0] == track)[0],4][0])
                if self.ch_euclidean_distance.checkState() == 2:
                    value.append(self.euclidean_distance[np.where(self.euclidean_distance[:,0] == track)[0],1][0])
                    value.append(np.around(self.euclidean_distance[np.where(self.euclidean_distance[:,0] == track)[0],1][0]/np.count_nonzero(self.tracks[:,0] == track),3))
                if self.ch_accumulated_distance.checkState() == 2:
                    value.append(self.accumulated_distance[np.where(self.accumulated_distance[:,0] == track)[0],1][0])
                    if self.ch_accumulated_distance.checkState() == 2:
                        value.append(np.around(directness[np.where(directness[:,0] == track)[0],1][0],3))            
                writer.writerow(value)
        csvfile.close()
        message("Export complete")
                

    def _select_track(self, tracks = []):
        """
        Displays only selected tracks
        
        :param tracks: list of IDs of tracks to display
        """
        if tracks == []:                
            if self.le_trajectory.text() == "": # Deleting the text returns the whole layer
                try:
                    self.viewer.layers.remove('Tracks')
                except ValueError:
                    print("No tracking layer found")
                self.viewer.add_tracks(self.tracks, name='Tracks')
                return
            try: # This works for a single value
                tracks = int(self.le_trajectory.text())
            except ValueError: # This works for multiple  comma separated values
                txt = self.le_trajectory.text()
                tracks = []
                try:
                    for i in range(0,len(txt.split(","))):
                        tracks.append(int((txt.split(",")[i])))
                except ValueError:
                    message("Please use a single integer (whole number) or a comma separated list of integers")
                    return
        try:
            self.viewer.layers.remove('Tracks')
        except ValueError:
            print("No tracking layer found")
        if isinstance(tracks,int): # Single value
            if tracks < 0:
                self.viewer.add_tracks(self.tracks, name='Tracks')
                self.le_trajectory.setText("") # Negative number gets removed
            else:
                tracks_data = [
                    track
                    for track in self.tracks
                    if track[0] == tracks
                ]
                if not tracks_data:
                    print("No tracking data found for id " + str(tracks) + ", displaying all tracks instead")
                    self.viewer.add_tracks(self.tracks, name='Tracks')
                    return
                self.viewer.add_tracks(tracks_data, name='Tracks')
            self._mouse(State.default)
        else: # Multiple values, tracks is instance of "list"
            tracks = list(dict.fromkeys(tracks)) # Removes duplicate values
            for i in range(0,len(tracks)): # Remove illegal values (<0) from tracks
                if int(tracks[i]) < 0:
                    tracks.pop(i)
            # ID now only contains legal values, can be written back to line edit
            txt = ""
            for i in range(0,len(tracks)):
                if len(txt)>0:
                    txt = txt + ","
                txt = f'{txt}{tracks[i]}'
            self.le_trajectory.setText(txt)
            # Get tracks data for selected IDs
            tracks_data = [
                track
                for track in self.tracks
                if track[0] in tracks
            ]
            if not tracks_data:
                print("No tracking data found for ids " + str(tracks) + ", displaying all tracks instead")
                self.viewer.add_tracks(self.tracks,name='Tracks')
                return
            self.viewer.add_tracks(tracks_data, name='Tracks')
            self._mouse(State.default)

    @napari.Viewer.bind_key('e')
    def _hotkey_get_free_id(self):
        MMVTracking.dock._set_free_id()
        MMVTracking.dock.viewer.layers[MMVTracking.dock.viewer.layers.index("Segmentation Data")].mode = "paint"

    def _set_free_id(self):
        """
        Sets free segmentation ID on label layer
        """
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            message("Missing label layer")
            return
        label_layer.selected_label = self._get_free_id(label_layer)
        napari.viewer.current_viewer().layers.select_all()
        napari.viewer.current_viewer().layers.selection.select_only(label_layer)

    def _get_free_id(self, layer):
        """
        Finds a free segmentation ID
        
        :return: integer, free segmentation ID
        """
        return np.amax(layer.data)+1

    @napari.Viewer.bind_key('r')
    def _hotkey_remove_fp(self):
        MMVTracking.dock._remove_fp()

    def _remove_fp(self):
        """
        Removes the clicked on cell from segmentation layer
        """
        self._mouse(State.remove)

    @napari.Viewer.bind_key('t')
    def _hotkey_false_merge(self):
        MMVTracking.dock._false_merge()

    def _false_merge(self):
        """
        Changes ID for clicked on cell from segmentation layer
        """
        self._mouse(State.recolour)

    @napari.Viewer.bind_key('z')
    def _hotkey_false_cut(self):
        MMVTracking.dock._false_cut()

    def _false_cut(self):
        """
        Adapts ID from second to first selected cell
        """
        self._mouse(State.merge_from)

    # Tracking correction
    @napari.Viewer.bind_key('u')
    def _hotkey_link(self):
        MMVTracking.dock._link()

    def _link(self, auto = False):
        """
        Links cells together to form a track
        Records inputs on first run, creates track on second run
        """
        try: # Check if segmentation layer exists
            layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            message("No label layer found!")
            return
        try:
            tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data
        except ValueError:
            pass
        else:
            if not np.array_equal(tracks,self.tracks): # Check if full tracks layer is displayed
                ret = message(title="No Tracks", text="Missing Tracks", informative_text="You need to have all Tracks displayed to add a new Track. Do you want to display all Tracks now?", buttons=[("Display all", QMessageBox.AcceptRole),QMessageBox.Cancel])
                # ret = 0 -> Display all, ret = 4194304 -> Cancel
                if ret == 4194304:
                    return
                self.le_trajectory.setText("")
                self._select_track()
        for i in range(len(layer.mouse_drag_callbacks)):
            if layer.mouse_drag_callbacks[i].__name__ == "_record" or auto: # Check if we are in recording mode already or are in auto mode
                print("yes auto")
                if len(self.to_track) < 2: # Less than two cells can not be tracked
                    print("too little slices for track")
                    self.to_track = []
                    self._mouse(State.default)
                    self._set_state()
                    self._set_state_info()
                    return
                if len(np.asarray(self.to_track)[:,0]) != len(set(np.asarray(self.to_track)[:,0])): # Check for duplicates
                    message(title="Duplicate Cells", text="Duplicate cells per slice", informative_text="Looks like you selected more than one cell per slice. This does not work.", buttons=[QMessageBox.Ok])
                    self.to_track = []
                    self._mouse(State.default)
                    self._set_state()
                    self._set_state_info()
                    return
                self.to_track.sort()
                try: # Check if tracks layer must be created
                    track = self.viewer.layers.index("Tracks")
                except ValueError:
                    track_id = 1
                else:
                    tracks = self.viewer.layers[track].data
                    self.viewer.layers.remove('Tracks')
                    track_id = max(np.amax(tracks[:,0]),np.amax(tracks[:,0])) + 1 # Determine id for the new track
                old_ids = [0,0]
                if track_id != 1: # Tracking data is not empty
                    for j in range(len(tracks)):
                        if tracks[j][1] == self.to_track[0][0] and tracks[j][2] == self.to_track[0][1] and tracks[j][3] == self.to_track[0][2]: # New track starting point exists in tracking data
                            old_ids[0] = tracks[j][0]
                            self.to_track.remove(self.to_track[0])
                            break
                    for j in range(len(tracks)):
                        if tracks[j][1] == self.to_track[-1][0] and tracks[j][2] == self.to_track[-1][1] and tracks[j][3] == self.to_track[-1][2]: # New track end point exists in tracking data
                            old_ids[1] = tracks[j][0]
                            self.to_track.remove(self.to_track[-1])
                            break
                if max(old_ids) > 0:
                    if min(old_ids) == 0: # One end connects to existing track
                        track_id = max(old_ids)
                    else: # Both ends connect to existing track, (higher) id of second existing track changed to id of first track
                        track_id = min(old_ids)
                        for track_entry in tracks:
                            if track_entry[0] == max(old_ids):
                                track_entry[0] = track_id
                for entry in self.to_track: # Entries are added to tracking data (current and cached, in case those are different)
                    try:
                        tracks = np.r_[tracks, [[track_id] + entry]]
                    except UnboundLocalError:
                        tracks = [[track_id] + entry]
                self.to_track = []
                df = pd.DataFrame(tracks, columns=['ID', 'Z', 'Y', 'X'])
                df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                self.tracks = df.values
                self.viewer.add_tracks(df.values, name='Tracks')
                self._mouse(State.default)
                self._set_state()
                self._set_state_info()
                return
        print("never should have come here")
        try:
            self.viewer.layers.selection.active.help = ""
        except AttributeError:
            pass
        self.to_track = []
        self._mouse(State.link)
        self._set_state("Selected cells to link in slices:")

    @napari.Viewer.bind_key('i')
    def _hotkey_unlink(self):
        MMVTracking.dock._unlink()

    def _unlink(self):
        """
        Removes cells from track
        Records inputs on first run, removes cells from tracking on second run
        This deletes tracks with length < 2
        """
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            message("No label layer found!")
            return
        try:
            tracks_layer = self.viewer.layers[self.viewer.layers.index("Tracks")]
        except ValueError:
            message("No tracks layer found!")
            return
        track_id = max(np.amax(tracks_layer.data[:,0]),np.amax(self.tracks[:,0])) + 1
        tracks = tracks_layer.data
        for i in range(len(label_layer.mouse_drag_callbacks)):
            if label_layer.mouse_drag_callbacks[i].__name__ == "_cut":
                if len(self.to_cut) < 2:
                    message("Please select more than one cell!")
                    self.to_cut = []
                    self._mouse(State.default)
                    self._switch_button()
                    self._set_state()
                    self._set_state_info()
                    return
                self.to_cut.sort()
                track = 0
                for j in range(len(tracks_layer.data)): # Find track ID
                    if tracks[j,1] == self.to_cut[0][0] and tracks[j,2] == self.to_cut[0][1] and tracks[j,3] == self.to_cut[0][2]:
                        track = tracks[j,0]
                        break
                for j in range(len(tracks_layer.data)):  # Confirm track ID matches other entries
                    if tracks[j,1] == self.to_cut[-1][0] and tracks[j,2] == self.to_cut[-1][1] and tracks[j,3] == self.to_cut[-1][2]:
                        if track != tracks[j,0]:
                            message("Please select cells that belong to the same Track!")
                            self.to_cut = []
                            self._mouse(State.default)
                            return
                j = 0
                while j < len(tracks):
                    if tracks[j,0] == track:
                        if tracks[j,1] > self.to_cut[0][0]:
                            if tracks[j,1] < self.to_cut[-1][0]: # Cell is removed from tracking
                                to_delete = tracks[j]
                                tracks = np.delete(tracks,j,0)
                                k = 0
                                while k < len(self.tracks):
                                    if np.array_equal(self.tracks[k], to_delete):
                                        self.tracks = np.delete(self.tracks,k,0)
                                        break
                                    k = k + 1
                                j = j - 1
                            elif tracks[j,1] >= self.to_cut[-1][0]: # Cell gets moved to track with new ID
                                tracks[j,0] = track_id
                                k = 0
                                while k < len(self.tracks):
                                    if np.array_equal(self.tracks[k],np.array([track,tracks[j,1],tracks[j,2],tracks[j,3]])):
                                        self.tracks[k,0] = track_id
                                        break
                                    k = k + 1
                    j = j + 1
                self.to_cut = []
                df = pd.DataFrame(tracks, columns=['ID', 'Z', 'Y', 'X'])
                df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                tracks = df.values
                df_cache = pd.DataFrame(self.tracks, columns=['ID', 'Z', 'Y', 'X'])
                df_cache.sort_values(['ID', 'Z'], ascending=True, inplace=True)
                self.tracks = df_cache.values
                tmp = np.unique(tracks[:,0],return_counts = True) # Count occurrences of each id
                tmp = np.delete(tmp,tmp[1] == 1,1)
                tracks = np.delete(tracks,np.where(np.isin(tracks[:,0],tmp[0,:],invert=True)),0) # Remove tracks of length <2
                tmp = np.unique(self.tracks[:,0],return_counts = True) # Count occurrences of each id
                tmp = np.delete(tmp,tmp[1] == 1,1)
                self.tracks = np.delete(self.tracks,np.where(np.isin(self.tracks[:,0],tmp[0,:],invert=True)),0) # Remove tracks of length <2
                self.viewer.layers.remove('Tracks')
                print(tracks.shape)
                if tracks.shape[0] > 0:
                    self.viewer.add_tracks(tracks, name='Tracks')
                elif self.tracks.shape[0] > 0:
                    self.viewer.add_tracks(self.tracks, name ='Tracks')
                self._mouse(State.default)
                self._switch_button()
                self._set_state()
                self._set_state_info()
                return
        self.to_cut = []
        self._mouse(State.unlink)
        self._switch_button()
        self._set_state("Selected cells to unlink in slices:")
        
    def _remove_displayed_tracks(self):
        try:
            tracks_layer = self.viewer.layers[self.viewer.layers.index("Tracks")]
        except ValueError:
            message("No tracks layer found!")
            return
        to_remove = np.unique(tracks_layer.data[:,0])
        if np.array_equal(to_remove,np.unique(self.tracks[:,0])):
            message("Can not delete whole tracks layer!")
            return
        ret = message(text="Are you sure? This will delete the following tracks: " + str(to_remove), buttons=[("Continue", QMessageBox.AcceptRole),QMessageBox.Cancel])
        # ret = 0 means Continue was selected, ret = 4194304 means Cancel was selected
        if ret == 4194304:
            return
        print("<(<) <( )> (>)>")
        self.tracks = np.delete(self.tracks, np.isin(self.tracks[:,0],to_remove),0)
        self.viewer.layers.remove('Tracks')
        self.le_trajectory.setText("")
        self.viewer.add_tracks(self.tracks, name='Tracks')
        pass
        
    def _switch_button(self):
        if not self.btn_remove_correspondence.text() == "Unlink":
            self.btn_remove_correspondence.setText("Unlink")
        else:
            self.btn_remove_correspondence.setText("Confirm")
            
    @napari.Viewer.bind_key('x')
    def _default_hotkey(self):
        MMVTracking.dock._default()

    def _default(self):
        self._mouse(State.default)

    @napari.Viewer.bind_key('a')
    def _hotkey_grab_label(self):
        MMVTracking.dock._grab_label(paint = True)

    def _grab_label(self, paint = False):
        """
        Sets layer to ID of selected cell
        
        :param paint: Puts label layer in paint mode if true
        """
        try:
            self.viewer.layers[self.viewer.layers.index("Segmentation Data")].mode = "pan_zoom"
        except ValueError:
            message("Missing label layer")
            return
        self._mouse(State.select, paint = paint)

    @napari.Viewer.bind_key('1')
    def _hotkey_zone_1(self):
        MMVTracking.dock.toolbox.setCurrentIndex(0)

    @napari.Viewer.bind_key('2')
    def _hotkey_zone_2(self):
        MMVTracking.dock.toolbox.setCurrentIndex(1)

    @napari.Viewer.bind_key('3')
    def _hotkey_zone_3(self):
        MMVTracking.dock.toolbox.setCurrentIndex(2)

    @napari.Viewer.bind_key('4')
    def _hotkey_zone_4(self):
        MMVTracking.dock.toolbox.setCurrentIndex(3)
        
    def _calculate_speed(self):
        """
        Calculates average speed and standard deviation for all cells
        """
        
        """ Speed metric:
            - avg speed (every cell) <- prio
            - std (standard deviation from avg speed)
            - peak speed (overall)
            - mean speed (every cell)
        """
        for unique_id in np.unique(self.tracks[:,0]):
            track = np.delete(self.tracks,np.where(self.tracks[:,0] != unique_id),0)
            distance = []
            for i in range(0,len(track)-1):
                distance.append(np.hypot(track[i,2] - track[i+1,2],track[i,3] - track[i+1,3]))
            avg_speed = np.around(np.average(distance),3)
            std_speed = np.around(np.std(distance),3)
            try:
                retval = np.append(retval, [[unique_id,avg_speed,std_speed]],0)
            except UnboundLocalError:
                retval = np.array([[unique_id,avg_speed,std_speed]])
        self.speed =  retval
    
    def _calculate_size(self): #TODO: Multithread
        """
        Calculates average size and standard deviation for all cells
        """
        
        """Size metric:
            - avg size
            - std
            - mean
            - peak?
            - minimum?
        """            
        
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            message("No label layer found!")
            return
        """#def __init__(self,parent,label_layer,tracks,counter,completed):
        import multiprocessing
        if self.rb_eco.isChecked():
            self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.4))
        else:
            self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.8))
        
        self.rb_eco.setEnabled(False)
        self.rb_heavy.setEnabled(False)
        
        self.done = ThreadSafeCounter(0)
        self.completed = ThreadSafeCounter(0)
        next_id = SliceCounter()
        
        self.threads = [QThread() for _ in range(self.AMOUNT_OF_THREADS)]
        self.workers = [SizeWorker(self, label_layer, self.tracks, next_id, self.completed) for _ in range(self.AMOUNT_OF_THREADS)]
        
        for i in range(0,self.AMOUNT_OF_THREADS):
            self.workers[i].moveToThread(self.threads[i])
            self.threads[i].started.connect(self.workers[i].run)
            self.workers[i].finished.connect(self.threads[i].quit)
            
        def _test():
            print("TEST")
        for worker in self.workers:
            worker.starting.connect(_test)
            worker.update_progress.connect(self._multithread_progress)
            worker.status.connect(self._multithread_description)
            worker.values.connect(self._append_size)
            worker.finished.connect(worker.deleteLater)
            
        self._update_progress(0)
            
        for thread in self.threads:
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._enable_rb)
            thread.finished.connect(self._sort_size)
            thread.start()
            
        print("Main thread now waiting for worker threads")
        #self.threads[0].wait()
        for thread in self.threads:
            thread.wait()
            print("Thread done waiting")
            
        print("(╯°□°)╯︵ ┻━┻")"""
            
        for unique_id in np.unique(self.tracks[:,0]): 
            track = np.delete(self.tracks,np.where(self.tracks[:,0] != unique_id),0)
            size = []
            for i in range(0,len(track)-1):
                seg_id = label_layer.data[track[i,1],track[i,2],track[i,3]]
                size.append(len(np.where(label_layer.data[track[i,1]] == seg_id)[0]))
            avg_size = np.around(np.average(size),3)
            std_size = np.around(np.std(size),3)
            try:
                retval = np.append(retval, [[unique_id,avg_size,std_size]],0)
            except UnboundLocalError:
                retval = np.array([[unique_id,avg_size,std_size]])
        self.size = retval
        
    def _append_size(self, values):
        try:
            self.size = np.append(self.size, [[values[0],values[1],values[2]]],0)
        except ValueError:
            self.size = np.array([[values[0],values[1],values[2]]])
        
    def _sort_size(self):
        if self.completed.value() == len(self.viewer.layers[self.viewer.layers.index("Raw Image")].data):
            df = pd.DataFrame(self.size, columns=['ID','AVG','STD'])
            df.sort_values(['ID'],ascending=True, inplace=True)
            self.size = df.values
        
    def _calculate_travel(self):
        """
        Calculates direction and distance traveled for all cells
        """
        for unique_id in np.unique(self.tracks[:,0]):
            track = np.delete(self.tracks,np.where(self.tracks[:,0] != unique_id),0)
            x = track[-1,3] - track[0,3]
            y = track[0,2] - track[-1,2]
            if y > 0:
                if x == 0:
                    direction = np.pi /2
                else:
                    direction = np.pi - np.arctan(np.abs(y/x))
            elif y < 0:
                if x == 0:
                    direction = 1.5 * np.pi
                else:
                    direction = np.pi - np.arctan(np.abs(y/x))
            else:
                if x < 0:
                    direction = np.pi
                else:
                    direction = 0
            distance = np.around(np.sqrt(np.square(x) + np.square(y)),3)
            direction = np.around(direction,3)
            try:
                retval = np.append(retval, [[unique_id,x,y,direction,distance]],0)
            except UnboundLocalError:
                retval = np.array([[unique_id,x,y,direction,distance]])
        self.direction = retval
        
        
        # example 
        ###### METRICS HERE
    def _calculate_euclidean_distance(self):            # Already implemented in _calculate_travel, we could combine these
        """
        Calculates euclidean distance between first and last frame in which a cell is tracked
        """    
        for unique_id in np.unique(self.tracks[:,0]):
            track = np.delete(self.tracks,np.where(self.tracks[:,0] != unique_id),0)
            x = track[-1,3] - track[0,3]
            y = track[0,2] - track[-1,2]            
            euclidean_distance = np.around(np.sqrt(np.square(x) + np.square(y)),3)
            try:
                retval = np.append(retval, [[unique_id,euclidean_distance,0]],0)
            except UnboundLocalError:
                retval = np.array([[unique_id,euclidean_distance,0]])
        self.euclidean_distance = retval

    def _calculate_accumulated_distance(self):
        """
        Calculates accumulated distance for each cell
        """
        for unique_id in np.unique(self.tracks[:,0]):
            track = np.delete(self.tracks,np.where(self.tracks[:,0] != unique_id),0)
            distance = [] # TODO: clarify/rename
            for i in range(0,len(track)-1):
                distance.append(np.hypot(track[i,2] - track[i+1,2],track[i,3] - track[i+1,3]))
            accumulated_distance = np.around(np.sum(distance),3)
            try:
                retval = np.append(retval, [[unique_id,accumulated_distance,0]],0)
            except UnboundLocalError:
                retval = np.array([[unique_id,accumulated_distance,0]])
        self.accumulated_distance =  retval        

        
    def _adjust_ids(self):
        """
        Replaces Track ID 0 with new Track ID
        Changes Segmentation IDs to corresponding Track IDs
        """
        import multiprocessing
        if self.rb_eco.isChecked():
            self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.4))
        else:
            self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.8))
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            message("No label layer found!")
            return
        self.btn_adjust_seg_ids.setEnabled(False)
        self.rb_eco.setEnabled(False)
        self.rb_heavy.setEnabled(False)
        
        self.done = ThreadSafeCounter(0) 
        self.new_id = 0
        self.completed = ThreadSafeCounter(0)
        next_slice = SliceCounter()
        
        self.threads = [QThread() for _ in range(self.AMOUNT_OF_THREADS)]
        
        for thread in self.threads:
            thread.finished.connect(thread.deleteLater)
            
        self.tracks_worker = AdjustTracksWorker(self,self.tracks)
        self.tracks_worker.moveToThread(self.threads[0])
        self.threads[0].started.connect(self.tracks_worker.run)
        self.tracks_worker.progress.connect(self._update_progress)
        self.tracks_worker.tracks_ready.connect(self._replace_tracks)
        for thread in self.threads[1:self.AMOUNT_OF_THREADS]:
            self.tracks_worker.finished.connect(thread.start)
        self.tracks_worker.status.connect(self._set_description)
        self.tracks_worker.finished.connect(self.tracks_worker.deleteLater)
        
        self.seg_workers = [AdjustSegWorker(self, label_layer, self.tracks, next_slice, self.completed) for _ in range(self.AMOUNT_OF_THREADS)]
        for i in range(0,self.AMOUNT_OF_THREADS):
            self.seg_workers[i].moveToThread(self.threads[i])
            
        self.tracks_worker.finished.connect(self.seg_workers[0].run)
        for i in range(1,self.AMOUNT_OF_THREADS):
            self.threads[i].started.connect(self.seg_workers[i].run)
        
        for seg_worker in self.seg_workers:
            seg_worker.update_progress.connect(self._multithread_progress)
            seg_worker.status.connect(self._multithread_description)
            seg_worker.finished.connect(seg_worker.deleteLater)
            
        for i in range(0,self.AMOUNT_OF_THREADS):
            self.seg_workers[i].finished.connect(self.threads[i].quit)
            
        for thread in self.threads:
            thread.finished.connect(self._enable_adjust_button)
            
        self.threads[0].start()
        print("(>\'-\')>")
    
    def _enable_adjust_button(self):
        if self.completed.value() == len(self.viewer.layers[self.viewer.layers.index("Raw Image")].data):
            self.btn_adjust_seg_ids.setEnabled(True)
            self._enable_rb()
            
    def _enable_rb(self):
        if self.completed.value() == len(self.viewer.layers[self.viewer.layers.index("Raw Image")].data):
            self.rb_eco.setEnabled(True)
            self.rb_heavy.setEnabled(True)
        
    def _replace_tracks(self,tracks):
        self.viewer.layers.remove("Tracks")
        self.viewer.add_tracks(tracks, name='Tracks')
        
    def _multithread_progress(self):
        self._update_progress(self.completed.value() / len(self.viewer.layers[self.viewer.layers.index("Raw Image")].data) * 100)
            
    def _update_progress(self,value):
        self.pb_global_progress.setValue(np.rint(value))
        
    def _auto_track(self):
        self._mouse(State.auto_track)
        
    def _auto_track_all(self):
        # Expects tracks layer to not exist
        self.tracks = np.empty((1,4),dtype=np.int8)
        try:
            label_layer = self.viewer.layers[self.viewer.layers.index("Segmentation Data")]
        except ValueError:
            print("Missing label layer")
            return
        for start_slice in range(0,len(label_layer.data) - 5):
            print(start_slice)
            for current_cell in np.unique(label_layer.data[start_slice]):
                if current_cell == 0:
                    continue
                cell = np.where(label_layer.data[start_slice]== current_cell)
                current_slice = start_slice
                while current_slice + 1 < len(label_layer.data):
                    #print('Current cell: ',current_cell,'. Current slice: ', current_slice + 1, sep='')
                    matching = label_layer.data[current_slice + 1][cell]
                    matches = np.unique(matching,return_counts = True)
                    maximum = np.argmax(matches[1])
                    
                    if matches[1][maximum] <= 0.7 * np.sum(matches[1]):
                        print("ABORTING1")
                        break
                    if matches[0][maximum] == 0:
                        print("ABORTING2")
                        break
                    
                    if current_slice == start_slice:
                        centroid = ndimage.center_of_mass(label_layer.data[start_slice], labels = label_layer.data[start_slice], index = current_cell)
                        self.to_track.append([start_slice,int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                    centroid = ndimage.center_of_mass(label_layer.data[current_slice + 1], labels = label_layer.data[current_slice + 1], index = matches[0][maximum])
                    
                    if [current_slice+1,int(np.rint(centroid[0])),int(np.rint(centroid[1]))] in self.tracks[:,1:4].tolist():
                        print("Found tracked cell, aborting")
                        if (len(self.to_track)) < 5:
                            self.to_track = []
                            break
                        else:
                            self._mouse(State.link)
                            self._link(auto = True)
                            break
                        
                    self.to_track.append([current_slice+1,int(np.rint(centroid[0])),int(np.rint(centroid[1]))])
                    
                    current_cell = matches[0][maximum]
                    current_slice = current_slice + 1
                    cell = np.where(label_layer.data[current_slice] == current_cell)
                
                if len(self.to_track) >= 5:
                    self._mouse(State.link)
                    self._link(auto = True)
                self.to_track = []
        self._mouse(State.default)
                
    def _run_demo_segmentation(self):
        self._run_segmentation(True)
        pass
        
    def _run_segmentation(self, demo = False):
        from cellpose import models
        try:
            data = self.viewer.layers[self.viewer.layers.index("Raw Image")].data
        except ValueError:
            message("No image layer found!")
            return
        if demo:
            data = data[0:5]
        selected_model = self.c_segmentation.currentText()
        if selected_model == "model 1":
            model_path = 'models/cellpose_neutrophils'
            diameter=15
            chan=0
            chan2=0
            flow_threshold = 0.4
            cellprob_threshold=0
        elif selected_model == "Cellpose Tumor Cell":
            pass
        elif selected_model == "EmbedSeg":
            pass
        #from cellpose import core
        #gpu = core.use_gpu() # TODO: fix w/ justin
        gpu = False
        if not gpu:
            import multiprocessing
            if self.rb_eco.isChecked():
                self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.4))
            else:
                self.AMOUNT_OF_THREADS = np.maximum(1,int(multiprocessing.cpu_count() * 0.8))
            
            self.rb_eco.setEnabled(False)
            self.rb_heavy.setEnabled(False)
            
            self.done = ThreadSafeCounter(0) 
            self.completed = ThreadSafeCounter(0)
            next_slice = SliceCounter()
            self.result = np.zeros(data.shape)
            model = [models.CellposeModel(gpu=False, pretrained_model=model_path) for _ in range(self.AMOUNT_OF_THREADS)]
            
            self.threads = [QThread() for _ in range(self.AMOUNT_OF_THREADS)]
            self.workers = []
            for i in range(self.AMOUNT_OF_THREADS):
                self.workers.append(CPUSegWorker(self, model[i], self.result, next_slice, data, self.completed, diameter, (chan, chan2), flow_threshold, cellprob_threshold))
            
            for i in range(0,self.AMOUNT_OF_THREADS):
                self.workers[i].moveToThread(self.threads[i])
                self.threads[i].started.connect(self.workers[i].run)
                self.workers[i].finished.connect(self.threads[i].quit)
                
            for worker in self.workers:
                worker.update_progress.connect(self._multithread_progress)
                worker.status.connect(self._multithread_description)
                worker.finished.connect(worker.deleteLater)
                
            self._update_progress(0)
                
            for thread in self.threads:
                thread.finished.connect(thread.deleteLater)
                thread.finished.connect(self._enable_rb)
                thread.finished.connect(self._multithread_add_labels)
                thread.start()
                
            print("<(\'.\'<)")
        else:
            
            model = models.CellposeModel(gpu=True, pretrained_model=model_path)
            self.thread = QThread()
            self.worker = GPUSegWorker(self,model,data,diameter, (chan,chan2), flow_threshold, cellprob_threshold)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.status.connect(self._set_description)
            self.worker.update_progress.connect(self._update_progress)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.segmentation.connect(self._add_labels_help)
            self._update_progress(0)
            self.thread.start()
            print("<(\'.\')>")
        
    def _multithread_add_labels(self,name="CPU SEGMENTATION"):
        if self.done.value() == self.AMOUNT_OF_THREADS:
            self._add_labels(np.asarray(self.result,dtype="int8"),name)
        
    def _add_labels(self,data,name = "Segmentation Data"):
        self.viewer.add_labels(data, name = name)
        
    def _add_labels_help(self, param):
        self._add_labels(param[0],param[1])
        
    def _run_tracking(self): # TODO: remove unnecessary steps

        
        if "Tracks" in self.viewer.layers:
            ret = message(title="Tracks exists", text="Tracks layer found", informative_text="A tracks layer currently exists. Running this step will replace that layer. are you sure?", buttons=[("Continue", QMessageBox.AcceptRole),QMessageBox.Cancel])
            # ret = 0 means Continue was selected, ret = 4194304 means Cancel was selected
            if ret == 4194304:
                return
        self.viewer.layers.remove("Tracks")
        
        from scipy import spatial, optimize
        
        APPROX_INF = 65535
        MAX_MATCHING_DIST = 45
        TRACK_DISPLAY_LENGTH = 20
        
        img = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        total_time = img.shape[0]
        traj = dict()
        lineage = dict()
        
        for tt in range(total_time): # important
            seg = img[tt,:,:]
            if sum(sum(seg))==0:
                print("Empty label layer, no tracking possible")
                return

            # get label image
            _, num_cells = ndimage.label(seg)

            seg_label = seg

            # calculate center of mass
            centroid = ndimage.center_of_mass(seg, labels=seg_label, index=np.unique(seg)[1:])     # sometimes returns nans, TBD why ??
            
            # generate cell information of this frame
            traj.update({
                tt : {"centroid": centroid, "parent": [], "child": [], "ID": []}
            })


        # initialize trajectory ID, parent node, track pts for the first frame
        max_cell_id =  len(traj[0].get("centroid"))
        traj[0].update(
            {"ID": np.arange(0, max_cell_id, 1)}
        )
        traj[0].update(
            {"parent": -1 * np.ones(max_cell_id, dtype=int)}
        )
        centers = traj[0].get("centroid")
        pts = []
        for ii in range(max_cell_id):
            pts.append([centers[ii]])
            lineage.update({ii: [centers[ii]]})
        traj[0].update({"track_pts": pts})



        for tt in np.arange(1, total_time):
            p_prev = traj[tt-1].get("centroid")
            p_next = traj[tt].get("centroid")

            ###########################################################
            # simple LAP tracking
            ###########################################################
            num_cell_prev = len(p_prev)
            num_cell_next = len(p_next)

            # calculate distance between each pair of cells

            # if condition == '3':
            #     print(tt)
            #import pdb; pdb.set_trace()

            cost_mat = spatial.distance.cdist(p_prev, p_next)

            # if the distance is too far, change to approx. Inf.

            # if tt >= 406:
            #     import pdb; pdb.set_trace()
            cost_mat[cost_mat > MAX_MATCHING_DIST] = APPROX_INF

            # add edges from cells in previous frame to auxillary vertices
            # in order to accomendate segmentation errors and leaving cells
            cost_mat_aug = MAX_MATCHING_DIST * 1.2 * np.ones(
                (num_cell_prev, num_cell_next + num_cell_prev), dtype=float
            )
            cost_mat_aug[:num_cell_prev, :num_cell_next] = cost_mat[:, :]

            # solve the optimization problem

            if sum(sum(1*np.isnan(cost_mat))) > 0:  # check if there is at least one np.nan in cost_mat
                #print(well_name + ' terminated at frame ' + str(tt))
                print("??Justin hatte einen Denkfehler")
                break   
            row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)

            #########################################################
            # parse the matching result
            #########################################################
            prev_child = np.ones(num_cell_prev, dtype=int)
            next_parent = np.ones(num_cell_next, dtype=int)
            next_ID = np.zeros(num_cell_next, dtype=int)
            next_track_pts = []

            # assign child for cells in previous frame
            for ii in range(num_cell_prev):
                if col_ind[ii] >= num_cell_next:
                    prev_child[ii] = -1
                else:
                    prev_child[ii] = col_ind[ii]

            # assign parent for cells in next frame, update ID and track pts
            prev_pt = traj[tt-1].get("track_pts")
            prev_id = traj[tt-1].get("ID")
            for ii in range(num_cell_next):
                if ii in col_ind:
                    # a matched cell is found
                    next_parent[ii] = np.where(col_ind == ii)[0][0]
                    next_ID[ii] = prev_id[next_parent[ii]]

                    current_pts = prev_pt[next_parent[ii]].copy()
                    current_pts.append(p_next[ii])
                    if len(current_pts) > TRACK_DISPLAY_LENGTH:
                        current_pts.pop(0)
                    next_track_pts.append(current_pts)
                    # attach this point to the lineage
                    single_lineage = lineage.get(next_ID[ii])
                    try:
                        single_lineage.append(p_next[ii])
                    except Exception:
                        pdb.set_trace()
                    lineage.update({next_ID[ii]: single_lineage})
                else:
                    # a new cell
                    next_parent[ii] = -1
                    next_ID[ii] = max_cell_id
                    next_track_pts.append([p_next[ii]])
                    lineage.update({max_cell_id: [p_next[ii]]})
                    max_cell_id += 1

            # update record
            traj[tt-1].update({"child": prev_child})
            traj[tt].update({"parent": next_parent})
            traj[tt].update({"ID": next_ID})
            traj[tt].update({"track_pts": next_track_pts})
            
        tracks_layer = np.round(np.asarray(traj[0]['centroid'][0])) 
        tracks_layer = np.append(tracks_layer, [0])
        tracks_layer = np.append(tracks_layer, [traj[0]['ID'][0]])
        tracks_layer=tracks_layer[[3,2,0,1]]
    
    
        tracks_layer = np.expand_dims(tracks_layer, axis=1)
        tracks_layer = tracks_layer.T
        
    
        for i in range(len(traj[0]['ID'])-1):
            track = np.round(np.asarray(traj[0]['centroid'][i+1]))
            track = np.append(track, [0])
            track = np.append(track, [traj[0]['ID'][i+1]])
            track = track[[3,2,0,1]]
            track = np.expand_dims(track, axis=1)
            track = track.T
            tracks_layer = np.concatenate((tracks_layer, track), axis=0)
    
    
        #for i in range(9):                                                             # 10 images
        for i in range(len(traj)-1):                                                   # all images    
            for cell_ID in range(len(traj[i+1]['ID'])):
                track = np.round(np.asarray(traj[i+1]['centroid'][cell_ID]))      # centroid
                track = np.append(track, [i+1])                          # frame
                track = np.append(track, [traj[i+1]['ID'][cell_ID]])       # ID
                track = track[[3,2,0,1]]
                track = np.expand_dims(track, axis=1)
                track = track.T
                tracks_layer = np.concatenate((tracks_layer, track), axis=0)   
        
        
        
        df = pd.DataFrame(tracks_layer, columns=['ID', 'Z', 'Y', 'X'])
        df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
        tracks_formatted = df.values # this is tracks layer
        
        self.viewer.add_tracks(tracks_formatted, name = 'Tracks')
        print("      •.,¸,.•*`•.,¸¸,.•*¯ ╭━━━━╮\n •.,¸,.•*¯`•.,¸,.•*¯.|:::::::::: /___/\n•.,¸,.•*¯`•.,¸,.•* <|:::::::::(｡ ●ω●｡)\n  •.,¸,.•¯•.,¸,.•╰ * >し------し---Ｊ")
        
    def _set_state(self,state=""):
        self.progress_state.setText(state)
        
    def _set_state_info(self,info=""):
        self.progress_info.setText(info)
    
    def _multithread_description(self,value):
        if value == "Adjusting Segmentation IDs":
            self._set_description("Adjusting Segmentation IDs")
        elif value == "Calculating Segmentation":
            self._set_description("Calculating Segmentation")
        elif value == "Calculating Size":
            self._set_description("Calculating Size")
        else:
            self.done.increment()
            if self.done.value() == self.AMOUNT_OF_THREADS:
                self._set_description("Done")
                self._update_progress(100)
        
    def _set_description(self,description=""):
        self.progress_description.setText(description)
        
    def _store_segmentation(self):
        self.segmentation_old = copy.deepcopy(self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data)
        print("segmentation stored")
    
    def _store_tracks(self):
        tracks = copy.deepcopy(self.viewer.layers[self.viewer.layers.index("Tracks")].data)
        self.tracks_old = []
        for line in tracks:
            #print(line)
            self.tracks_old.append([line,self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data[line[1],line[2],line[3]]])
            pass
        #self.tracks_old = self.viewer.layers[self.viewer.layers.index("Tracks")].data
        print("tracks stored")
        
    def _evaluate_segmentation(self):
        auto_segmentation = self.segmentation_old
        human_segmentation = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        
        #### FOR CURRENT FRAME
        
        frame = int(self.viewer.dims.point[0])
        
        # Intersection/Union calculation
        intersection = np.sum( np.sum( np.logical_and(auto_segmentation[frame], human_segmentation[frame])))
        union = np.sum( np.sum( np.logical_or(auto_segmentation[frame], human_segmentation[frame])))
        
        # IoU calculation
        iou_score = intersection / union
        print("IoU score for frame %d: %s" % (frame, iou_score))
        
        # DICE score calculation
        dice_score = (2 * intersection) / (np.count_nonzero(auto_segmentation[frame]) + np.count_nonzero(human_segmentation[frame]))
        print("DICE score for frame %d: %s" % (frame, dice_score))
        
        # F1 score calculation
        # Intersection = True Positive
        # Union = True Positive + False Positive + False Negative
        f1_score = (2 * intersection) / (intersection + union)
        print("F1 score for frame %d: %s" % (frame, f1_score))
        
        #### FOR SELECTED FRAMES
        
        try:
            selected_limit = int(self.le_limit_evaluation.text())
        except ValueError:
            message(title="Wrong type", text="String detected", informative_text="Please use integer instead of text")
            return
        
        
        if frame <= selected_limit:
            frame_ids = list(range(frame, selected_limit + 1))
        else:
            frame_ids = list(range(selected_limit, frame + 1))
        
        
        # Intersection/Union calculation
        intersection = np.sum( np.sum( np.sum( np.logical_and(auto_segmentation[frame_ids], human_segmentation[frame_ids]))))
        union = np.sum( np.sum( np.sum( np.logical_or(auto_segmentation[frame_ids], human_segmentation[frame_ids]))))
        
        # IoU calculation
        iou_score = intersection / union
        print("IoU score for slices", frame_ids[0], "to", frame_ids[-1],":", iou_score)
        
        # DICE score calculation
        dice_score = (2 * intersection) / (np.count_nonzero(auto_segmentation[frame_ids]) + np.count_nonzero(human_segmentation[frame_ids]))
        print("DICE score for slices", frame_ids[0], "to", frame_ids[-1],":", dice_score)
        
        # F1 score calculation
        # Intersection = True Positive
        # Union = True Positive + False Positive + False Negative
        f1_score = (2 * intersection) / (intersection + union)
        print("F1 score for slices", frame_ids[0], "to", frame_ids[-1],":", f1_score)
        
        #### FOR WHOLE MOVIE
        
        # Intersection/Union calculation
        intersection = np.sum( np.sum( np.sum( np.logical_and(auto_segmentation, human_segmentation))))
        union = np.sum( np.sum( np.sum( np.logical_or(auto_segmentation, human_segmentation))))
        
        # IoU calculation
        iou_score = intersection / union
        print("IoU score for whole movie: %s" % iou_score)
        
        # DICE score calculation
        dice_score = (2 * intersection) / (np.count_nonzero(auto_segmentation) + np.count_nonzero(human_segmentation))
        print("DICE score for whole movie: %s" % dice_score)
        
        # F1 score calculation
        # Intersection = True Positive
        # Union = True Positive + False Positive + False Negative
        f1_score = (2 * intersection) / (intersection + union)
        print("F1 score for whole movie: %s" % f1_score)
            
    def _evaluate_tracking(self):
        
        IOU_THRESHOLD = 0.2 # TODO: reevaluate
        
        false_positive = 0 # 1 weight (vertex deleting)
        #false_negative = 0 # 10 weight (vertex adding)
        delete_edge = 0 # 1 weight (edge deleting)
        add_edge = 0 # 1.5 weight (edge adding)
        split_cell = 0 # 5 weight (vertex splitting)
        
        current_frame = int(self.viewer.dims.point[0])
        try:
            selected_limit = int(self.le_limit_evaluation.text())
        except ValueError:
            message(title="Wrong type", text="String detected", informative_text="Please use integer instead of text")
            return
        
        if current_frame == selected_limit:
            message("Must select at least 2 frames to evaluate tracking!")
            return
        elif current_frame < selected_limit:
            frame_ids = list(range(current_frame, selected_limit + 1))
        else:
            frame_ids = list(range(selected_limit, current_frame + 1))
        
        tracks = self.viewer.layers[self.viewer.layers.index("Tracks")].data
        tracks_auto = self.tracks_old
        tracks_human = []
        segmentation = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        for line in tracks:
            tracks_human.append(np.append(line, segmentation[line[1], line[2], line[3]]))
            
        tracks_human = np.asarray(tracks_human)
        
        false_positive, false_negative, split_cell = self._evaluate_slices(frame_ids, IOU_THRESHOLD)
        
        """relevant_human_tracks = tracks_human[np.where((tracks_human[:,1] >= frame_ids[0]) & (tracks_human[:,1] <= frame_ids[-1]))]
        relevant_auto_tracks = tracks_autp[np.where((tracks_auto[:,1] >= frame_ids[0]) & (tracks_auto[:,1] <= frame_ids[-1]))]

        print(relevant_human_tracks)
        
        for frame in frame_id[0,-2]:
            for segmentation_id in relevant_
            pass"""
        
        
        
        
        
        
        print("False negatives: %s" % false_negative)
        print("False positives: %s" % false_positive)
        print("Deleted edges: %s" % delete_edge)
        print("Added edges: %s" % add_edge)
        print("Split cells: %s" % split_cell)
        
        fault_value = false_positive + false_negative * 10 + delete_edge + add_edge * 1.5 + split_cell * 5
        
        print("Fault value: %s" % fault_value)
        
    def _evaluate_slices(self, frame_ids, iou_threshold):
        
        false_positive = 0
        split_cell = 0
        
        segmentation_auto = self.segmentation_old
        segmentation_human = self.viewer.layers[self.viewer.layers.index("Segmentation Data")].data
        
        for frame in frame_ids:
            # Relevant z-frames
            frame_auto = segmentation_auto[frame]
            frame_human = segmentation_human[frame]
            
            segmentation_ids = np.unique(frame_auto)
            segmentation_ids = segmentation_ids[segmentation_ids > 0]
            
            for id in segmentation_ids:
                print("ID:", id)
                # This is the ID of the cell from the generated segmentation we are checking on
                
                ids_human = np.unique(frame_human[np.where(frame_auto == id)], return_counts = True)
                ids_human = np.asarray(ids_human)
                print("Human ids:", ids_human)
                # These are the IDs of all cells (including background) in the corrected segmentation that are overlapping the original cell 
                
                # If there is no cell in the corrected segmentation our find was a false positive
                if len(ids_human[0,ids_human[0] > 0]) == 0:
                    # FALSE POSITIVE
                    false_positive += 1
                    continue
                
                # At least one cell is overlapping the original cell
                
                if len(ids_human[0,ids_human[0] > 0]) == 1:
                    # Exactly one cell is overlapping the original cell
                    
                    if ids_human.shape[0] == 2:
                        ids_human = ids_human[:,ids_human[0] > 0].T[0]
                    print("Adjusted Human IDs:", ids_human)
                    
                    intersection = ids_human[1]
                    union = len(frame_auto[frame_auto == id]) + len(frame_human[frame_human == ids_human[0]]) - intersection
                    iou = [(ids_human[0], intersection / union)]
                    print("IoU:",iou)
                    
                    if not self._is_match(id, iou, frame_auto, frame_human):
                        false_positive += 1
                        print("False Positive for ID", id)
                    else:
                        print("Match for ID", id)
                        
                
                if len(ids_human[0,ids_human[0] > 0]) > 1:
                    # Two or more cells are overlapping the original cell
                    
                    ids_human = ids_human[:,ids_human[0] > 0]
                    ids_human = np.asarray(ids_human)
                    print("Human IDs:", ids_human, ", shape:", ids_human.shape)
                    
                    if len(ids_human[0]) == 1:
                        # SPLIT CELL (POTENTIALLY MULTIPLE)
                        split_cell += len(ids_human[0,ids_human[0] > 0]) - 1
                        print("Split Cell times", len(ids_human[0,ids_human[0] > 0]) - 1 , "for ID", id)
                        continue
                    
                    if len(ids_human[0]) > 1:
                        # At least two IDs found at old cells location
                        ious = []
                        
                        ids_human = ids_human.T
                        
                        if ids_human.shape[1] == 1:
                            intersection == ids_human[1]
                            union = len(frame_auto[frame_auto == id]) + len(frame_human[frame_human == ids_human[0]]) - intersection
                            ious.append((ids_human[0], intersection / union))
                        else:
                            for id_human in ids_human:
                                if id_human[0] == 0:
                                    continue
                                intersection = id_human[1] 
                                union = len(frame_auto[frame_auto == id]) + len(frame_human[frame_human == id_human[0]]) - intersection
                                ious.append((id_human[0], intersection / union))
                        
                    
                        split = True
                        for iou in ious:
                            if iou[1] < iou_threshold:
                                split = False
                            
                        if split:
                            # SPLIT CELL (POTENTIALLY MULTIPLE)
                            split_cell += len(ious) - 1
                            print("Split Cell times", len(ious) - 1 , "for ID", id)
                        else:
                            
                            if len(ids_human[0,ids_human[0] > 0]) == 1:
                                # It's a match :)
                                print("Match for ID", id)
                                continue
    
                            df = pd.DataFrame(ious, columns=['ID', 'IOU'])
                            df.sort_values(['IOU'], ascending=False, inplace=True)
                            ious = df.values
                            
                            if not self._is_match(id, ious, frame_auto, frame_human):
                                false_positive += 1
                                print("False Positive for ID", id)
                            else:
                                print("Match for ID", id)
            
            
            new_segmentation_ids = np.unique(segmentation_human[frame])
            false_negative = len(new_segmentation_ids[new_segmentation_ids > 0]) - len(segmentation_ids) + false_positive - split_cell
        
        return (false_positive, false_negative, split_cell)
            
            
    def _is_match(self, original_id, ious, frame_auto, frame_human):
        
        ious_array = np.asarray(ious)
        while ious_array.ndim > 1:
            print("IoUs:", ious_array, ", shape:", ious_array.shape)
            print(ious_array)
            if ious_array.shape[1] == 1:
                return ious_array[0] == original_id
            
            auto_ids = np.unique(frame_auto[np.where(frame_human == ious_array[0,0])], return_counts = True)
            
            auto_ids = np.asarray(auto_ids).T
            auto_ious = []
            
            print("Auto-IDs:", auto_ids, ", shape:", auto_ids.shape)
            
            for auto_id in auto_ids:
                intersection = auto_id[1]
                union = len(frame_auto[frame_auto == auto_id[0]]) + len(frame_human[frame_human == ious_array[0,0]]) - intersection
                auto_ious.append((auto_id[0], intersection / union))
            
            df = pd.DataFrame(auto_ious, columns=['ID', 'IOU'])
            df.sort_values(['IOU'], ascending=False, inplace=True)
            auto_ious = df.values
            
            while True:
                print("Auto-IoUs:", auto_ious, ", shape:", auto_ious.shape)
                
                if auto_ious[0,0] == original_id:
                    return True
                
                human_ids = np.unique(frame_human[np.where(frame_auto == ious_array[0,0])], return_counts = True)
                human_ids = np.asarray(human_ids).T
                print("New human IDs:", human_ids, ", shape:", human_ids.shape)
                if human_ids.shape[0] == 0:
                    return False
                human_ious = []
                for id_human in human_ids:
                    intersection = id_human[1]
                    union = len(frame_auto[frame_auto == id]) + len(frame_human[frame_human == id_human]) - intersection
                    human_ious.append((id_human[0], intersection / union))
                
                df = pd.DataFrame(human_ious, columns=['ID', 'IOU'])
                df.sort_values(['IOU'], ascending=False, inplace=True)
                human_ious = df.values
                
                if human_ious[0,0] == ious_array[0,0]:
                    ious_array = np.delete(ious_array, 0)
                    break
                
                auto_ious = np.delete(auto_ious, 0)
        
        return False
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            