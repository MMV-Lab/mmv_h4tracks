import enum


import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QObject#, pyqtSignal <- DOESN'T WORK FOR SOME REASON, THUS EXPLICITLY IMPORTED FROM PYQT5
from qtpy.QtWidgets import QWidget

class State(enum.Enum):
    test = -1
    default =  0
    remove = 1
    recolour = 2
    merge_from = 3
    merge_to = 4
    select = 5
    link = 6
    unlink = 7
    auto_track = 8
    
class Window(QWidget):
    def __init__(self):
        super().__init__()
        
class SelectFromCollection:
    def __init__(self, parent, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.parent = parent
        
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        
        from matplotlib.widgets import LassoSelector
        self.lasso = LassoSelector(ax, onselect = self.onselect, button = 1)
        self.ind = []
        
    def onselect(self,verts):
        from matplotlib.path import Path
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, :] = np.array([.8,.2,.0,1])
        self.fc[self.ind, :] = np.array([0,.5,0,1])
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.selected_coordinates = self.xys[self.ind].data
        
    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:,-1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
    def apply(self):
        if self.ind == []:
            self.ind = -1
        if min(self.parent.tracks[:,0] > 0):
            self.ind = self.ind + 1
        self.parent._select_track(self.ind)
        self.parent.window.close()
        
class Worker(QObject):
    tracks_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    def __init__(self,label_layer,tracks):
        super().__init__()
        self.label_layer = label_layer
        self.tracks = tracks
    
    def run(self):
        i = 0
        self.new_id = max(self.tracks[:,0]) + 1
        while self.tracks[i,0] == 0: # Replace Track ID 0 as we cannot have Segmentation ID 0 (Background)
            self.tracks[i,0] = self.new_id
            i = i + 1
        df = pd.DataFrame(self.tracks, columns=['ID', 'Z', 'Y', 'X'])
        df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
        self.tracks = df.values
        self.tracks_ready.emit(self.tracks)
        
        self.label_layer.data[self.label_layer.data > 0] = self.label_layer.data[self.label_layer.data > 0] + self.new_id
        done = 0
        for track in self.tracks:
            self.label_layer.fill([track[1],track[2],track[3]],track[0])
            #print(100*done/len(self.tracks))
            self.progress.emit(100*done/len(self.tracks))
            #self.progress.update(100*done/len(self.tracks))
            done = done + 1
        self.finished.emit()
        
