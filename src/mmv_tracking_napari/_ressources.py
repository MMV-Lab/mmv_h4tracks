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
    def __init__(self, parent, ax, collection, track_ids, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.parent = parent
        self.track_ids = track_ids
        
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
        self.fc[:, :] = np.array([0.859375,0.1953125,0.125,1]) # .8,.2,.0,1
        self.fc[self.ind, :] = np.array([0,0.240802676,0.70703125,1]) # 0,.5,0,1
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
        else:
            self.ind = self.track_ids[self.ind]
        self.parent._select_track(self.ind)
        self.parent.window.close()
        
class Worker(QObject):
    tracks_ready = pyqtSignal(np.ndarray)
    starting = pyqtSignal()
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
        self.progress.emit(0)
        for i in range(len(self.label_layer.data[:])):
            tracks_in_layer = self.tracks[self.tracks[:,1]==i]
            for entry in tracks_in_layer:
                centroid = entry[2:4]
                self.label_layer.fill([i,centroid[0],centroid[1]],entry[0] + self.new_id)
            self.progress.emit(100*(i+0.5)/len(self.label_layer.data[:]))
            if len(np.unique(self.label_layer.data[i])) - 1 == len(np.unique(tracks_in_layer[:,0])):
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([i,centroid[0],centroid[1]],entry[0])
            else:
                ids = np.unique(self.label_layer.data[i])
                untracked = np.where(ids < self.new_id,ids,0)
                untracked = untracked[untracked != 0]
                for entry in untracked:
                    self.label_layer.data[i] = np.where(self.label_layer.data[i] == entry,self.label_layer.data[i] + self.new_id,self.label_layer.data[i])
                untracked = untracked + self.new_id
                
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([i,centroid[0],centroid[1]],entry[0])
                #self.label_layer.data[i] = np.where(self.label_layer.data[i] > 0,self.label_layer.data[i] + self.new_id,self.label_layer.data[i])
            self.progress.emit(100*(i+1)/len(self.label_layer.data[:]))
                
        
        """self.label_layer.data[self.label_layer.data > 0] = self.label_layer.data[self.label_layer.data > 0] + self.new_id
        done = 0
        for track in self.tracks:
            self.label_layer.fill([track[1],track[2],track[3]],track[0])
            #print(100*done/len(self.tracks))
            self.progress.emit(100*done/len(self.tracks))
            #self.progress.update(100*done/len(self.tracks))
            done = done + 1"""
        self.finished.emit()
        
class AdjustTracksWorker(QObject):
    tracks_ready = pyqtSignal(np.ndarray)
    starting = pyqtSignal()
    finished = pyqtSignal()
    status = pyqtSignal(str)
    progress = pyqtSignal(float)
    def __init__(self,parent,tracks):
        super().__init__()
        self.parent = parent
        self.tracks = tracks
    
    def run(self):
        self.starting.emit()
        self.status.emit("Adjusting Tracks")
        print("Adjusting Tracks")
        i = 0
        self.new_id = max(self.tracks[:,0]) + 1
        self.parent.new_id = self.new_id
        while self.tracks[i,0] == 0: # Replace Track ID 0 as we cannot have Segmentation ID 0 (Background)
            self.tracks[i,0] = self.new_id
            i = i + 1
            self.progress.emit(i/len(self.tracks)*100)
        df = pd.DataFrame(self.tracks, columns=['ID', 'Z', 'Y', 'X'])
        df.sort_values(['ID', 'Z'], ascending=True, inplace=True)
        self.tracks = df.values
        self.tracks_ready.emit(self.tracks)
        self.status.emit("Done")
        self.finished.emit()
        
class AdjustSegWorker(QObject):
    starting = pyqtSignal()
    finished = pyqtSignal()
    status = pyqtSignal(str)
    #progress = pyqtSignal(tuple)
    update_progress = pyqtSignal()
    def __init__(self,parent,label_layer,tracks,counter,completed):
        super().__init__()
        self.label_layer = label_layer
        self.tracks = tracks
        self.parent = parent
        self.counter = counter
        self.completed = completed
    
    def run(self):
        
        self.starting.emit()
        self.status.emit("Adjusting Segmentation IDs")
        self.new_id = self.parent.new_id
        slice_id = self.counter.getSlice()
        self.update_progress.emit()
        while slice_id < len(self.label_layer.data):
            print("Adjusting Slice " + str(slice_id))
            tracks_in_layer = self.tracks[self.tracks[:,1]==slice_id]
            for entry in tracks_in_layer:
                centroid = entry[2:4]
                self.label_layer.fill([slice_id,centroid[0],centroid[1]],entry[0] + self.new_id)
            if len(np.unique(self.label_layer.data[slice_id])) - 1 == len(np.unique(tracks_in_layer[:,0])):
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([slice_id,centroid[0],centroid[1]],entry[0])
            else:
                ids = np.unique(self.label_layer.data[slice_id])
                untracked = np.where(ids < self.new_id,ids,0)
                untracked = untracked[untracked != 0]
                for entry in untracked:
                    self.label_layer.data[slice_id] = np.where(self.label_layer.data[slice_id] == entry,self.label_layer.data[slice_id] + self.new_id,self.label_layer.data[slice_id])
                untracked = untracked + self.new_id
                
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([slice_id,centroid[0],centroid[1]],entry[0])
            self.completed.increment()
            self.update_progress.emit()
            slice_id = self.counter.getSlice()
        
        self.status.emit("Done")
        self.finished.emit()
        
        
        """self.starting.emit()
        self.status.emit(("Adjusting Segmentation IDs",self.worker_id-1))
        self.progress.emit((0,self.worker_id-1))
        self.new_id = self.parent.new_id
        
        
        for i in range(self.start,self.end):
            print("Thread " + str(self.worker_id) + ": Adjusting Slice " + str(i))
        #for i in range(len(self.label_layer.data[:])):
            tracks_in_layer = self.tracks[self.tracks[:,1]==i]
            for entry in tracks_in_layer:
                centroid = entry[2:4]
                self.label_layer.fill([i,centroid[0],centroid[1]],entry[0] + self.new_id)
            self.progress.emit((100*(i-self.start+0.5)/(self.end-self.start),self.worker_id-1))   #len(self.label_layer.data[:]))
            if len(np.unique(self.label_layer.data[i])) - 1 == len(np.unique(tracks_in_layer[:,0])):
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([i,centroid[0],centroid[1]],entry[0])
            else:
                ids = np.unique(self.label_layer.data[i])
                untracked = np.where(ids < self.new_id,ids,0)
                untracked = untracked[untracked != 0]
                for entry in untracked:
                    self.label_layer.data[i] = np.where(self.label_layer.data[i] == entry,self.label_layer.data[i] + self.new_id,self.label_layer.data[i])
                untracked = untracked + self.new_id
                
                for entry in tracks_in_layer:
                    centroid = entry[2:4]
                    self.label_layer.fill([i,centroid[0],centroid[1]],entry[0])
                #self.label_layer.data[i] = np.where(self.label_layer.data[i] > 0,self.label_layer.data[i] + self.new_id,self.label_layer.data[i])
            self.progress.emit((100*(i-self.start+1)/(self.end-self.start),self.worker_id-1))   #len(self.label_layer.data[:]))"""
                
        
        """self.label_layer.data[self.label_layer.data > 0] = self.label_layer.data[self.label_layer.data > 0] + self.new_id
        done = 0
        for track in self.tracks:
            self.label_layer.fill([track[1],track[2],track[3]],track[0])
            #print(100*done/len(self.tracks))
            self.progress.emit(100*done/len(self.tracks))
            #self.progress.update(100*done/len(self.tracks))
            done = done + 1
        self.status.emit(("Done",self.worker_id-1))
        self.finished.emit()"""
        
class SizeWorker(QObject):
    starting = pyqtSignal()
    finished = pyqtSignal()
    status = pyqtSignal(str)
    update_progress = pyqtSignal()
    
    def __init__(self):
        super.__init__()
        pass
    
    def run(self):
        self.starting.emit()
        self.status.emit("Calculating Size")
        
        
        self.status.emit("Done")
        self.finished.emit()
        
class GenericWorker(QObject):
    value = pyqtSignal(object)
    starting = pyqtSignal()
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    def __init__(self, function, *args):
        super().__init__()
        self.function = function
        self.args = args
    
    def run(self):
        self.starting.emit()
        self.value.emit(self.function(self.args))
        self.finished.emit()
        

from threading import Lock
class SliceCounter():
    def __init__(self):
        self._counter = - 1
        self._lock = Lock()
    
    def getSlice(self):
        with self._lock:
            self._counter += 1
            return self._counter
        
class ThreadSafeCounter():
    def __init__(self):
        self._counter = 0
        self._lock = Lock()
        
    def increment(self):
        with self._lock:
            self._counter += 1
            
    def value(self):
        with self._lock:
            return self._counter
    
    