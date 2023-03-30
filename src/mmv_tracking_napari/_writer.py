from ._functions import message

def save_zarr():
    """
    Saves the (changed) layers to a zarr file
    """
    print("that worked!")
    """# Useful if we later want to allow saving to new file
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
    message("Zarr file has been saved.")"""

def log(text, file):
    pass