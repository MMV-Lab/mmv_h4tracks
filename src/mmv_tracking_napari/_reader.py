from ._functions import message

def load_zarr():
    """
    Opens a dialog to select a zarr file.
    Loads the zarr file's content as layers into the viewer
    """
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
    
    self._store_segmentation()