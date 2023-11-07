import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class Selector:
    def __init__(self, parent, ax, results):
        self.parent = parent
        self.canvas = ax.figure.canvas
        self.track_ids = results[:, 0]
        self.collection = ax.scatter(
            results[:, 1], results[:, 2], c=np.array([[0, 0.240802676, 0.70703125, 1]])
        )
        self.xys = self.collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = self.collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        
        self.lasso = LassoSelector(ax, onselect = self.onselect, button=1)

    def onselect(self, vertices):
        path = Path(vertices)
        self.highlighted = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, :] = np.array([0.859375, 0.1953125, 0.125, 1])
        self.fc[self.highlighted, :] = np.array([0, 0.240802676, 0.70703125, 1])
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    """def disconnect(self):
        print("disconnect")
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()"""

    def apply(self):
        widget = self.parent.parent
        if self.highlighted == []:
            self.hightlighted = -1
        else:
            self.hightlighted = self.track_ids[self.highlighted]
        print("User selected tracks {}".format(self.hightlighted))
        if not hasattr(widget, "tracking_window"):
            widget._tracking()
            remove_window = True
        widget.tracking_window._replace_tracks(self.highlighted)
        if remove_window:
            widget.tracking_window.close()
        widget.plot_window.close()
