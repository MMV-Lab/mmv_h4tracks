import math

import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# Scroll-zoom: fraction the visible range shrinks/grows by per wheel step
ZOOM_SCALE_PER_STEP = 1.2

# Off-screen indicator badges: fixed marker-size bounds (in points) so that a
# direction with many off-screen points never gets a visually "loud" marker.
# Size grows logarithmically between the bounds, reaching
# OFFSCREEN_BADGE_MAX_SIZE at OFFSCREEN_BADGE_REF_COUNT points.
OFFSCREEN_BADGE_MIN_SIZE = 6
OFFSCREEN_BADGE_MAX_SIZE = 14
OFFSCREEN_BADGE_REF_COUNT = 50
OFFSCREEN_BADGE_COLOR = "0.75"  # light gray: visible but unobtrusive on the dark theme

# Off-screen badges sit just outside the axes, in the padding _plot() reserves
# around the title/x-label/y-label for exactly this (see their pad/labelpad).
OFFSCREEN_BADGE_OFFSET = 0.02  # axes-fraction, from the axes edge


def compute_offscreen_counts(xys, xlim, ylim):
    """
    Count how many points lie beyond each edge of the current view.

    Parameters
    ----------
    xys : array-like
        (N, 2) array of point coordinates
    xlim, ylim : tuple
        Current axes limits

    Returns
    -------
    dict
        ``{"left": int, "right": int, "top": int, "bottom": int}``. A point
        beyond a corner (e.g. above and to the right) counts toward both
        edges it exceeds.
    """
    xys = np.asarray(xys)
    if xys.size == 0:
        return {"left": 0, "right": 0, "top": 0, "bottom": 0}
    x, y = xys[:, 0], xys[:, 1]
    return {
        "left": int(np.count_nonzero(x < xlim[0])),
        "right": int(np.count_nonzero(x > xlim[1])),
        "bottom": int(np.count_nonzero(y < ylim[0])),
        "top": int(np.count_nonzero(y > ylim[1])),
    }


def badge_size_for_count(
    count,
    min_size=OFFSCREEN_BADGE_MIN_SIZE,
    max_size=OFFSCREEN_BADGE_MAX_SIZE,
    ref_count=OFFSCREEN_BADGE_REF_COUNT,
):
    """
    Marker size for an off-screen badge.

    ``min_size`` at a single off-screen point, growing logarithmically up to
    ``max_size`` at ``ref_count`` points or more — magnitude is mostly
    conveyed by the badge's count label, not by its visual size.

    Parameters
    ----------
    count : int
        Number of off-screen points in that direction (>= 1)
    min_size, max_size : float
        Marker size bounds, in points
    ref_count : int
        Count at which the size saturates to ``max_size``

    Returns
    -------
    float
        The marker size to use
    """
    if count <= 1:
        return min_size
    fraction = math.log(count) / math.log(ref_count)
    return min_size + (max_size - min_size) * min(1.0, fraction)


class Selector:
    def __init__(self, parent, ax, results):
        """
        Parameters
        ----------
        parent : napari viewer
            The napari viewer
        ax : matplotlib axis
            The axis to draw the selector on
        results : np.ndarray
            The results to display
        """
        self.parent = parent
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.highlighted = []
        self._pan_start = None
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

        # Left-click drag: lasso selection. Scroll: zoom. Right-click drag: pan.
        self.lasso = LassoSelector(ax, onselect=self.onselect, button=1)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)

        # "Standard view" to return to via reset_view(); captured last, since
        # the caller has already applied the plot's initial axis limits by
        # the time this constructor runs.
        self._home_xlim = ax.get_xlim()
        self._home_ylim = ax.get_ylim()

        self._offscreen_artists = {}
        self.ax.callbacks.connect(
            "xlim_changed", lambda _ax: self._update_offscreen_indicators()
        )
        self.ax.callbacks.connect(
            "ylim_changed", lambda _ax: self._update_offscreen_indicators()
        )
        self._update_offscreen_indicators()

    def reset_view(self):
        """
        Restores the axes to the view that was shown when the plot was created.
        """
        self.ax.set_xlim(self._home_xlim)
        self.ax.set_ylim(self._home_ylim)
        self.canvas.draw_idle()

    def _update_offscreen_indicators(self):
        """
        Shows/updates a small arrow on each edge that currently has points
        beyond the visible view; hides the arrow for edges that don't.

        Deliberately does not touch the figure's layout (e.g. via
        ``tight_layout()``): recomputing it per call/interaction made the
        axes box (and everything anchored to it) visibly shift as tick
        labels changed width during a zoom/pan. ``_plot()`` instead reserves
        a fixed margin and pins the y-label's position once, up front, so
        arrows can rely on a stable frame that never needs to move.
        """
        counts = compute_offscreen_counts(
            self.xys, self.ax.get_xlim(), self.ax.get_ylim()
        )
        offset = OFFSCREEN_BADGE_OFFSET
        layout = {
            "top": ("^", (0.5, 1 + offset)),
            "bottom": ("v", (0.5, -offset)),
            "left": ("<", (-offset, 0.5)),
            "right": (">", (1 + offset, 0.5)),
        }
        for direction, count in counts.items():
            marker_artist = self._offscreen_artists.get(direction)
            if count == 0:
                if marker_artist is not None:
                    marker_artist.remove()
                    del self._offscreen_artists[direction]
                continue

            marker, marker_pos = layout[direction]
            size = badge_size_for_count(count)
            if marker_artist is None:
                self._offscreen_artists[direction] = self.ax.plot(
                    [marker_pos[0]],
                    [marker_pos[1]],
                    marker=marker,
                    markersize=size,
                    color=OFFSCREEN_BADGE_COLOR,
                    transform=self.ax.transAxes,
                    clip_on=False,
                    linestyle="none",
                )[0]
            else:
                marker_artist.set_markersize(size)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """
        Zooms the plot around the cursor position on mouse wheel scroll.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The scroll event
        """
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            scale_factor = 1 / ZOOM_SCALE_PER_STEP
        elif event.button == "down":
            scale_factor = ZOOM_SCALE_PER_STEP
        else:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        # Keep the point under the cursor fixed while the range around it scales
        relx = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        self.ax.set_xlim(
            event.xdata - new_width * relx, event.xdata + new_width * (1 - relx)
        )
        self.ax.set_ylim(
            event.ydata - new_height * rely, event.ydata + new_height * (1 - rely)
        )
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        """
        Starts panning on right-click.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The button press event
        """
        if event.button != 3 or event.inaxes != self.ax:
            return
        self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_motion(self, event):
        """
        Pans the plot while the right mouse button is held down.

        Uses the pixel offset from the button-press position (rather than
        ``event.xdata``/``ydata``, which are relative to the axes' current,
        already-shifted limits) so repeated moves don't compound errors.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse motion event
        """
        if self._pan_start is None:
            return
        x0, y0, xlim, ylim = self._pan_start
        bbox = self.ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return

        dx = -(event.x - x0) / bbox.width * (xlim[1] - xlim[0])
        dy = -(event.y - y0) / bbox.height * (ylim[1] - ylim[0])
        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.canvas.draw_idle()

    def _on_button_release(self, event):
        """
        Stops panning once the right mouse button is released.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The button release event
        """
        if event.button == 3:
            self._pan_start = None

    def onselect(self, vertices):
        """
        Redraws the selector with the selected vertices highlighted

        Parameters
        ----------
        vertices : np.ndarray
            The vertices of the lasso
        """
        path = Path(vertices)
        self.highlighted = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, :] = np.array([0.859375, 0.1953125, 0.125, 1])
        self.fc[self.highlighted, :] = np.array([0, 0.240802676, 0.70703125, 1])
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def apply(self):
        """
        Passes the selected tracks to the tracking window to update the tracks
        """
        widget = self.parent.parent
        if len(self.highlighted) == 0:
            selected_text = ""
            widget.tracking_window.display_cached_tracks()
        else:
            highlighted_float = self.track_ids[self.highlighted]
            highlighted_int = [int(i) for i in highlighted_float]
            selected_text = ", ".join(map(str, highlighted_int))
            widget.tracking_window.display_selected_tracks(highlighted_int)
        widget.tracking_window.lineedit_filter.setText(selected_text)
        widget.plot_window.close()
