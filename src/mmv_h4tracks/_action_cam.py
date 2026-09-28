"""Action Cam: a cropped, playable view centered on one tracked cell.

Not the main napari viewer - a dock tab with its own matplotlib canvas,
following the same dark-themed Figure/FigureCanvas pattern as
``AnalysisWindow._plot``. A QTimer drives play/pause with adjustable speed;
the crop follows the cell's centroid, zero-padded so it never shrinks near
image edges.
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QApplication,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIntValidator
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from skimage import measure
from skimage.segmentation import find_boundaries

from ._constants import METRIC_NAMES
from ._qt_utils import apply_napari_dark_theme, layer_as_numpy
from ._logger import notify, handle_exception

DEFAULT_CROP_RADIUS = 60
MIN_CROP_RADIUS = 10
MAX_CROP_RADIUS = 300
DEFAULT_FPS = 10
MIN_FPS = 1
MAX_FPS = 30

# Above this panel height, metrics move below the canvas instead of beside
# it, so the canvas gets the panel's full width to grow into. Below it (e.g.
# docked alongside the main widget, where height is shared and scarce),
# side-by-side already uses the available space well.
STACK_HEIGHT_THRESHOLD = 800

# Metrics whose AnalysisWindow._sort_plot_data() row needs the segmentation
# layer (size/shape metrics); everything else in METRIC_NAMES only needs the
# tracks array.
_SEGMENTATION_METRICS = ("Size", "Perimeter", "Eccentricity")


class _NoOpReporter:
    """Stand-in for a DockProgressReporter that keeps metric functions on
    their cheap sequential path (some branch on ``reporter is None`` to
    decide whether to spin up a process pool - overkill for one track)."""

    def increment(self, step: int = 1) -> None:
        pass


class ActionCamWindow(QWidget):
    """
    Tab that crops and re-centers the view on one tracked cell for every
    frame of its track, with play/pause and adjustable speed, plus a panel
    of that track's whole-track metrics.

    Parameters
    ----------
    parent : MMVH4TRACKS
        The main dock widget
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setLayout(QVBoxLayout())
        apply_napari_dark_theme(self)

        self._frames = np.empty(0, dtype=int)
        self._centroids = np.empty((0, 2))
        self._pos = 0
        self._raw = None
        self._raw_layer_name = None
        self._segmentation = None
        self._segmentation_layer_name = None
        self._image_artist = None
        self._path_artist = None
        self._outline_artist = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        # A different Image/Labels layer means the cached array is stale;
        # switching *tracks* must not force a full-volume reload (that was
        # the freeze on longer movies) - only switching *layers* does.
        self.parent.combobox_image.currentTextChanged.connect(
            self._invalidate_raw_cache
        )
        self.parent.combobox_segmentation.currentTextChanged.connect(
            self._invalidate_segmentation_cache
        )

        ### Target group
        self.lineedit_track_id = QLineEdit()
        self.lineedit_track_id.setValidator(QIntValidator(0, 2**31 - 1))
        self.lineedit_track_id.setPlaceholderText("Track ID")
        self.lineedit_track_id.returnPressed.connect(self._on_track_id_entered)
        btn_go = QPushButton("Go")
        btn_go.setToolTip("Follow the entered track ID")
        btn_go.clicked.connect(self._on_track_id_entered)

        self.btn_pick_cell = QPushButton("Pick cell in viewer")
        self.btn_pick_cell.setToolTip(
            "Click a cell in the main viewer to follow its track"
        )
        self.btn_pick_cell.clicked.connect(self._start_pick_from_viewer)

        self.spinbox_crop_radius = QSpinBox()
        self.spinbox_crop_radius.setRange(MIN_CROP_RADIUS, MAX_CROP_RADIUS)
        self.spinbox_crop_radius.setValue(DEFAULT_CROP_RADIUS)
        self.spinbox_crop_radius.setSuffix(" px")
        self.spinbox_crop_radius.valueChanged.connect(self._on_crop_radius_changed)

        self.checkbox_show_segmentation = QCheckBox("Show segmentation outline")
        self.checkbox_show_segmentation.setChecked(True)
        self.checkbox_show_segmentation.toggled.connect(self._render_current_frame)
        self.checkbox_show_track_path = QCheckBox("Show track path")
        self.checkbox_show_track_path.setChecked(True)
        self.checkbox_show_track_path.toggled.connect(self._render_current_frame)

        target_group = QGroupBox("Target")
        target_group.setLayout(QGridLayout())
        target_group.layout().addWidget(QLabel("Track ID"), 0, 0)
        target_group.layout().addWidget(self.lineedit_track_id, 0, 1)
        target_group.layout().addWidget(btn_go, 0, 2)
        target_group.layout().addWidget(self.btn_pick_cell, 1, 0, 1, 3)
        target_group.layout().addWidget(QLabel("Crop radius"), 2, 0)
        target_group.layout().addWidget(self.spinbox_crop_radius, 2, 1, 1, 2)
        target_group.layout().addWidget(self.checkbox_show_segmentation, 3, 0, 1, 3)
        target_group.layout().addWidget(self.checkbox_show_track_path, 4, 0, 1, 3)

        ### Playback group
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        # Lock the width to the wider of "Play"/"Pause" so toggling the label
        # doesn't resize the button and shift the slider's start position.
        self.btn_play_pause.setText("Pause")
        self.btn_play_pause.setMinimumWidth(self.btn_play_pause.sizeHint().width())
        self.btn_play_pause.setText("Play")

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self._on_slider_changed)

        self.label_frame = QLabel("Frame: –")

        self.spinbox_fps = QSpinBox()
        self.spinbox_fps.setRange(MIN_FPS, MAX_FPS)
        self.spinbox_fps.setValue(DEFAULT_FPS)
        self.spinbox_fps.setSuffix(" fps")
        self.spinbox_fps.valueChanged.connect(self._on_fps_changed)
        self._timer.setInterval(int(1000 / DEFAULT_FPS))

        playback_row = QWidget()
        playback_row.setLayout(QHBoxLayout())
        playback_row.layout().addWidget(self.btn_play_pause)
        playback_row.layout().addWidget(self.slider_frame, 1)
        playback_row.layout().addWidget(self.label_frame)
        playback_row.layout().addWidget(self.spinbox_fps)

        ### Canvas
        # As its own dock widget (not a tab sharing the main dock's fixed
        # height budget), this can use a comfortable natural size instead of
        # being capped.
        self._figure = Figure(figsize=(5, 4))
        self._figure.patch.set_facecolor("#262930")
        # add_axes([0,0,1,1]) instead of add_subplot(111): no axis
        # labels/ticks are shown (see below), so matplotlib's default ~10-15%
        # margins around the subplot were just wasted space - the crop
        # (imshow keeps it at 1:1/"equal" aspect, so it can't be stretched to
        # fill a non-matching container without distorting the cell) now
        # fills as much of the canvas as its aspect ratio allows.
        self._axes = self._figure.add_axes([0, 0, 1, 1])
        self._axes.set_facecolor("#262930")
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._canvas = FigureCanvas(self._figure)
        # Explicit, rather than relying on FigureCanvasQTAgg's default: grow
        # to fill whatever space it's given.
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ### Metrics group (whole-track only, refreshed on track selection)
        self.label_metrics = QLabel("Select a track to see its metrics.")
        self.label_metrics.setWordWrap(True)
        self.label_metrics.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metrics_group = QGroupBox("Track metrics")
        self.metrics_group.setLayout(QVBoxLayout())
        self.metrics_group.layout().addWidget(self.label_metrics)

        # Side-by-side (canvas | metrics) by default - reflowed to stacked
        # (canvas on top, full width; metrics below) once the panel has
        # enough height to make that worthwhile (see resizeEvent). Rebuilt
        # in place rather than pre-building both arrangements, since a
        # QWidget can only belong to one layout at a time.
        self._canvas_host = QWidget()
        self._stacked = False
        self._apply_canvas_layout(stacked=False)

        self.layout().addWidget(target_group)
        self.layout().addWidget(playback_row)
        # Stretch factor (not addStretch()) so the canvas/metrics area grows
        # into whatever space this panel is given - e.g. docked alone,
        # taking up the whole napari window - instead of staying pinned at
        # its natural compact size with the extra space left blank below it.
        self.layout().addWidget(self._canvas_host, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        stacked = self._should_stack(event.size().height())
        if stacked != self._stacked:
            self._apply_canvas_layout(stacked)

    @staticmethod
    def _should_stack(height: int) -> bool:
        """Whether there's enough panel height to move metrics below the
        canvas (full-width) rather than beside it."""
        return height >= STACK_HEIGHT_THRESHOLD

    def _apply_canvas_layout(self, stacked: bool) -> None:
        """
        (Re)build ``self._canvas_host``'s layout, moving ``self._canvas`` and
        ``self.metrics_group`` between side-by-side and stacked arrangement.

        Parameters
        ----------
        stacked : bool
            True: canvas on top (full width), metrics below.
            False (default): canvas and metrics side by side.
        """
        self._stacked = stacked
        old_layout = self._canvas_host.layout()
        # A widget can't get a new layout while it still has one; hand the
        # old layout (and, transiently, its children) to a parking widget so
        # it can be replaced, then reclaim the children into the new layout
        # below. Keeping a named reference to the parking widget (not just
        # `QWidget().setLayout(old_layout)`) matters: with no reference at
        # all, Python could garbage-collect it - and, with it, the children
        # it was just handed - before they're reclaimed a few lines down.
        parking_widget = None
        if old_layout is not None:
            parking_widget = QWidget()
            parking_widget.setLayout(old_layout)

        if stacked:
            new_layout = QVBoxLayout()
            new_layout.addWidget(self._canvas, 1)
            new_layout.addWidget(self.metrics_group)
            self.metrics_group.setSizePolicy(
                QSizePolicy.Preferred, QSizePolicy.Preferred
            )
        else:
            new_layout = QHBoxLayout()
            new_layout.addWidget(self._canvas, 1)
            new_layout.addWidget(self.metrics_group)
            # Match the canvas's height (a QGroupBox is Preferred by default,
            # which would leave it - and its dark background - short of the
            # canvas's height once the row is allowed to grow).
            self.metrics_group.setSizePolicy(
                QSizePolicy.Preferred, QSizePolicy.Expanding
            )
        self._canvas_host.setLayout(new_layout)

    ### Target selection ---------------------------------------------------

    def _invalidate_raw_cache(self, *_args):
        self._raw = None
        self._raw_layer_name = None

    def _invalidate_segmentation_cache(self, *_args):
        self._segmentation = None
        self._segmentation_layer_name = None

    def _ensure_raw_cached(self):
        """Return the raw volume, reusing the cached copy unless the selected
        Image layer changed - re-materializing it on every track pick is
        what caused the freeze on longer movies."""
        layer = self.parent.selected_image_layer()
        if self._raw is None or self._raw_layer_name != layer.name:
            self._raw = layer_as_numpy(layer)
            self._raw_layer_name = layer.name
        return self._raw

    def _ensure_segmentation_cached(self):
        """Same caching as ``_ensure_raw_cached``, for the Labels layer."""
        layer = self.parent.selected_labels_layer()
        if self._segmentation is None or self._segmentation_layer_name != layer.name:
            self._segmentation = layer_as_numpy(layer)
            self._segmentation_layer_name = layer.name
        return self._segmentation

    def _on_track_id_entered(self):
        text = self.lineedit_track_id.text().strip()
        if not text:
            return
        self._load_track(int(text))

    def _start_pick_from_viewer(self):
        """Arm a one-shot click on the main viewer to resolve its track ID."""
        try:
            self.parent.selected_labels_layer()
            self.parent.selected_tracks_layer()
        except ValueError as exc:
            notify(str(exc))
            return
        self.parent.callback_handler.add_callback_viewer(self._pick_cell_click_callback)
        QApplication.setOverrideCursor(Qt.CrossCursor)
        self.parent.set_status_text("Action Cam: click a cell to follow its track…")

    def _pick_cell_click_callback(self, _layer, event):
        """One-shot mouse callback: resolve the clicked cell's track ID and load it."""
        self.parent.callback_handler.remove_callback_viewer()
        try:
            segmentation = self._ensure_segmentation_cached()
            ndim = segmentation.ndim
            position = tuple(int(round(p)) for p in event.position[-ndim:])
            # Read directly out of the cached volume rather than
            # label_layer.get_value(position): get_value resolves against the
            # layer's *currently displayed* slice, not the frame embedded in
            # position, so it silently picks the wrong frame's data whenever
            # position's frame differs from whatever frame happens to be on
            # screen.
            try:
                clicked_label = segmentation[position]
            except IndexError:
                notify("Clicked position is outside the image.")
                return
            if clicked_label == 0:
                notify("The background can not be followed.")
                return
            frame = position[0]
            tracks = np.asarray(self.parent.selected_tracks_layer().data)
            rows_at_frame = tracks[tracks[:, 1] == frame]
            track_id = None
            for tid, _, y, x in rows_at_frame:
                if segmentation[frame, int(y), int(x)] == clicked_label:
                    track_id = int(tid)
                    break
            if track_id is None:
                notify("No track found for the clicked cell at this frame.")
                return
            self.lineedit_track_id.setText(str(track_id))
            self._load_track(track_id)
        except ValueError as exc:
            handle_exception(exc)

    def _load_track(self, track_id):
        """
        Snapshot this track's frames/centroids/whole-track metrics and reset
        playback to its start.

        Parameters
        ----------
        track_id : int
            ID from the Tracks layer to follow
        """
        self._timer.stop()
        self.btn_play_pause.setText("Play")
        try:
            tracks = np.asarray(self.parent.selected_tracks_layer().data)
        except ValueError as exc:
            notify(str(exc))
            return
        rows = tracks[tracks[:, 0] == track_id]
        if len(rows) == 0:
            notify(f"No track with ID {track_id} in the selected Tracks layer.")
            return
        rows = rows[np.argsort(rows[:, 1])]
        self._frames = rows[:, 1].astype(int)
        self._centroids = rows[:, 2:4].astype(float)

        try:
            self._ensure_raw_cached()
        except ValueError as exc:
            notify(str(exc))
        try:
            self._ensure_segmentation_cached()
        except ValueError:
            pass

        has_frames = len(self._frames) > 0
        self.btn_play_pause.setEnabled(has_frames)
        self.slider_frame.setEnabled(has_frames)
        self.slider_frame.blockSignals(True)
        self.slider_frame.setRange(0, max(0, len(self._frames) - 1))
        self.slider_frame.setValue(0)
        self.slider_frame.blockSignals(False)
        self._pos = 0
        if has_frames:
            self.label_frame.setText(
                f"Frame: {self._frames[0]} (1/{len(self._frames)})"
            )

        self._refresh_metrics_panel(rows, track_id)
        self._render_current_frame()

    ### Playback -------------------------------------------------------

    def _toggle_play_pause(self):
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play_pause.setText("Play")
            return
        if len(self._frames) < 2:
            return
        self._timer.start()
        self.btn_play_pause.setText("Pause")

    def _on_fps_changed(self, fps):
        self._timer.setInterval(int(1000 / fps))

    def _on_crop_radius_changed(self, _value):
        self._render_current_frame()

    def _advance_frame(self):
        """QTimer tick: step to the next frame the track has data for, looping."""
        if len(self._frames) == 0:
            return
        self._pos = (self._pos + 1) % len(self._frames)
        # Triggers _on_slider_changed -> _render_current_frame.
        self.slider_frame.setValue(self._pos)

    def _on_slider_changed(self, value):
        self._pos = value
        self.label_frame.setText(
            f"Frame: {self._frames[self._pos]} ({self._pos + 1}/{len(self._frames)})"
        )
        self._render_current_frame()

    ### Rendering --------------------------------------------------------

    def _current_crop_bounds(self):
        """(y0, y1, x0, x1) of the crop window around the current frame's centroid."""
        y, x = self._centroids[self._pos]
        r = self.spinbox_crop_radius.value()
        y, x = int(round(y)), int(round(x))
        return y - r, y + r, x - r, x + r

    @staticmethod
    def _cropped(array_2d, y0, y1, x0, x1):
        """Crop with zero-padding so the result is always (y1-y0, x1-x0), centered."""
        height, width = array_2d.shape[:2]
        pad_top, pad_left = max(0, -y0), max(0, -x0)
        pad_bottom, pad_right = max(0, y1 - height), max(0, x1 - width)
        crop = array_2d[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
        if pad_top or pad_bottom or pad_left or pad_right:
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)))
        return crop

    def _render_current_frame(self):
        if self._raw is None or len(self._frames) == 0:
            return
        frame = self._frames[self._pos]
        y0, y1, x0, x1 = self._current_crop_bounds()
        crop = self._cropped(self._raw[frame], y0, y1, x0, x1)

        if self._image_artist is None:
            self._image_artist = self._axes.imshow(crop, cmap="gray")
        else:
            self._image_artist.set_data(crop)
            # set_data() does not resize the image's extent to match a
            # differently-shaped array (e.g. after a crop-radius change) -
            # without this it keeps rendering at its old size, pinned to the
            # axes origin, while set_xlim/set_ylim below expand the visible
            # range around it.
            self._image_artist.set_extent((0, crop.shape[1], crop.shape[0], 0))
            vmin, vmax = crop.min(), crop.max()
            if vmax > vmin:
                self._image_artist.set_clim(vmin, vmax)
        self._axes.set_xlim(0, crop.shape[1])
        self._axes.set_ylim(crop.shape[0], 0)

        if self._outline_artist is not None:
            self._outline_artist.remove()
            self._outline_artist = None
        if self.checkbox_show_segmentation.isChecked() and self._segmentation is not None:
            self._outline_artist = self._draw_segmentation_outline(
                frame, y0, y1, x0, x1
            )

        if self._path_artist is not None:
            self._path_artist.remove()
            self._path_artist = None
        if self.checkbox_show_track_path.isChecked():
            path = self._centroids[: self._pos + 1]
            local_y = path[:, 0] - y0
            local_x = path[:, 1] - x0
            (self._path_artist,) = self._axes.plot(
                local_x, local_y, color="#4a90d9", linewidth=1
            )

        self._canvas.draw_idle()

    def _draw_segmentation_outline(self, frame, y0, y1, x0, x1):
        """Outline of the followed cell's own mask, or None if it has none here."""
        seg_crop = self._cropped(self._segmentation[frame], y0, y1, x0, x1)
        y, x = self._centroids[self._pos]
        local_y, local_x = int(round(y)) - y0, int(round(x)) - x0
        if not (0 <= local_y < seg_crop.shape[0] and 0 <= local_x < seg_crop.shape[1]):
            return None
        cell_id = seg_crop[local_y, local_x]
        if cell_id == 0:
            return None
        boundary_mask = find_boundaries(seg_crop == cell_id, mode="outer")
        ys, xs = np.nonzero(boundary_mask)
        if len(xs) == 0:
            return None
        return self._axes.scatter(xs, ys, s=4, c="#ff4d4d")

    ### Metrics ----------------------------------------------------------

    # Perimeter/Eccentricity via AnalysisWindow run skimage.measure.regionprops
    # on the *whole frame*, once per metric, once per frame the track spans -
    # designed for scoring every track in one shared pass (Analysis tab), but
    # measured at ~0.8-1.4s each on a 400+ frame movie when done per single
    # track pick here. Computed locally instead, on the same small crop
    # already used for display, sharing one regionprops call between both
    # metrics (they only depend on the region's shape, so cropping tightly
    # around it changes nothing numerically as long as the crop contains the
    # whole cell - the same assumption the visible crop already makes).
    _LOCAL_SHAPE_METRICS = ("Perimeter", "Eccentricity")

    def _fast_local_shape_stats(self, track_rows):
        """
        Perimeter/eccentricity mean+std across the track, computed on a tight
        local crop per frame instead of the full frame.

        Returns
        -------
        dict
            ``{"Perimeter": (avg, std) | None, "Eccentricity": (avg, std) | None}``
            (``None`` for a metric with no measurable frames)
        """
        r = self.spinbox_crop_radius.value()
        perimeters, eccentricities = [], []
        for _tid, frame, y, x in track_rows:
            frame, y, x = int(frame), int(y), int(x)
            seg_id = self._segmentation[frame, y, x]
            if seg_id == 0:
                continue
            local = self._cropped(
                self._segmentation[frame], y - r, y + r, x - r, x + r
            )
            props = measure.regionprops((local == seg_id).astype(int))
            if not props:
                continue
            perimeters.append(props[0].perimeter)
            eccentricities.append(props[0].eccentricity)

        def summarize(values):
            if not values:
                return None
            return (
                float(np.around(np.average(values), 3)),
                float(np.around(np.std(values), 3)),
            )

        return {
            "Perimeter": summarize(perimeters),
            "Eccentricity": summarize(eccentricities),
        }

    def _refresh_metrics_panel(self, track_rows, track_id):
        """
        Whole-track metrics only (no per-frame values), reusing
        AnalysisWindow._sort_plot_data - "frame-based" metrics (Speed, Size,
        Perimeter, Eccentricity) already return mean/std across the track's
        frames; the rest are inherently single whole-track values.
        """
        analysis_window = self.parent.analysis_window
        lines = [f"Track {track_id} — {len(track_rows)} frame(s)"]
        local_shape_stats = None
        for name in METRIC_NAMES:
            needs_segmentation = name in _SEGMENTATION_METRICS
            if needs_segmentation and self._segmentation is None:
                lines.append(f"{name}: unavailable (no segmentation layer selected)")
                continue
            try:
                if name in self._LOCAL_SHAPE_METRICS:
                    if local_shape_stats is None:
                        local_shape_stats = self._fast_local_shape_stats(track_rows)
                    stat = local_shape_stats[name]
                    if stat is None:
                        lines.append(f"{name}: n/a")
                    else:
                        lines.append(f"{name}: {stat[0]:.2f} ± {stat[1]:.2f}")
                    continue
                result = analysis_window._sort_plot_data(
                    name,
                    track_rows,
                    self._segmentation if needs_segmentation else None,
                    _NoOpReporter(),
                )
                row = result["Results"]
            except Exception as exc:
                lines.append(f"{name}: error ({exc})")
                continue
            if row.shape[0] == 0:
                lines.append(f"{name}: n/a")
                continue
            lines.append(self._format_metric_line(name, row[0]))
        self.label_metrics.setText("\n".join(lines))

    @staticmethod
    def _format_metric_line(name, row):
        if name in ("Speed", "Perimeter", "Eccentricity"):
            return f"{name}: {row[1]:.2f} ± {row[2]:.2f}"
        if name == "Size":
            return (
                f"{name}: {row[1]:.2f} ± {row[2]:.2f} "
                f"(min {row[3]:.0f}, max {row[4]:.0f})"
            )
        if name == "Direction":
            return f"{name}: {row[3]:.1f}°"
        if name in ("Euclidean distance", "Accumulated distance"):
            return f"{name}: {row[1]:.2f} px"
        if name == "Velocity":
            return f"{name}: {row[1]:.2f} px/frame"
        if name == "Track duration":
            return f"{name}: {int(row[1])} frame(s)"
        return f"{name}: {row}"
