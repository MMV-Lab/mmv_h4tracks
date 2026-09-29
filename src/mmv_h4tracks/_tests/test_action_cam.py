"""Tests for the Action Cam tab."""

from types import SimpleNamespace

import numpy as np
import pytest

import mmv_h4tracks._action_cam as action_cam_module

FRAME_SHAPE = (100, 100)
N_FRAMES = 6


@pytest.fixture
def action_cam_window(create_widget):
    """The Action Cam dock widget, opened on a clean main widget with
    synthetic layers loaded."""
    widget = create_widget
    viewer = widget.viewer

    raw = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.uint16)
    segmentation = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.int32)
    for frame in range(N_FRAMES):
        # Track id 1 moves along the diagonal; present on every frame except 2
        # (a gap) so tests can verify gap handling.
        if frame != 2:
            cy, cx = 10 + frame * 5, 10 + frame * 5
            segmentation[frame, cy - 3 : cy + 3, cx - 3 : cx + 3] = 1
        # Track id 2 stays still, present on every frame.
        segmentation[frame, 80:86, 80:86] = 2
    raw[segmentation > 0] = 100

    # Track 1 has a gap at frame 2 (id, frame, y, x) - centroids matching the
    # segmentation blobs above.
    track_1_frames = [f for f in range(N_FRAMES) if f != 2]
    tracks_data = [
        [1, f, 10 + f * 5, 10 + f * 5] for f in track_1_frames
    ] + [[2, f, 83, 83] for f in range(N_FRAMES)]
    tracks = np.array(tracks_data, dtype=int)

    viewer.add_image(raw, name="raw")
    viewer.add_labels(segmentation, name="segmentation")
    viewer.add_tracks(tracks, name="tracks")
    widget.combobox_image.setCurrentText("raw")
    widget.combobox_segmentation.setCurrentText("segmentation")
    widget.combobox_tracks.setCurrentText("tracks")

    widget.open_action_cam()
    return widget.action_cam_window


def test_load_track_follows_only_its_own_frames_across_a_gap(action_cam_window):
    """Track 1 skips frame 2; the player must only ever land on frames it has data for."""
    action_cam_window._load_track(1)

    assert list(action_cam_window._frames) == [0, 1, 3, 4, 5]
    assert action_cam_window.slider_frame.maximum() == 4
    expected_centroids = np.array(
        [[10 + f * 5, 10 + f * 5] for f in [0, 1, 3, 4, 5]], dtype=float
    )
    assert np.array_equal(action_cam_window._centroids, expected_centroids)


def test_load_track_resets_playback_to_the_start(action_cam_window):
    action_cam_window._load_track(1)
    action_cam_window.slider_frame.setValue(3)
    assert action_cam_window._pos == 3

    action_cam_window._load_track(2)
    assert action_cam_window._pos == 0
    assert action_cam_window.slider_frame.value() == 0


def test_entering_a_track_id_loads_it(action_cam_window):
    action_cam_window.lineedit_track_id.setText("2")
    action_cam_window._on_track_id_entered()

    assert list(action_cam_window._frames) == list(range(N_FRAMES))


def test_entering_an_unknown_track_id_leaves_current_view_untouched(
    action_cam_window, monkeypatch
):
    action_cam_window._load_track(1)
    frames_before = list(action_cam_window._frames)

    notified = []
    monkeypatch.setattr(action_cam_module, "notify", notified.append)
    action_cam_window.lineedit_track_id.setText("999")
    action_cam_window._on_track_id_entered()

    assert list(action_cam_window._frames) == frames_before
    assert notified and "999" in notified[0]


def test_pick_cell_in_viewer_resolves_track_id_and_loads_it(action_cam_window):
    """Simulates a click at frame 3, position (25, 25) - inside track 1's cell -
    without needing a real Qt mouse event.

    Regression: the viewer's current step defaults to frame 0, different from
    the frame 3 being clicked here. Resolving via
    ``label_layer.get_value(position)`` silently reads frame 0's data instead
    of the frame embedded in ``position`` (it resolves against whatever
    frame is *currently displayed*, not the click's own frame), so this
    would incorrectly resolve to background/no track if that regressed.
    """
    # action_cam_window is a single instance reused across this file's tests
    # (module-scoped widget); start from a known lineedit state rather than
    # assuming it's still empty from construction.
    action_cam_window.lineedit_track_id.setText("")
    # Force a mismatch with the clicked frame (3), rather than assuming the
    # viewer is still at its default step - see the regression note above.
    action_cam_window.parent.viewer.dims.set_point(0, 0)
    fake_event = SimpleNamespace(position=(3, 25, 25))

    action_cam_window._pick_cell_click_callback(None, fake_event)

    assert action_cam_window.lineedit_track_id.text() == "1"
    assert list(action_cam_window._frames) == [0, 1, 3, 4, 5]


def test_pick_cell_on_background_does_not_load_a_track(action_cam_window, monkeypatch):
    notified = []
    monkeypatch.setattr(action_cam_module, "notify", notified.append)
    # Known sentinel (not "" - see above) so the assertion proves the
    # callback left it untouched, rather than assuming a default value.
    action_cam_window.lineedit_track_id.setText("sentinel")
    fake_event = SimpleNamespace(position=(0, 0, 0))  # background

    action_cam_window._pick_cell_click_callback(None, fake_event)

    assert action_cam_window.lineedit_track_id.text() == "sentinel"
    assert notified and "background" in notified[0].lower()


def test_crop_is_always_full_size_and_zero_padded_near_edges(action_cam_window):
    array = np.arange(25).reshape(5, 5)

    # Fully inside: no padding, exact slice.
    crop = action_cam_window._cropped(array, 1, 4, 1, 4)
    assert crop.shape == (3, 3)
    assert np.array_equal(crop, array[1:4, 1:4])

    # Straddling the top-left corner: still full requested size, zero-padded.
    crop = action_cam_window._cropped(array, -2, 2, -2, 2)
    assert crop.shape == (4, 4)
    assert np.array_equal(crop[2:, 2:], array[0:2, 0:2])
    assert np.array_equal(crop[:2, :], np.zeros((2, 4)))
    assert np.array_equal(crop[:, :2], np.zeros((4, 2)))

    # Straddling the bottom-right corner.
    crop = action_cam_window._cropped(array, 3, 7, 3, 7)
    assert crop.shape == (4, 4)
    assert np.array_equal(crop[:2, :2], array[3:5, 3:5])


def test_changing_crop_radius_keeps_the_image_filling_the_axes(action_cam_window):
    """Regression: increasing the crop radius used to leave the image pinned
    at its old (smaller) size in the corner, because set_data() alone does
    not resize an AxesImage's extent to match the new array shape."""
    action_cam_window._load_track(2)

    action_cam_window.spinbox_crop_radius.setValue(120)
    extent = tuple(action_cam_window._image_artist.get_extent())
    xlim = tuple(action_cam_window._axes.get_xlim())
    ylim = tuple(action_cam_window._axes.get_ylim())

    assert extent == (0, 240, 240, 0)
    assert xlim == (0, 240)
    assert ylim == (240, 0)


def test_metrics_panel_reuses_analysis_window_math(action_cam_window):
    """The displayed Speed/Track duration values must match calling
    AnalysisWindow._sort_plot_data directly on the same filtered track - no
    separate reimplementation of the metric math."""
    action_cam_window._load_track(2)

    analysis_window = action_cam_window.parent.analysis_window
    tracks = np.asarray(action_cam_window.parent.selected_tracks_layer().data)
    track_2 = tracks[tracks[:, 0] == 2]

    speed_row = analysis_window._sort_plot_data(
        "Speed", track_2, None, None
    )["Results"][0]
    duration_row = analysis_window._sort_plot_data(
        "Track duration", track_2, None, None
    )["Results"][0]

    text = action_cam_window.label_metrics.text()
    assert f"{speed_row[1]:.2f}" in text
    assert f"{speed_row[2]:.2f}" in text
    assert f"{int(duration_row[1])} frame(s)" in text


def test_perimeter_and_eccentricity_match_full_frame_computation(action_cam_window):
    """The crop-local fast path must give the same numbers as
    AnalysisWindow's full-frame regionprops computation, since the crop
    (spinbox_crop_radius) fully contains the (tiny, synthetic) cell."""
    action_cam_window._load_track(2)

    analysis_window = action_cam_window.parent.analysis_window
    tracks = np.asarray(action_cam_window.parent.selected_tracks_layer().data)
    track_2 = tracks[tracks[:, 0] == 2]
    segmentation = action_cam_window._segmentation

    perimeter_row = analysis_window._sort_plot_data(
        "Perimeter", track_2, segmentation, action_cam_module._NoOpReporter()
    )["Results"][0]
    eccentricity_row = analysis_window._sort_plot_data(
        "Eccentricity", track_2, segmentation, action_cam_module._NoOpReporter()
    )["Results"][0]

    text = action_cam_window.label_metrics.text()
    assert f"Perimeter: {perimeter_row[1]:.2f} ± {perimeter_row[2]:.2f}" in text
    assert (
        f"Eccentricity: {eccentricity_row[1]:.2f} ± {eccentricity_row[2]:.2f}"
        in text
    )


def test_perimeter_and_eccentricity_avoid_the_slow_full_frame_path(
    action_cam_window, monkeypatch
):
    """Regression: these two metrics used to run regionprops on the whole
    frame once per metric per frame (measured at ~0.8-1.4s each on a 400+
    frame movie); they must go through the fast crop-local path instead."""
    analysis_window = action_cam_window.parent.analysis_window

    def fail(*args, **kwargs):
        raise AssertionError("must not use the full-frame metric path")

    monkeypatch.setattr(analysis_window, "calculate_cell_perimeter", fail)
    monkeypatch.setattr(analysis_window, "calculate_cell_eccentricity", fail)

    action_cam_window._load_track(2)  # must not raise

    text = action_cam_window.label_metrics.text()
    assert "Perimeter:" in text
    assert "Eccentricity:" in text


def test_shape_stats_share_one_regionprops_call_per_frame(action_cam_window, monkeypatch):
    """Perimeter and Eccentricity must share one regionprops call per frame,
    not one each (that duplicated the expensive connected-component analysis)."""
    calls = []
    original = action_cam_module.measure.regionprops

    def counting_regionprops(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(action_cam_module.measure, "regionprops", counting_regionprops)

    action_cam_window._load_track(2)  # present on all N_FRAMES frames

    assert len(calls) == N_FRAMES


def test_metrics_panel_reports_segmentation_metrics_unavailable_without_labels(
    action_cam_window, monkeypatch
):
    """No Labels layer selected (grab_layer's contract: raises ValueError)."""

    def _raise(*args, **kwargs):
        raise ValueError("Layer name can not be blank")

    monkeypatch.setattr(action_cam_window.parent, "selected_labels_layer", _raise)
    action_cam_window._load_track(2)

    text = action_cam_window.label_metrics.text()
    assert "Size: unavailable" in text
    assert "Perimeter: unavailable" in text
    assert "Eccentricity: unavailable" in text


def test_play_pause_toggles_the_timer(action_cam_window):
    action_cam_window._load_track(2)
    assert not action_cam_window._timer.isActive()

    action_cam_window._toggle_play_pause()
    assert action_cam_window._timer.isActive()
    assert action_cam_window.btn_play_pause.text() == "Pause"

    action_cam_window._toggle_play_pause()
    assert not action_cam_window._timer.isActive()
    assert action_cam_window.btn_play_pause.text() == "Play"


def test_play_pause_button_width_is_stable_across_states(action_cam_window):
    """Regression: toggling the Play/Pause label used to resize the button
    and shift the slider's start position."""
    btn = action_cam_window.btn_play_pause
    min_width = btn.minimumWidth()
    assert min_width > 0

    btn.setText("Play")
    play_hint = btn.sizeHint().width()
    btn.setText("Pause")
    pause_hint = btn.sizeHint().width()

    assert min_width >= max(play_hint, pause_hint)


def test_should_stack_threshold(action_cam_window):
    from mmv_h4tracks._action_cam import STACK_HEIGHT_THRESHOLD

    assert action_cam_window._should_stack(STACK_HEIGHT_THRESHOLD - 1) is False
    assert action_cam_window._should_stack(STACK_HEIGHT_THRESHOLD) is True


def test_apply_canvas_layout_switches_arrangement(action_cam_window):
    from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout

    action_cam_window._apply_canvas_layout(stacked=False)
    assert isinstance(action_cam_window._canvas_host.layout(), QHBoxLayout)

    action_cam_window._apply_canvas_layout(stacked=True)
    assert isinstance(action_cam_window._canvas_host.layout(), QVBoxLayout)
    assert action_cam_window._canvas_host.layout().count() == 2


def test_apply_canvas_layout_survives_repeated_toggling(action_cam_window):
    """Regression: swapping the canvas/metrics layout used to risk garbage-
    collecting an unreferenced parking widget before its children (the
    canvas and metrics group) were reclaimed into the new layout, which
    would delete those children's underlying Qt objects too."""
    for _ in range(4):
        action_cam_window._apply_canvas_layout(stacked=True)
        action_cam_window._apply_canvas_layout(stacked=False)

    host = action_cam_window._canvas_host
    assert action_cam_window._canvas.parent() is host
    assert action_cam_window.metrics_group.parent() is host
    # Both widgets must still be alive and usable, not deleted C++ objects.
    action_cam_window._load_track(2)
    assert "Track 2" in action_cam_window.label_metrics.text()


def test_resize_event_reflows_only_on_threshold_crossing(action_cam_window):
    from qtpy.QtCore import QSize
    from qtpy.QtGui import QResizeEvent

    from mmv_h4tracks._action_cam import STACK_HEIGHT_THRESHOLD

    # Start from a known state - the widget instance is reused across tests
    # in this file, so its _stacked flag may be left over from another test.
    action_cam_window._apply_canvas_layout(stacked=False)

    event = QResizeEvent(
        QSize(400, STACK_HEIGHT_THRESHOLD + 50), QSize(400, 300)
    )
    action_cam_window.resizeEvent(event)
    assert action_cam_window._stacked is True

    same_height_event = QResizeEvent(
        QSize(500, STACK_HEIGHT_THRESHOLD + 50),
        QSize(400, STACK_HEIGHT_THRESHOLD + 50),
    )
    action_cam_window.resizeEvent(same_height_event)
    assert action_cam_window._stacked is True  # unchanged, no rebuild needed

    shrink_event = QResizeEvent(
        QSize(400, 200), QSize(500, STACK_HEIGHT_THRESHOLD + 50)
    )
    action_cam_window.resizeEvent(shrink_event)
    assert action_cam_window._stacked is False


def test_fps_change_updates_timer_interval(action_cam_window):
    action_cam_window.spinbox_fps.setValue(10)
    assert action_cam_window._timer.interval() == 100

    action_cam_window.spinbox_fps.setValue(5)
    assert action_cam_window._timer.interval() == 200


def test_advance_frame_wraps_around(action_cam_window):
    action_cam_window._load_track(1)
    n = len(action_cam_window._frames)

    for _ in range(n):
        action_cam_window._advance_frame()
    assert action_cam_window._pos == 0

    action_cam_window._advance_frame()
    assert action_cam_window._pos == 1


def test_switching_tracks_does_not_reload_the_raw_volume(action_cam_window, monkeypatch):
    """Regression: selecting a new track used to re-materialize the whole
    raw/segmentation volume every time, freezing the UI on long movies."""
    calls = []
    original = action_cam_module.layer_as_numpy

    def counting_layer_as_numpy(layer):
        calls.append(layer.name)
        return original(layer)

    monkeypatch.setattr(action_cam_module, "layer_as_numpy", counting_layer_as_numpy)

    action_cam_window._load_track(1)
    action_cam_window._load_track(2)
    action_cam_window._load_track(1)

    # One materialization per layer (raw, segmentation), not per track pick.
    assert calls.count("raw") == 1
    assert calls.count("segmentation") == 1


def test_changing_the_image_layer_invalidates_the_cache(action_cam_window, monkeypatch):
    widget = action_cam_window.parent
    second_raw = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.uint16)
    widget.viewer.add_image(second_raw, name="raw2")

    calls = []
    original = action_cam_module.layer_as_numpy

    def counting_layer_as_numpy(layer):
        calls.append(layer.name)
        return original(layer)

    monkeypatch.setattr(action_cam_module, "layer_as_numpy", counting_layer_as_numpy)

    action_cam_window._load_track(1)
    widget.combobox_image.setCurrentText("raw2")
    action_cam_window._load_track(2)

    assert calls.count("raw") == 1
    assert calls.count("raw2") == 1


def test_overlay_checkboxes_default_to_off(action_cam_window):
    assert action_cam_window.checkbox_show_segmentation.isChecked() is False
    assert action_cam_window.checkbox_show_track_path.isChecked() is False


def test_play_button_is_highlighted(action_cam_window):
    """The Play/Pause button is the panel's primary control and should stand
    out visually from the plain buttons around it."""
    assert action_cam_window.btn_play_pause.styleSheet() != ""
    other_buttons = [
        action_cam_window.btn_pick_cell,
        action_cam_window.btn_export,
    ]
    for button in other_buttons:
        assert button.styleSheet() != action_cam_window.btn_play_pause.styleSheet()


def test_render_frames_for_export_matches_the_track_and_restores_position(
    action_cam_window,
):
    """No overlays active (the default) - single-channel, not touched at all."""
    action_cam_window._load_track(1)
    action_cam_window.slider_frame.setValue(2)  # not frame 0
    assert action_cam_window._overlays_active() is False

    frames = action_cam_window._render_frames_for_export()

    assert len(frames) == len(action_cam_window._frames)
    for frame in frames:
        assert frame.ndim == 2
        assert frame.dtype == action_cam_window._raw.dtype
    # Live view must be left exactly where it was.
    assert action_cam_window._pos == 2


def test_render_frames_for_export_uses_rgb_only_when_an_overlay_is_active(
    action_cam_window,
):
    """Color is only needed once an overlay is drawn on top of the crop."""
    action_cam_window._load_track(1)
    action_cam_window.checkbox_show_track_path.setChecked(True)
    try:
        frames = action_cam_window._render_frames_for_export()
        assert len(frames) == len(action_cam_window._frames)
        for frame in frames:
            assert frame.ndim == 3 and frame.shape[2] == 3
            assert frame.dtype == np.uint8
    finally:
        action_cam_window.checkbox_show_track_path.setChecked(False)


def test_render_frames_for_export_has_no_letterbox_padding(action_cam_window):
    """Regression: capturing the interactive canvas directly exported
    whatever aspect ratio the panel happened to have on screen, letterboxing
    the (always square) crop with grey bars on the sides. Exported frames
    must be exactly crop-sized, not panel-sized."""
    action_cam_window._load_track(1)
    radius = action_cam_window.spinbox_crop_radius.value()
    expected_side = 2 * radius

    frames = action_cam_window._render_frames_for_export()

    for frame in frames:
        assert abs(frame.shape[0] - expected_side) <= 1
        assert abs(frame.shape[1] - expected_side) <= 1


def test_write_tiff_stack_round_trips(action_cam_window, tmp_path):
    action_cam_window._load_track(2)
    frames = action_cam_window._render_frames_for_export()

    path = tmp_path / "clip.tiff"
    action_cam_window._write_tiff_stack(path, frames)

    assert path.is_file()
    import tifffile

    stack = tifffile.imread(str(path))
    assert stack.shape[0] == len(frames)


def test_write_tiff_stack_preserves_native_dtype_without_overlays(
    action_cam_window, tmp_path
):
    """No color needed -> no reason to lose bit depth either."""
    action_cam_window._load_track(2)
    frames = action_cam_window._render_frames_for_export()
    assert frames[0].dtype == np.uint16  # the fixture's raw array dtype

    path = tmp_path / "clip.tiff"
    action_cam_window._write_tiff_stack(path, frames)

    import tifffile

    assert tifffile.imread(str(path)).dtype == np.uint16


def test_write_tiff_stack_compresses_the_raw_only_case(action_cam_window, tmp_path):
    action_cam_window._load_track(2)
    frames = action_cam_window._render_frames_for_export()
    assert frames[0].ndim == 2  # no overlays -> raw-only path

    path = tmp_path / "clip.tiff"
    action_cam_window._write_tiff_stack(path, frames)

    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        assert tif.pages[0].compression.name.lower() in ("adobe_deflate", "deflate")


def test_write_tiff_stack_does_not_compress_the_overlay_case(
    action_cam_window, tmp_path
):
    action_cam_window._load_track(2)
    action_cam_window.checkbox_show_track_path.setChecked(True)
    try:
        frames = action_cam_window._render_frames_for_export()
        assert frames[0].ndim == 3  # overlay active -> RGB path

        path = tmp_path / "clip.tiff"
        action_cam_window._write_tiff_stack(path, frames)

        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            assert tif.pages[0].compression.name.lower() == "none"
    finally:
        action_cam_window.checkbox_show_track_path.setChecked(False)


def test_overlays_active_reflects_both_checkboxes(action_cam_window):
    action_cam_window.checkbox_show_segmentation.setChecked(False)
    action_cam_window.checkbox_show_track_path.setChecked(False)
    assert action_cam_window._overlays_active() is False

    action_cam_window.checkbox_show_segmentation.setChecked(True)
    assert action_cam_window._overlays_active() is True
    action_cam_window.checkbox_show_segmentation.setChecked(False)

    action_cam_window.checkbox_show_track_path.setChecked(True)
    assert action_cam_window._overlays_active() is True
    action_cam_window.checkbox_show_track_path.setChecked(False)


def test_to_uint8_scales_per_frame(action_cam_window):
    frame = np.array([[0, 100], [200, 1000]], dtype=np.uint16)
    scaled = action_cam_window._to_uint8(frame)
    assert scaled.dtype == np.uint8
    assert scaled.min() == 0
    assert scaled.max() == 255


def test_to_uint8_handles_a_constant_frame(action_cam_window):
    frame = np.full((4, 4), 500, dtype=np.uint16)
    scaled = action_cam_window._to_uint8(frame)
    assert scaled.dtype == np.uint8
    assert np.all(scaled == 0)


def test_write_mp4_creates_a_nonempty_file(action_cam_window, tmp_path):
    action_cam_window._load_track(2)
    frames = action_cam_window._render_frames_for_export()

    path = tmp_path / "clip.mp4"
    action_cam_window._write_mp4(path, frames, fps=5)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_start_export_requires_a_loaded_track(action_cam_window, monkeypatch):
    # action_cam_window is reused across this file's tests; force the
    # "nothing loaded" state explicitly rather than assuming it.
    action_cam_window._frames = np.empty(0, dtype=int)
    notified = []
    monkeypatch.setattr(action_cam_module, "notify", notified.append)

    action_cam_window._start_export()

    assert notified and "load a track" in notified[0].lower()


def test_start_export_suggests_a_default_filename(action_cam_window, monkeypatch, tmp_path):
    action_cam_window._load_track(1)
    captured = {}

    def fake_get_save_file_name(*args, **kwargs):
        # Whether getSaveFileName ends up called bound or unbound depends on
        # binding details we don't want this test to depend on - just record
        # everything passed and search it below, rather than assuming which
        # positional index the suggested filename lands at.
        captured["values"] = list(args) + list(kwargs.values())
        return ("", "")  # cancelled - just checking what was suggested

    monkeypatch.setattr(
        action_cam_module.QFileDialog, "getSaveFileName", fake_get_save_file_name
    )

    action_cam_window._start_export()

    suggested_name = next(
        v for v in captured["values"] if isinstance(v, str) and v.endswith(".mp4")
    )
    # [image layer name]_ID_[cell/track id] - the image layer in the fixture
    # is named "raw", the loaded track is 1.
    assert suggested_name == "raw_ID_1.mp4"


def test_start_export_writes_the_chosen_format(action_cam_window, monkeypatch, tmp_path):
    action_cam_window._load_track(1)
    target = tmp_path / "my_clip.mp4"
    monkeypatch.setattr(
        action_cam_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "MP4 video (*.mp4)"),
    )
    notified = []
    monkeypatch.setattr(action_cam_module, "notify", notified.append)

    action_cam_window._start_export()

    assert target.is_file()
    assert notified and "Exported" in notified[0]


def test_start_export_cancelled_dialog_writes_nothing(action_cam_window, monkeypatch, tmp_path):
    action_cam_window._load_track(1)
    monkeypatch.setattr(
        action_cam_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )

    action_cam_window._start_export()

    assert list(tmp_path.iterdir()) == []


def test_start_export_pauses_and_resumes_playback(action_cam_window, monkeypatch, tmp_path):
    action_cam_window._load_track(1)
    target = tmp_path / "clip.mp4"
    monkeypatch.setattr(
        action_cam_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "MP4 video (*.mp4)"),
    )

    action_cam_window._toggle_play_pause()
    assert action_cam_window._timer.isActive()

    action_cam_window._start_export()

    assert action_cam_window._timer.isActive()
    action_cam_window._toggle_play_pause()  # leave it stopped for later tests


def test_draw_track_path_fades_and_truncates_after_30_frames(action_cam_window):
    """Regression: the path used to show the whole track travelled so far,
    unbounded - now it fades out and stops growing after
    TRACK_PATH_FADE_FRAMES frames."""
    n = 50
    action_cam_window._centroids = np.array([[i, i] for i in range(n)], dtype=float)

    line_collection = action_cam_window._draw_track_path(
        action_cam_window._axes, n - 1, y0=0, x0=0
    )

    try:
        assert line_collection is not None
        segments = line_collection.get_segments()
        assert len(segments) == action_cam_module.TRACK_PATH_FADE_FRAMES - 1

        # get_edgecolors() (the Collection base class API), not the
        # LineCollection-specific get_color()/get_colors() aliases whose
        # exact spelling varies across matplotlib versions.
        colors = line_collection.get_edgecolors()
        assert len(colors) == len(segments)
        assert colors[0][3] == pytest.approx(0.1, abs=0.01)  # oldest: faint
        assert colors[-1][3] == pytest.approx(1.0, abs=0.01)  # current: opaque
    finally:
        line_collection.remove()


def test_draw_track_path_returns_none_for_a_single_point(action_cam_window):
    action_cam_window._centroids = np.array([[5.0, 5.0]])
    result = action_cam_window._draw_track_path(
        action_cam_window._axes, 0, y0=0, x0=0
    )
    assert result is None


def test_draw_track_path_shows_the_whole_short_track(action_cam_window):
    """A track shorter than the fade window is shown in full, not truncated."""
    action_cam_window._load_track(1)  # 5 frames, well under the fade window

    line_collection = action_cam_window._draw_track_path(
        action_cam_window._axes, len(action_cam_window._frames) - 1, y0=0, x0=0
    )

    try:
        assert line_collection is not None
        assert len(line_collection.get_segments()) == len(action_cam_window._frames) - 1
    finally:
        line_collection.remove()


def test_draw_segmentation_outline_uses_an_image_overlay_not_a_scatter(
    action_cam_window,
):
    """Regression: a fixed-point-size scatter marker looked a very different
    relative size between the live view (crop upscaled to a much bigger
    canvas) and export (crop rendered at its native pixel size) - an image
    overlay scales exactly with the crop instead, avoiding the mismatch."""
    from matplotlib.image import AxesImage

    action_cam_window._load_track(1)
    frame = action_cam_window._frames[0]
    y0, y1, x0, x1 = action_cam_window._current_crop_bounds(0)

    artist = action_cam_window._draw_segmentation_outline(
        frame, y0, y1, x0, x1, index=0
    )

    try:
        assert artist is not None
        assert isinstance(artist, AxesImage)
        overlay = artist.get_array()
        assert overlay.shape[:2] == (y1 - y0, x1 - x0)
        # Some pixels are the opaque outline color, most are transparent.
        assert (overlay[..., 3] == 255).any()
        assert (overlay[..., 3] == 0).any()
    finally:
        artist.remove()
