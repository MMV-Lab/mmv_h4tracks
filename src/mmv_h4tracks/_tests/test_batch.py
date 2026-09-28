"""Module providing tests for the batch tab"""

import numpy as np
import pytest
import zarr

import mmv_h4tracks._batch as batch_module
import mmv_h4tracks._processing as processing
from mmv_h4tracks._constants import METRIC_NAMES

N_FRAMES = 8
FRAME_SHAPE = (32, 32)


@pytest.fixture
def batch_window(create_widget):
    """The batch tab of a clean main widget"""
    return create_widget.batch_window


def synthetic_movie():
    """Raw data and segmentation of two cells moving over ``N_FRAMES`` frames"""
    raw = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.uint16)
    segmentation = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.int32)
    for frame in range(N_FRAMES):
        segmentation[frame, 5 + frame : 9 + frame, 5:9] = 1
        segmentation[frame, 20:24, 20 + frame % 2 : 24 + frame % 2] = 2
    raw[segmentation > 0] = 100
    return raw, segmentation


@pytest.fixture
def stub_movie(monkeypatch):
    """Replace file reading and Cellpose with synthetic data"""
    raw, segmentation = synthetic_movie()
    monkeypatch.setattr(batch_module, "load_tiff", lambda file: raw)
    monkeypatch.setattr(
        processing,
        "segment_volume",
        lambda data, parameters, n_processes, progress_cb=None: segmentation,
    )
    return raw, segmentation


def settings_for(
    tmp_path,
    tracking=True,
    metrics=True,
    tracker="Coordinate-based tracking",
    metric_names=None,
):
    """Batch configuration as ``_run_on_click`` assembles it"""
    return {
        "parameters": {"model_path": "unused"},
        "n_processes": 1,
        "tracking": tracking,
        "tracker": tracker,
        "metrics": metrics,
        "metric_names": list(METRIC_NAMES) if metric_names is None else metric_names,
        "output_dir": tmp_path,
        "filters": ("", ""),
    }


def test_filter_lineedits_are_synced_with_analysis_tab(batch_window):
    """Movement/duration filters are shared with the analysis tab"""
    analysis_window = batch_window.parent.analysis_window

    batch_window.lineedit_movement.setText("17")
    assert analysis_window.lineedit_movement.text() == "17"

    analysis_window.lineedit_track_duration.setText("4")
    assert batch_window.lineedit_track_duration.text() == "4"

    batch_window.lineedit_movement.setText("")
    analysis_window.lineedit_track_duration.setText("")


def test_inputs_are_locked_while_running(batch_window):
    """The configuration can't be edited while a batch is running"""
    batch_window._set_inputs_enabled(False)
    assert not batch_window.btn_run.isEnabled()
    assert not batch_window.lineedit_movement.isEnabled()

    batch_window._set_inputs_enabled(True)
    assert batch_window.btn_run.isEnabled()
    assert batch_window.lineedit_movement.isEnabled()


def test_unchecking_tracking_disables_and_unchecks_metrics(batch_window):
    """Metrics require tracks, so they can't stay checked once tracking is off"""
    batch_window.checkbox_tracking.setChecked(True)
    batch_window.checkbox_metrics.setChecked(True)

    batch_window.checkbox_tracking.setChecked(False)
    assert not batch_window.checkbox_metrics.isEnabled()
    assert not batch_window.checkbox_metrics.isChecked()

    batch_window.checkbox_tracking.setChecked(True)
    assert batch_window.checkbox_metrics.isEnabled()
    batch_window.checkbox_metrics.setChecked(True)


def test_metric_checkboxes_default_to_checked_and_independent_of_analysis_tab(
    batch_window,
):
    """Batch has its own metric selection, unrelated to the Analysis tab's"""
    analysis_window = batch_window.parent.analysis_window

    for name in METRIC_NAMES:
        assert batch_window.metric_checkboxes[name].isChecked()

    for checkbox in analysis_window.checkboxes:
        checkbox.setChecked(False)
    assert all(
        checkbox.isChecked() for checkbox in batch_window.metric_checkboxes.values()
    )

    batch_window.metric_checkboxes["Size"].setChecked(False)
    assert all(not checkbox.isChecked() for checkbox in analysis_window.checkboxes)

    batch_window.metric_checkboxes["Size"].setChecked(True)


def test_metrics_group_follows_compute_metrics_checkbox(batch_window):
    """The per-metric checkboxes are only shown while metrics will be computed"""
    # isVisibleTo (rather than isVisible) reflects the explicit show/hide
    # state regardless of whether the top-level window itself is shown, which
    # it never is under the headless test viewer.
    def group_shown():
        return batch_window.metrics_group.isVisibleTo(batch_window)

    batch_window.checkbox_tracking.setChecked(True)
    batch_window.checkbox_metrics.setChecked(True)
    assert group_shown()

    batch_window.checkbox_metrics.setChecked(False)
    assert not group_shown()

    batch_window.checkbox_metrics.setChecked(True)
    assert group_shown()

    batch_window.checkbox_tracking.setChecked(False)
    assert not group_shown()

    batch_window.checkbox_tracking.setChecked(True)
    batch_window.checkbox_metrics.setChecked(True)


def test_filters_desync_while_running_and_resync_afterwards(batch_window):
    """Analysis tab edits during a run are picked up once the tab unlocks"""
    analysis_window = batch_window.parent.analysis_window

    batch_window._set_inputs_enabled(False)
    analysis_window.lineedit_movement.setText("23")
    assert batch_window.lineedit_movement.text() == ""

    batch_window._set_inputs_enabled(True)
    assert batch_window.lineedit_movement.text() == "23"

    analysis_window.lineedit_movement.setText("")


def test_running_batch_keeps_its_filter_values(batch_window, stub_movie, tmp_path):
    """The csv uses the filters of the moment the batch was started"""
    analysis_window = batch_window.parent.analysis_window
    settings = settings_for(tmp_path)
    settings["filters"] = ("7", "3")

    batch_window._set_inputs_enabled(False)
    analysis_window.lineedit_movement.setText("999")
    batch_window._process_movie(tmp_path / "movie.tif", settings)
    batch_window._set_inputs_enabled(True)
    analysis_window.lineedit_movement.setText("")

    assert "Movement Threshold: 7 pixels" in (tmp_path / "movie.csv").read_text()


@pytest.mark.parametrize(
    "tracker", ["Coordinate-based tracking", "Overlap-based tracking"]
)
def test_process_movie_writes_zarr_and_csv(batch_window, stub_movie, tmp_path, tracker):
    """Both trackers produce a zarr with all three arrays and a metrics csv"""
    raw, segmentation = stub_movie

    batch_window._process_movie(
        tmp_path / "movie.tif", settings_for(tmp_path, tracker=tracker)
    )

    zarr_path = tmp_path / "movie.zarr"
    csv_path = tmp_path / "movie.csv"
    assert zarr_path.exists()
    assert csv_path.exists()

    root = zarr.open(str(zarr_path), mode="r")
    assert np.array_equal(root["raw_data"][:], raw)
    assert np.array_equal(root["segmentation_data"][:], segmentation)
    tracks = root["tracking_data"][:]
    assert tracks.shape[1] == 4
    assert len(np.unique(tracks[:, 0])) == 2

    assert "Number of cells" in csv_path.read_text()


def test_process_movie_does_not_overwrite_results(batch_window, stub_movie, tmp_path):
    """A second run of the same movie is written under a unique name"""
    settings = settings_for(tmp_path)

    batch_window._process_movie(tmp_path / "movie.tif", settings)
    batch_window._process_movie(tmp_path / "movie.tif", settings)

    for name in ["movie.zarr", "movie.csv", "movie_1.zarr", "movie_1.csv"]:
        assert (tmp_path / name).exists()


def test_process_movie_without_tracking(batch_window, stub_movie, tmp_path):
    """Without tracking only a zarr with empty tracks is written"""
    batch_window._process_movie(
        tmp_path / "movie.tif", settings_for(tmp_path, tracking=False, metrics=False)
    )

    assert not (tmp_path / "movie.csv").exists()
    root = zarr.open(str(tmp_path / "movie.zarr"), mode="r")
    assert root["tracking_data"].shape == (0, 4)


def test_process_movie_without_cells_skips_metrics(
    batch_window, monkeypatch, tmp_path
):
    """An empty segmentation yields no tracks, so no csv is written"""
    raw, _ = synthetic_movie()
    empty = np.zeros((N_FRAMES, *FRAME_SHAPE), dtype=np.int32)
    monkeypatch.setattr(batch_module, "load_tiff", lambda file: raw)
    monkeypatch.setattr(
        processing,
        "segment_volume",
        lambda data, parameters, n_processes, progress_cb=None: empty,
    )

    batch_window._process_movie(tmp_path / "movie.tif", settings_for(tmp_path))

    assert (tmp_path / "movie.zarr").exists()
    assert not (tmp_path / "movie.csv").exists()


def test_run_batch_continues_after_failure(batch_window, monkeypatch, tmp_path):
    """A movie that cannot be processed is reported, the others are not skipped"""

    class Reporter:
        desc = ""

        def set_n(self, n):
            pass

    def fail_on_second(file, settings):
        if file.name == "broken.tif":
            raise ValueError("unreadable")

    monkeypatch.setattr(batch_window, "_process_movie", fail_on_second)
    images = [tmp_path / "first.tif", tmp_path / "broken.tif", tmp_path / "last.tif"]

    processed, failed = batch_window._worker_run_batch(
        images, settings_for(tmp_path), Reporter()
    )

    assert processed == ["first.tif", "last.tif"]
    assert failed == ["broken.tif"]
