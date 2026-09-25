import logging
from pathlib import Path

import numpy as np

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QGridLayout,
    QComboBox,
    QCheckBox,
)

from ._constants import METRIC_NAMES
from ._logger import notify
from ._qt_utils import apply_napari_dark_theme
from ._reader import open_dialog, load_tiff
from ._tracking import iter_overlap_tracking
from ._writer import write_zarr_data
import mmv_h4tracks._processing as processing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
for handler in logger.handlers:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(handler)
logger.debug("logger initialized")


def _drain(generator):
    """
    Run a progress-yielding generator to completion and return its result.

    Parameters
    ----------
    generator : generator
        Generator whose yields are progress only

    Returns
    -------
    object
        The value the generator returns
    """
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def _unique_output_paths(output_dir, stem, with_csv):
    """
    Find output names that are not taken yet, appending ``_1``, ``_2``, …

    Zarr and csv share the same index, so the results of one movie belong
    together by name.

    Parameters
    ----------
    output_dir : Path
        Directory the results are written to
    stem : str
        Name of the input file without its suffix
    with_csv : bool
        Whether a csv name is needed as well

    Returns
    -------
    tuple
        ``(zarr_path, csv_path)``, the latter ``None`` if ``with_csv`` is false
    """
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index}"
        zarr_path = output_dir / f"{stem}{suffix}.zarr"
        csv_path = output_dir / f"{stem}{suffix}.csv" if with_csv else None
        if not zarr_path.exists() and (csv_path is None or not csv_path.exists()):
            return zarr_path, csv_path
        index += 1


def _mirror_lineedits(first, second, suspended):
    """
    Keep two line edits showing the same text, in both directions.

    Parameters
    ----------
    first, second : QLineEdit
        The line edits to synchronize
    suspended : callable
        While this returns True no text is mirrored, so the two line edits are
        allowed to drift apart
    """

    def make_handler(target):
        def apply(text):
            if suspended():
                return
            if target.text() != text:
                target.setText(text)

        return apply

    first.textChanged.connect(make_handler(second))
    second.textChanged.connect(make_handler(first))


class BatchWindow(QWidget):
    """
    Tab for batch processing.

    Segments, tracks and analyses all tif/tiff movies of a source folder in one
    run, without loading them into the viewer. Per movie a zarr file (raw,
    segmentation, tracks) and optionally a csv with all metrics of the analysis
    tab are written to the output folder, named after the input file.

    Parameters
    ----------
    parent : MMVH4TRACKS
        The main dock widget
    """

    def __init__(self, parent):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.parent = parent
        apply_napari_dark_theme(self)

        self.custom_models = processing.read_custom_model_dict()
        # While a batch runs the filter fields may drift from the analysis tab
        self._filters_suspended = False

        ### QObjects

        # Buttons
        btn_source = QPushButton("Browse")
        btn_output = QPushButton("Browse")
        browse_tooltip = "Choose the folder containing the zarr datasets to process"
        btn_source.setToolTip(browse_tooltip)
        btn_output.setToolTip("Choose the folder where results will be written")

        self.btn_run = QPushButton("Run batch")
        self.btn_run.setToolTip(
            "Segment (and track/analyse) every tif/tiff movie of the source "
            "folder and write the results to the output folder"
        )

        # LineEdits
        self.lineedit_source = QLineEdit()
        self.lineedit_source.setReadOnly(True)
        self.lineedit_source.setPlaceholderText("Select source folder …")
        self.lineedit_source.setToolTip(browse_tooltip)
        self.lineedit_output = QLineEdit()
        self.lineedit_output.setReadOnly(True)
        self.lineedit_output.setPlaceholderText("Select output folder …")
        self.lineedit_output.setToolTip(
            "Choose the folder where results will be written"
        )

        btn_source.clicked.connect(self._select_source)
        btn_output.clicked.connect(self._select_output)
        self.btn_run.clicked.connect(self._run_on_click)

        # QComboBoxes
        self.combobox_cellpose_model = QComboBox()
        self.combobox_cellpose_model.setToolTip("select model")
        hardcoded_models, custom_models = processing.read_models(self)
        processing.display_models(self, hardcoded_models, custom_models)

        self.combobox_tracker = QComboBox()
        self.combobox_tracker.addItems(
            ["Coordinate-based tracking", "Overlap-based tracking"]
        )
        self.combobox_tracker.setToolTip(
            "Coordinate-based: match cells across frames by centroid proximity "
            "(linear assignment)\n"
            "Overlap-based: match cells across frames by segmentation overlap"
        )

        # QCheckBoxes
        self.checkbox_tracking = QCheckBox("Compute tracks")
        self.checkbox_tracking.setChecked(True)
        self.checkbox_tracking.toggled.connect(self.combobox_tracker.setEnabled)
        self.checkbox_metrics = QCheckBox("Compute metrics")
        self.checkbox_metrics.setChecked(True)
        self.checkbox_metrics.setToolTip(
            "Speed, size, direction, ... (selected below)\n"
            "Requires tracks, exported as csv next to the zarr file"
        )
        self.checkbox_tracking.toggled.connect(self._on_tracking_toggled)
        self.checkbox_metrics.toggled.connect(self._update_metrics_group_visible)

        # Which metrics to compute, independent of the Analysis tab's selection
        self.metric_checkboxes = {name: QCheckBox(name) for name in METRIC_NAMES}
        for checkbox in self.metric_checkboxes.values():
            checkbox.setChecked(True)
        self.metrics_group = QGroupBox("Metrics")
        self.metrics_group.setLayout(QGridLayout())
        for index, name in enumerate(METRIC_NAMES):
            row, column = divmod(index, 2)
            self.metrics_group.layout().addWidget(
                self.metric_checkboxes[name], row, column
            )

        # Line edits for the metric filters, kept in sync with the Analysis tab
        analysis_window = self.parent.analysis_window
        self.lineedit_movement = QLineEdit(analysis_window.lineedit_movement.text())
        self.lineedit_movement.setMaximumWidth(40)
        self.lineedit_track_duration = QLineEdit(
            analysis_window.lineedit_track_duration.text()
        )
        self.lineedit_track_duration.setMaximumWidth(40)
        def suspended():
            return self._filters_suspended

        _mirror_lineedits(
            self.lineedit_movement, analysis_window.lineedit_movement, suspended
        )
        _mirror_lineedits(
            self.lineedit_track_duration,
            analysis_window.lineedit_track_duration,
            suspended,
        )

        # Labels
        label_source = QLabel("Source folder")
        label_output = QLabel("Output folder")
        label_model = QLabel("Cellpose model")
        label_tracker = QLabel("Tracker")
        label_min_movement = QLabel("Movement minimum:")
        label_min_movement.setToolTip(
            "Sort exported tracks by smaller/larger than movement minimum in pixels\n"
            "Shared with the Analysis tab"
        )
        label_min_duration = QLabel("Minimum track length:")
        label_min_duration.setToolTip(
            "Sort exported tracks by shorter/longer than minimum track length in timesteps\n"
            "Shared with the Analysis tab"
        )

        ### Organize objects via widgets
        batch_group = QGroupBox("Batch processing")
        batch_group.setLayout(QGridLayout())
        batch_group.layout().addWidget(label_source, 0, 0)
        batch_group.layout().addWidget(self.lineedit_source, 0, 1)
        batch_group.layout().addWidget(btn_source, 0, 2)
        batch_group.layout().addWidget(label_output, 1, 0)
        batch_group.layout().addWidget(self.lineedit_output, 1, 1)
        batch_group.layout().addWidget(btn_output, 1, 2)
        batch_group.layout().addWidget(label_model, 2, 0)
        batch_group.layout().addWidget(self.combobox_cellpose_model, 2, 1, 1, -1)
        batch_group.layout().addWidget(label_tracker, 3, 0)
        batch_group.layout().addWidget(self.combobox_tracker, 3, 1, 1, -1)
        batch_group.layout().addWidget(self.checkbox_tracking, 4, 0)
        batch_group.layout().addWidget(self.checkbox_metrics, 4, 1)
        batch_group.layout().addWidget(label_min_movement, 5, 0)
        batch_group.layout().addWidget(self.lineedit_movement, 5, 1)
        batch_group.layout().addWidget(label_min_duration, 6, 0)
        batch_group.layout().addWidget(self.lineedit_track_duration, 6, 1)
        batch_group.layout().addWidget(self.btn_run, 7, 0, 1, -1)

        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(batch_group)
        content.layout().addWidget(self.metrics_group)
        content.layout().addStretch(1)

        self.layout().addWidget(content)

        # Everything the user must not touch while a batch is running
        self._inputs = [
            btn_source,
            btn_output,
            self.combobox_cellpose_model,
            self.combobox_tracker,
            self.checkbox_tracking,
            self.checkbox_metrics,
            self.metrics_group,
            self.lineedit_movement,
            self.lineedit_track_duration,
            self.btn_run,
        ]

        # Initial visibility: tracking is on and metrics are checked.
        self._update_metrics_group_visible()

    def _on_tracking_toggled(self, tracking):
        """
        Enable/disable "Compute metrics" with tracking; metrics require
        tracks, so unchecking tracking also unchecks metrics.

        Parameters
        ----------
        tracking : bool
            Whether "Compute tracks" is checked
        """
        self.checkbox_metrics.setEnabled(tracking)
        if not tracking:
            self.checkbox_metrics.setChecked(False)
        self._update_metrics_group_visible()

    def _update_metrics_group_visible(self, *_args):
        """Show the per-metric checkboxes only while metrics will be computed."""
        self.metrics_group.setVisible(self.checkbox_metrics.isChecked())

    def _set_inputs_enabled(self, enabled):
        """
        Lock the tab while a batch is running, unlock it when it is done.

        While locked the filter fields are not synchronized with the analysis
        tab, so the running batch keeps the values it was started with. On
        unlock the analysis tab's current values win.

        Parameters
        ----------
        enabled : bool
            Whether the user may edit the batch configuration
        """
        self._filters_suspended = not enabled
        for widget in self._inputs:
            widget.setEnabled(enabled)
        if not enabled:
            return
        analysis_window = self.parent.analysis_window
        self.lineedit_movement.setText(analysis_window.lineedit_movement.text())
        self.lineedit_track_duration.setText(
            analysis_window.lineedit_track_duration.text()
        )
        # These two only apply if tracks are computed
        tracking = self.checkbox_tracking.isChecked()
        self.combobox_tracker.setEnabled(tracking)
        self._on_tracking_toggled(tracking)

    def _select_source(self):
        folder = open_dialog(
            self.parent,
            filetype="Folder",
            directory=self.lineedit_source.text(),
        )
        if folder:
            self.lineedit_source.setText(folder)

    def _select_output(self):
        folder = open_dialog(
            self.parent,
            filetype="Folder",
            directory=self.lineedit_output.text(),
        )
        if folder:
            self.lineedit_output.setText(folder)

    def _run_on_click(self):
        source = self.lineedit_source.text()
        output = self.lineedit_output.text()
        images = []
        if source:
            source_dir = Path(source)
            images = sorted(
                [
                    file
                    for file in source_dir.iterdir()
                    if file.is_file() and file.suffix.lower() in (".tif", ".tiff")
                ]
            )

        # show a dialog only when the input is not valid
        if not source:
            notify("Select a source folder first.")
            return
        if not images:
            notify(f"No tif/tiff files found in {source}.")
            return
        if not output:
            notify("Select an output folder.")
            return

        model = self.combobox_cellpose_model.currentText()
        try:
            parameters = processing._get_parameters(self, model)
        except ValueError as exc:
            notify(str(exc))
            return

        # Read all GUI state here: the batch itself runs on a worker thread.
        tracking = self.checkbox_tracking.isChecked()
        metrics_requested = tracking and self.checkbox_metrics.isChecked()
        metric_names = [
            name for name in METRIC_NAMES if self.metric_checkboxes[name].isChecked()
        ]
        if metrics_requested and not metric_names:
            notify("Select at least one metric, or uncheck Compute metrics.")
            return

        settings = {
            "parameters": parameters,
            "n_processes": self.parent.get_process_limit(),
            "tracking": tracking,
            "tracker": self.combobox_tracker.currentText(),
            "metrics": metrics_requested,
            "metric_names": metric_names,
            "output_dir": Path(output),
            "filters": (
                self.lineedit_movement.text(),
                self.lineedit_track_duration.text(),
            ),
        }

        config_lines = [
            "Batch processing configuration:",
            f"Source folder: {source}",
            f"Output folder: {output}",
            f"Cellpose model: {model}",
            f"Tracking: {'yes' if tracking else 'no'}",
        ]
        if tracking:
            config_lines.append(f"Tracker: {settings['tracker']}")
        config_lines.append(f"Metrics: {'yes' if settings['metrics'] else 'no'}")
        for line in config_lines:
            logger.info(line)
        logger.info("Found %d tif/tiff file(s) in %s", len(images), source_dir)

        self._set_inputs_enabled(False)
        processing.run_with_dock_progress(
            self.parent,
            self._worker_run_batch,
            images,
            settings,
            desc="Batch processing",
            total=len(images),
            on_returned=self._batch_finished,
            on_errored=lambda _exc: self._set_inputs_enabled(True),
        )

    def _worker_run_batch(self, images, settings, reporter):
        """
        Process all movies, one after the other. A failing movie is logged and
        skipped so the remaining ones are still processed.

        Parameters
        ----------
        images : list of Path
            The movies to process
        settings : dict
            Batch configuration, see ``_run_on_click``
        reporter : DockProgressReporter
            One step per movie

        Returns
        -------
        tuple
            ``(processed, failed)``, both lists of file names
        """
        processed = []
        failed = []
        for index, file in enumerate(images):
            reporter.desc = f"Batch processing ({file.name})"
            reporter.set_n(index)
            try:
                self._process_movie(file, settings)
            except Exception as exc:
                logger.exception("%s: failed (%s)", file.name, exc)
                failed.append(file.name)
            else:
                processed.append(file.name)
        reporter.desc = "Batch processing"
        reporter.set_n(len(images))
        return processed, failed

    def _process_movie(self, file, settings):
        """
        Run one movie through segmentation, tracking and analysis and write the
        results to the output folder.

        Parameters
        ----------
        file : Path
            The movie to process
        settings : dict
            Batch configuration, see ``_run_on_click``
        """
        logger.info("Processing %s", file.name)
        raw = np.squeeze(load_tiff(file))
        if raw.ndim not in (2, 3):
            raise ValueError(
                f"{file.name}: expected a 2D or 3D movie, got shape {raw.shape}"
            )

        segmentation = processing.segment_volume(
            raw, settings["parameters"], settings["n_processes"]
        )

        tracks = np.empty((0, 4), dtype=int)
        if settings["tracking"]:
            tracks = self._compute_tracks(segmentation, settings)
            logger.info(
                "%s: %d track(s) found", file.name, len(np.unique(tracks[:, 0]))
            )

        zarr_path, csv_path = _unique_output_paths(
            settings["output_dir"], file.stem, settings["metrics"]
        )
        write_zarr_data(str(zarr_path), raw, segmentation, tracks)
        logger.info("%s: wrote %s", file.name, zarr_path)

        if csv_path is None:
            return
        if len(tracks) == 0:
            logger.warning("%s: no tracks found, skipping metrics", file.name)
            return
        analysis_window = self.parent.analysis_window
        analysis_window._export(
            (str(csv_path),),
            settings["metric_names"],
            tracks,
            segmentation,
            None,
            filters=settings["filters"],
        )
        logger.info("%s: wrote %s", file.name, csv_path)

    def _compute_tracks(self, segmentation, settings):
        """
        Track the segmentation with the tracker selected in the batch tab.

        Parameters
        ----------
        segmentation : nd array
            3D label volume (ZYX)
        settings : dict
            Batch configuration, see ``_run_on_click``

        Returns
        -------
        nd array
            (N,4) shape array in napari's tracks layer format (ID, z, y, x)
        """
        if not np.any(segmentation):
            logger.warning("Segmentation is empty, skipping tracking")
            return np.empty((0, 4), dtype=int)
        if settings["tracker"].startswith("Overlap"):
            tracks = _drain(
                iter_overlap_tracking(segmentation, settings["n_processes"])
            )
        else:
            tracks = processing.track_by_coordinates(
                segmentation, settings["n_processes"]
            )
        if tracks is None or len(tracks) == 0:
            return np.empty((0, 4), dtype=int)
        return np.asarray(tracks, dtype=int)

    def _batch_finished(self, result):
        """Unlock the tab and report how many movies were processed."""
        self._set_inputs_enabled(True)
        processed, failed = result
        message = f"Batch finished: {len(processed)} of {len(processed) + len(failed)} movie(s) processed."
        if failed:
            message += "\nFailed: " + ", ".join(failed)
        notify(message)
