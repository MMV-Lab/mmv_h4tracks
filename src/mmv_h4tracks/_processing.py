import logging
import multiprocessing
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from cellpose import models, core
from napari.qt.threading import create_worker
from qtpy.QtCore import Qt, QThread, QObject, Signal
from qtpy.QtWidgets import QApplication, QMessageBox
from scipy import ndimage, optimize, spatial

from ._constants import (
    APPROX_INF,
    MAX_MATCHING_DIST,
    CUSTOM_MODEL_PREFIX,
    DEFAULT_TRACKS_LAYER_NAME,
)
from ._concurrency import (
    iter_map_as_completed,
    iter_starmap_as_completed,
    map_parallel,
    starmap_parallel,
)
from ._custom_models import (
    get_custom_model_store,
    package_models_dir,
)
from ._grabber import grab_layer
from ._session_trained_models import overlap_training_frames_with_stack
from ._logger import handle_exception, notify
from ._qt_utils import layer_as_numpy, _iter_multiscale_levels, awaiting_user_dialog
from ._train import CELLPOSE_TRAIN_N_EPOCHS_DEFAULT, _sanitize_model_name_fragment

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


def segment_slice_cpu(layer_slice, parameters):
    """
    Parameters
    ----------
    layer_slice : nd array
        the slice of raw image data to calculate segmentation for
    parameters : dict
        the parameters for the segmentation model

    Returns
    -------
    nd array
        the segmentation mask for the slice
    """
    starttime = time.time()
    logger.debug(
        "Starting pid: "
        + str(multiprocessing.current_process().pid)
        + " for slice at "
        + str(starttime)
    )
    logger.info("Segmentation process started")
    model = models.CellposeModel(gpu=False, pretrained_model=parameters["model_path"])
    eval_params = {k: v for k, v in parameters.items() if k != "model_path"}
    mask, _, _ = model.eval(layer_slice, **eval_params)
    endtime = time.time()
    logger.debug(
        "Ending pid: "
        + str(multiprocessing.current_process().pid)
        + " for slice at "
        + str(endtime)
        + " after "
        + str(endtime - starttime)
        + " seconds"
    )
    logger.info(
        "Segmentation process finished with runtime: " + str(endtime - starttime)
    )
    return mask


def calculate_centroids(label_slice):
    """
    Calculate the centroids of objects in a 2D slice.

    Parameters
    ----------
    label_slice : numpy.ndarray
        A 2D numpy array representing the slice.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays: the centroids and the labels.
    """
    labels = np.unique(label_slice)[1:]
    centroids = ndimage.center_of_mass(label_slice, labels=label_slice, index=labels)

    return (centroids, labels)


def match_centroids(
    slice_pair,
):
    """
    Match centroids between two slices.

    Parameters
    ----------
    slice_pair : tuple
        A tuple containing two slices, each represented by a tuple
        containing the centroids and IDs of cells in that slice.

    Returns
    -------
    list
        A list of matched pairs, where each pair consists of the centroid and ID
    """
    parent_centroids = slice_pair[0][0]
    parent_ids = slice_pair[0][1]
    child_centroids = slice_pair[1][0]
    child_ids = slice_pair[1][1]

    num_cells_parent = len(parent_centroids)
    num_cells_child = len(child_centroids)

    # calculate distance between each pair of cells
    cost_mat = spatial.distance.cdist(parent_centroids, child_centroids)

    # if the distance is too far, change to approx. Inf.
    cost_mat[cost_mat > MAX_MATCHING_DIST] = APPROX_INF

    # add edges from cells in previous frame to auxillary vertices
    # in order to accomendate segmentation errors and leaving cells
    cost_mat_aug = (
        MAX_MATCHING_DIST
        * 1.2
        * np.ones((num_cells_parent, num_cells_child + num_cells_parent), dtype=float)
    )
    cost_mat_aug[:num_cells_parent, :num_cells_child] = cost_mat[:, :]

    # solve the optimization problem
    row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)

    matched_pairs = []

    for i in range(len(row_ind)):
        parent_centroid = np.around(parent_centroids[row_ind[i]])
        parent_id = parent_ids[row_ind[i]]
        try:
            child_centroid = np.around(child_centroids[col_ind[i]])
            child_id = child_ids[col_ind[i]]
        except IndexError:
            continue

        matched_pairs.append(
            {
                "parent": {"centroid": parent_centroid, "id": parent_id},
                "child": {"centroid": child_centroid, "id": child_id},
            }
        )

    return matched_pairs


def read_custom_model_dict():
    """
    Read custom model parameters from the user data store
    (``~/.mmv_h4tracks`` or ``MMV_H4TRACKS_USER_DATA``).
    """
    return get_custom_model_store().load()


def read_models(widget):
    """
    Reads the available models from the package ``models/`` dir and the user
    custom-model store and returns them.
    """
    path = package_models_dir()

    hardcoded_models = [file.name for file in path.iterdir() if not file.is_dir()]
    custom_models = []

    store = get_custom_model_store()
    custom_model_filenames = store.list_weight_filenames()
    for custom_model in widget.custom_models:
        if widget.custom_models[custom_model]["filename"] in custom_model_filenames:
            custom_models.append(CUSTOM_MODEL_PREFIX + custom_model)
    return hardcoded_models, custom_models


def display_models(widget, hardcoded_models, custom_models):
    """
    Adds the passed models to the Cellpose model combobox.
    """
    hardcoded_models.sort()
    custom_models.sort()
    widget.combobox_cellpose_model.clear()
    widget.combobox_cellpose_model.addItems(hardcoded_models)
    widget.combobox_cellpose_model.addItems(custom_models)


def custom_model_weights_basename(display_name: str) -> str:
    """
    Canonical custom model name: used as ``custom_models.json`` key and as the weights
    filename (no extension) under the user custom-models directory.
    """
    stem = _sanitize_model_name_fragment(display_name)
    return stem if stem else "model"


def is_custom_model_display_name_taken(widget, display_name: str) -> bool:
    """True if the canonical name is already a JSON key or weights file on disk."""
    canonical = custom_model_weights_basename(display_name)
    if canonical in widget.custom_models:
        return True
    return get_custom_model_store().weights_path(canonical).is_file()


def persist_custom_model_entry(widget, display_name: str, source_weights: Path, params: dict) -> Path:
    """
    Copy Cellpose weights into the user custom-model store and update the registry JSON.

    The JSON key and on-disk basename are always ``custom_model_weights_basename(display_name)``.
    """
    store = get_custom_model_store()
    canonical = custom_model_weights_basename(display_name)
    return store.persist(
        display_name,
        source_weights,
        params,
        widget.custom_models,
        canonical=canonical,
    )


def _worker_train_cellpose(export_dir, n_epochs: int, reporter):
    """Thread worker: run Cellpose CLI training with optional dock progress."""
    from ._train import train_cellpose

    return train_cellpose(Path(export_dir), n_epochs=n_epochs, reporter=reporter)


def start_cellpose_training_worker(
    widget,
    export_dir: Path,
    *,
    n_epochs: int | None = None,
    on_returned=None,
):
    """
    Start train-only worker with dock progress (1 step per 10 epochs via log parsing).

    ``returned`` resets progress then calls ``on_returned`` with ``CellposeCliTrainingResult``.
    """
    from ._train import (
        CELLPOSE_TRAIN_PROGRESS_EPOCH_STRIDE,
        cellpose_train_progress_steps,
    )

    parent = widget.parent
    ne = CELLPOSE_TRAIN_N_EPOCHS_DEFAULT if n_epochs is None else int(n_epochs)
    steps = cellpose_train_progress_steps(ne, CELLPOSE_TRAIN_PROGRESS_EPOCH_STRIDE)
    return run_with_dock_progress(
        parent,
        _worker_train_cellpose,
        Path(export_dir),
        ne,
        desc="Training Cellpose",
        total=steps,
        on_returned=on_returned,
    )


def _load_segmentation_image_data(widget, demo: bool):
    """
    Load the selected image layer as a squeezed numpy volume (same as segmentation worker).
    Returns (data_squeezed, removed_dims).
    """
    layer = widget.parent.selected_image_layer()

    raw = layer.data
    levels = None
    if isinstance(raw, (list, tuple)) or getattr(layer, "multiscale", False):
        try:
            levels = _iter_multiscale_levels(raw)
            if levels is None and hasattr(raw, "__len__"):
                levels = [raw[i] for i in range(len(raw))]
        except Exception:
            levels = None
        if levels is not None:
            shapes = []
            for level in levels:
                try:
                    shapes.append(tuple(np.asarray(level).shape))
                except Exception:
                    shapes.append(None)
            logger.info(
                "Segmentation image %r is multiscale; level shapes=%s",
                getattr(layer, "name", None),
                shapes,
            )

    data = layer_as_numpy(layer)

    original_shape = data.shape
    data_squeezed = np.squeeze(data)
    if data_squeezed.ndim not in (2, 3):
        raise ValueError(
            f"Data must be 2D or 3D after removing trivial dimensions. "
            f"Original shape: {original_shape}, squeezed: {data_squeezed.shape}"
        )
    removed_dims = [i for i, size in enumerate(original_shape) if size == 1]
    if demo:
        data_squeezed = data_squeezed[0:5]

    logger.info(
        "Segmentation input resolution: layer=%r multiscale=%s "
        "full_shape=%s squeezed_shape=%s dtype=%s "
        "yx=%s demo=%s",
        getattr(layer, "name", None),
        bool(getattr(layer, "multiscale", False))
        or isinstance(raw, (list, tuple)),
        original_shape,
        data_squeezed.shape,
        data_squeezed.dtype,
        data_squeezed.shape[-2:] if data_squeezed.ndim >= 2 else data_squeezed.shape,
        demo,
    )
    return data_squeezed, removed_dims


def _fill_excluded_frames_from_labels_layer(
    viewer,
    mask: np.ndarray,
    exclude_set: frozenset,
    labels_layer_name: str,
    image_ndim: int,
) -> None:
    """Copy label data for excluded time indices from the training segmentation layer."""
    from ._train import _frame_2d, _get_array_from_layer

    try:
        seg_layer = grab_layer(viewer, labels_layer_name)
        seg_vol = _get_array_from_layer(seg_layer)
    except Exception as exc:
        logger.warning(
            "Excluded frames not copied from labels layer %r: %s",
            labels_layer_name,
            exc,
        )
        return
    if image_ndim == 2:
        if 0 not in exclude_set:
            return
        try:
            sl = _frame_2d(seg_vol, 0)
            if sl.shape != mask.shape:
                logger.warning(
                    "Training labels shape %s does not match output %s; frame not copied.",
                    sl.shape,
                    mask.shape,
                )
                return
            mask[:] = np.asarray(sl, dtype=np.int32)
        except ValueError as exc:
            logger.warning("Could not copy 2D excluded frame from training labels: %s", exc)
        return
    for i in exclude_set:
        try:
            sl = _frame_2d(seg_vol, i)
            if sl.shape != mask[i].shape:
                logger.warning(
                    "Training labels shape %s does not match output slice %s at t=%s; skipped.",
                    sl.shape,
                    mask[i].shape,
                    i,
                )
                continue
            mask[i] = np.asarray(sl, dtype=np.int32)
        except ValueError as exc:
            logger.warning("Could not copy excluded frame %s from training labels: %s", i, exc)


def _read_tiff_path(path: Path) -> np.ndarray:
    try:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))
    except ImportError:
        import imageio.v2 as imageio

        return np.asarray(imageio.imread(str(path)))


def _fill_excluded_frames_from_training_masks_dir(
    mask: np.ndarray,
    exclude_set: frozenset,
    masks_dir: Path,
    layer_prefix: str,
    image_ndim: int,
) -> None:
    """Fill excluded time indices from saved training mask TIFFs on disk."""
    masks_dir = Path(masks_dir)
    for i in exclude_set:
        tif_path = masks_dir / f"{layer_prefix}_frame_{i:05d}_masks.tif"
        if not tif_path.is_file():
            logger.warning("Missing training mask %s; frame left blank.", tif_path)
            continue
        try:
            sl = _read_tiff_path(tif_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", tif_path, exc)
            continue
        if image_ndim == 2:
            if i != 0:
                continue
            if sl.shape != mask.shape:
                logger.warning(
                    "Training mask shape %s does not match output %s; frame not copied.",
                    sl.shape,
                    mask.shape,
                )
                continue
            mask[:] = np.asarray(sl, dtype=np.int32)
        else:
            if i < 0 or i >= mask.shape[0]:
                continue
            if sl.shape != mask[i].shape:
                logger.warning(
                    "Training mask shape %s does not match output slice %s at t=%s; skipped.",
                    sl.shape,
                    mask[i].shape,
                    i,
                )
                continue
            mask[i] = np.asarray(sl, dtype=np.int32)


def _prompt_exclude_training_frames_if_applicable(
    widget, demo: bool
) -> tuple[frozenset, Path | None, str | None] | None:
    """
    If the current model was trained this session on the same image layer (by sanitized
    name vs mask filename prefix) and overlapping frames, ask whether to exclude those
    frames from prediction.

    Returns
    -------
    None
        Predict all frames (no exclusion).
    (frozenset[int], Path | None, str | None)
        Excluded frame indices, training masks directory, layer prefix for mask filenames.
    """
    selected = widget.combobox_cellpose_model.currentText()
    if not selected.startswith(CUSTOM_MODEL_PREFIX):
        return None
    display_name = selected[len(CUSTOM_MODEL_PREFIX) :]
    meta = widget.parent.session_trained_models.get(display_name)
    if not meta:
        return None
    image_name = widget.parent.combobox_image.currentText()
    if _sanitize_model_name_fragment(image_name) != meta["layer_prefix"]:
        return None
    try:
        data_squeezed, _ = _load_segmentation_image_data(widget, demo)
    except ValueError:
        return None
    training_frames = tuple(meta["frames"])
    if data_squeezed.ndim == 2:
        overlap = overlap_training_frames_with_stack(
            training_frames, 1, is_single_frame_2d=True
        )
    else:
        n_in_stack = data_squeezed.shape[0]
        overlap = overlap_training_frames_with_stack(
            training_frames, n_in_stack, is_single_frame_2d=False
        )
    if not overlap:
        return None
    frames_str = ", ".join(str(f) for f in overlap)
    text = (
        f"Frames {frames_str} were used to train this model on this image layer.\n\n"
        "Exclude those frames from Cellpose prediction and copy their labels from the "
        "saved training mask TIFFs instead?\n\n"
        "(Yes: predict only other frames; training frames are filled from disk. "
        "No: run Cellpose on the full stack.)"
    )
    with awaiting_user_dialog(widget):
        reply = QMessageBox.question(
            widget,
            "napari",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
    if reply != QMessageBox.Yes:
        return None
    masks_dir = Path(meta["training_masks_dir"])
    layer_prefix = meta["layer_prefix"]
    return (frozenset(overlap), masks_dir, layer_prefix)


def run_segmentation(widget):
    """
    Calls segmentation without demo flag set
    """
    pr = _prompt_exclude_training_frames_if_applicable(widget, False)
    excl, mdir, mpfx = (None, None, None) if pr is None else pr
    _start_segmentation_worker(widget, False, excl, None, mdir, mpfx)


def run_demo_segmentation(widget):
    """
    Calls segmentation with the demo flag set
    """
    pr = _prompt_exclude_training_frames_if_applicable(widget, True)
    excl, mdir, mpfx = (None, None, None) if pr is None else pr
    _start_segmentation_worker(widget, True, excl, None, mdir, mpfx)


def _segmentation_progress(widget, exclude_frame_indices, demo: bool = False):
    """Build a dock progress dict for Cellpose (GPU or CPU)."""
    use_gpu = core.use_gpu()
    desc = "Cellpose (GPU)" if use_gpu else "Cellpose (CPU)"
    try:
        data = np.squeeze(layer_as_numpy(widget.parent.selected_image_layer()))
    except Exception:
        return {"desc": desc}
    if demo and data.ndim >= 3:
        data = data[0:5]
    excl = exclude_frame_indices or frozenset()
    if data.ndim == 2:
        total = 0 if 0 in excl else 1
        return {"total": total, "desc": desc} if total else None
    if data.ndim >= 3:
        n_t = data.shape[0]
        total = sum(1 for i in range(n_t) if i not in excl)
        return {"total": total, "desc": desc} if total else None
    return None


def _paint_dock_progress(parent) -> None:
    """Force the dock bar and status label to paint (avoids stale n/total text)."""
    status = getattr(parent, "status_label", None)
    bar = getattr(parent, "progress_bar", None)
    if status is not None:
        status.repaint()
    if bar is not None:
        bar.repaint()


class _GuiInvoker(QObject):
    """Marshal callables onto the GUI thread that owns the dock widget."""

    _call = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self._call.connect(self._invoke, Qt.QueuedConnection)

    def _invoke(self, fn):
        fn()

    def post(self, fn) -> None:
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            fn()
            return
        self._call.emit(fn)


def _gui_invoker(parent) -> _GuiInvoker:
    inv = getattr(parent, "_mmv_gui_invoker", None)
    if inv is None:
        inv = _GuiInvoker(parent)
        parent._mmv_gui_invoker = inv
    return inv


def _run_on_gui_thread(parent, fn) -> None:
    """
    Run ``fn`` on the GUI thread that owns ``parent``.

    Bare callables connected to napari worker signals often use a direct
    connection and run on the worker thread; a queued signal through a
    child ``QObject`` keeps widget updates on the GUI thread.
    """
    _gui_invoker(parent).post(fn)


def _apply_dock_progress_widgets(parent, *, n: int, total: int, desc: str) -> None:
    """Update bar + status together on the GUI thread."""
    if total > 0:
        pct = int(round(100.0 * n / total))
        parent.set_progress_value(pct)
        parent.progress_bar.setFormat("%p%")
        parent.set_status_text(f"{desc} {n}/{total}")
    else:
        parent.progress_bar.setFormat("%p%")
        parent.set_status_text(desc)
    _paint_dock_progress(parent)


def _wire_dock_progress(parent, worker, progress: dict | None) -> None:
    """Drive the plugin progress bar/status from worker ``yielded`` signals.

    Prefer :func:`run_with_dock_progress` with ``mode="yield"`` for new call
    sites. Do not pass ``_progress=`` to napari ``create_worker``: napari's
    activity bar calls ``QApplication.processEvents()`` inside ``setValue``,
    which re-enters on rapid yields and stack-overflows (especially on Windows).

    If ``progress["absolute"]`` is true, each yielded int is treated as the
    completed count (``n`` out of ``total``); otherwise each yield increments
    by one (used when completion order is unordered).

    Always marks the plugin busy (buttons disabled) for the duration of the job.
    """
    if hasattr(parent, "set_plugin_busy"):
        parent.set_plugin_busy(True)
    if progress is None:
        return
    total = int(progress.get("total") or 0)
    desc = str(progress.get("desc") or "Working")
    absolute = bool(progress.get("absolute"))
    if total > 0:
        parent.set_progress_range(0, 100)
        _apply_dock_progress_widgets(parent, n=0, total=total, desc=desc)
        done = {"n": 0}

        def _on_yielded(value):
            def _update():
                if absolute:
                    done["n"] = max(0, min(total, int(value)))
                else:
                    done["n"] += 1
                    if done["n"] > total:
                        done["n"] = total
                _apply_dock_progress_widgets(
                    parent, n=done["n"], total=total, desc=desc
                )

            _run_on_gui_thread(parent, _update)

        worker.yielded.connect(_on_yielded)
    else:
        parent.set_progress_range(0, 0)
        parent.progress_bar.setFormat("%p%")
        parent.set_status_text(desc)
        _paint_dock_progress(parent)


def _reset_dock_progress(parent) -> None:
    """After a job: leave determinate bars at 100%, otherwise a blank idle bar."""
    bar = parent.progress_bar
    if bar.maximum() > 0:
        bar.setValue(bar.maximum())
        bar.setFormat("%p%")
    else:
        bar.setRange(0, 1)
        bar.setValue(0)
        bar.setFormat("")
    parent.clear_status()
    if hasattr(parent, "set_plugin_busy"):
        parent.set_plugin_busy(False)


def run_with_dock_progress(
    parent,
    worker_fn,
    *args,
    on_returned=None,
    on_errored=None,
    desc: str = "Working",
    total: int = 0,
    finish_desc: str | None = None,
    mode: str = "reporter",
    progress: dict | None = None,
    idle_status: str | None = None,
    pass_reporter: bool = True,
    skip_none_result: bool = False,
):
    """
    Start a napari worker and drive the plugin dock progress bar.

    Use this as the single entry point for progress-tracked jobs. Two modes:

    **``mode="reporter"``** (default)
        Creates a :class:`DockProgressReporter` (queued signal / side-channel).
        Prefer for pool-backed work or loops that call ``reporter.increment()``.
        If ``pass_reporter`` is true (default), ``reporter`` is appended as the
        last positional argument to ``worker_fn``.
        If ``finish_desc`` is set, bar total is ``total + 1`` so the GUI
        callback can claim the last step (e.g. adding layers) before reset.

    **``mode="yield"``**
        Wires :func:`_wire_dock_progress` to the worker's ``yielded`` signal.
        Prefer for generator workers that ``yield`` progress. Pass ``progress``
        as ``{"total": N, "desc": "...", "absolute": bool}`` or ``None`` for
        busy-only (optional ``idle_status`` text). Does not pass a reporter.

    Never pass napari ``_progress=`` to ``create_worker`` (Windows stack overflow).

    On success: stops any reporter, optionally shows ``finish_desc``, runs
    ``on_returned(result)``, then resets the dock. On error: resets, runs
    optional ``on_errored(exc)``, then ``handle_exception``.
    """
    mode = str(mode or "reporter").lower()
    if mode not in ("reporter", "yield"):
        raise ValueError(f"Unknown dock progress mode: {mode!r}")

    reporter = None
    if mode == "reporter":
        worker_steps = max(0, int(total))
        bar_total = worker_steps + (1 if finish_desc else 0)
        reporter = DockProgressReporter(parent, bar_total, desc)
        reporter.start()
        worker_args = (*args, reporter) if pass_reporter else args
        worker = create_worker(worker_fn, *worker_args, _start_thread=True)
        worker._dock_progress_reporter = reporter
    else:
        bar_total = 0
        # Intentionally no napari ``_progress`` — see ``_wire_dock_progress``.
        worker = create_worker(worker_fn, *args, _start_thread=True)
        _wire_dock_progress(parent, worker, progress)
        if progress is None and idle_status:
            parent.set_status_text(idle_status)

    def _on_returned(result):
        def _run():
            _stop_worker_progress_reporter(worker)
            try:
                if finish_desc and mode == "reporter":
                    parent.set_status_text(f"{finish_desc} {bar_total}/{bar_total}")
                    parent.set_progress_value(100)
                    parent.progress_bar.setFormat("%p%")
                    _paint_dock_progress(parent)
                if on_returned is not None and not (
                    skip_none_result and result is None
                ):
                    on_returned(result)
            finally:
                _reset_dock_progress(parent)

        _run_on_gui_thread(parent, _run)

    def _on_errored(exc):
        def _run():
            _stop_worker_progress_reporter(worker)
            _reset_dock_progress(parent)
            if on_errored is not None:
                on_errored(exc)
            handle_exception(exc)

        _run_on_gui_thread(parent, _run)

    worker.returned.connect(_on_returned)
    worker.errored.connect(_on_errored)
    return worker


class DockProgressReporter(QObject):
    """Thread-safe progress reporter for the dock bar.

    Prefer starting jobs via :func:`run_with_dock_progress` (``mode="reporter"``).
    ``increment`` / ``set_n`` may be called from a worker thread; each change is
    delivered to the GUI thread with a queued signal (no polling timer).

    The bar uses a 0–100 range so ``%p%`` matches the ``n/total`` fraction
    shown in the status label.
    """

    _progress = Signal(int, int)  # n, epoch

    def __init__(
        self,
        parent,
        total: int,
        desc: str,
        *,
        interval_ms: int = 50,
    ):
        super().__init__(parent)
        self._parent = parent
        self.total = max(0, int(total))
        self.desc = str(desc)
        self._n = 0
        self._epoch = 0
        self._lock = threading.Lock()
        self._progress.connect(self._apply_progress, Qt.QueuedConnection)
        _ = interval_ms  # retained for call-site compatibility

    def start(self) -> None:
        if hasattr(self._parent, "set_plugin_busy"):
            self._parent.set_plugin_busy(True)
        if self.total > 0:
            self._parent.set_progress_range(0, 100)
            _apply_dock_progress_widgets(
                self._parent, n=0, total=self.total, desc=self.desc
            )
        else:
            self._parent.set_progress_range(0, 0)
            self._parent.progress_bar.setFormat("%p%")
            self._parent.set_status_text(self.desc)
            _paint_dock_progress(self._parent)

    def increment(self, step: int = 1) -> int:
        with self._lock:
            self._n = (
                min(self.total, self._n + int(step)) if self.total else self._n + int(step)
            )
            n = self._n
            epoch = self._epoch
        self._progress.emit(n, epoch)
        return n

    def set_n(self, n: int) -> None:
        with self._lock:
            self._n = max(0, int(n))
            if self.total:
                self._n = min(self.total, self._n)
            n = self._n
            epoch = self._epoch
        self._progress.emit(n, epoch)

    def _apply_progress(self, n: int, epoch: int) -> None:
        # Drop stale queued updates after stop() bumps the epoch.
        if epoch != self._epoch:
            return
        _apply_dock_progress_widgets(
            self._parent, n=n, total=self.total, desc=self.desc
        )

    def stop(self) -> None:
        """Push a final UI update (call from the GUI thread when possible)."""
        with self._lock:
            self._epoch += 1
            n = self._n
            epoch = self._epoch
        # Apply synchronously when already on this QObject's thread so finish_desc
        # cannot race ahead of a still-queued progress event.
        if QThread.currentThread() is self.thread():
            self._apply_progress(n, epoch)
        else:
            self._progress.emit(n, epoch)


def _stop_worker_progress_reporter(worker) -> None:
    reporter = getattr(worker, "_dock_progress_reporter", None)
    if reporter is not None:
        reporter.stop()
        worker._dock_progress_reporter = None


def _start_segmentation_worker(
    widget,
    demo,
    exclude_frame_indices,
    copy_excluded_frames_from_layer,
    excluded_frames_masks_dir,
    excluded_frames_layer_prefix,
):
    """Start Cellpose on a worker with dock progress (CPU and GPU)."""
    parent = widget.parent
    progress = _segmentation_progress(widget, exclude_frame_indices, demo)
    worker_fn = _segment_image_gpu if core.use_gpu() else _segment_image_cpu
    backend = "GPU" if core.use_gpu() else "CPU"
    return run_with_dock_progress(
        parent,
        worker_fn,
        widget,
        demo,
        exclude_frame_indices,
        copy_excluded_frames_from_layer,
        excluded_frames_masks_dir,
        excluded_frames_layer_prefix,
        mode="yield",
        progress=progress,
        idle_status=f"Cellpose ({backend}) — running…",
        on_returned=_add_segmentation_to_viewer,
    )


def _add_segmentation_to_viewer(widget_and_mask):
    """
    Adds the segmentation as a layer to the viewer with a specified name

    Parameters
    ----------
    mask : array
        the segmentation data to add to the viewer
    """
    widget, mask = widget_and_mask
    labels = widget.viewer.add_labels(mask, name="calculated segmentation")
    widget.parent.combobox_segmentation.setCurrentText(labels.name)
    notify("Segmentation finished.")


def _finalize_segmentation_mask(
    widget,
    mask,
    data_squeezed,
    removed_dims,
    exclude_set,
    copy_excluded_frames_from_layer,
    excluded_frames_masks_dir,
    excluded_frames_layer_prefix,
    demo,
):
    """Fill excluded frames, restore shape, cache, and return ``(widget, mask)``."""
    if (
        exclude_set
        and excluded_frames_masks_dir is not None
        and excluded_frames_layer_prefix
    ):
        _fill_excluded_frames_from_training_masks_dir(
            mask,
            exclude_set,
            Path(excluded_frames_masks_dir),
            excluded_frames_layer_prefix,
            data_squeezed.ndim,
        )
    elif exclude_set and copy_excluded_frames_from_layer:
        _fill_excluded_frames_from_labels_layer(
            widget.viewer,
            mask,
            exclude_set,
            copy_excluded_frames_from_layer,
            data_squeezed.ndim,
        )

    if isinstance(mask, list):
        if len(mask) > 0:
            first_shape = mask[0].shape
            if not all(m.shape == first_shape for m in mask):
                logger.warning(
                    f"Masks have different shapes. First: {first_shape}, others may differ."
                )
            mask = np.asarray(mask)
        else:
            raise ValueError("Mask list is empty - no segmentation results")

    if mask.dtype != np.int32 and mask.dtype != np.int64:
        mask = mask.astype(np.int32)

    for dim_idx in sorted(removed_dims):
        mask = np.expand_dims(mask, axis=dim_idx)

    if mask.size == 0:
        raise ValueError("Mask is empty after processing")
    if np.all(mask == 0):
        logger.warning(
            "Mask contains only zeros - no segmentation found. "
            "This may indicate a problem with the model or parameters. "
            "Mask shape=%s (check preceding 'Segmentation input resolution' log).",
            getattr(mask, "shape", None),
        )

    if not demo:
        widget.parent.align_cache = mask
    QApplication.restoreOverrideCursor()
    logger.info("Segmentation finished")
    return widget, mask


def _segment_image_cpu(
    widget,
    demo=False,
    exclude_frame_indices=None,
    copy_excluded_frames_from_layer=None,
    excluded_frames_masks_dir: Path | None = None,
    excluded_frames_layer_prefix: str | None = None,
):
    """
    CPU Cellpose. Yields once per segmented frame for ``create_worker`` progress.

    Parallel path streams results via ``iter_starmap_as_completed`` so the dock
    progress bar advances as worker processes finish (not only at the end).
    """
    logger.info("Starting segmentation")
    QApplication.setOverrideCursor(Qt.WaitCursor)

    data_squeezed, removed_dims = _load_segmentation_image_data(widget, demo)
    exclude_set = exclude_frame_indices or frozenset()
    selected_model = widget.combobox_cellpose_model.currentText()
    parameters = _get_parameters(widget, selected_model)

    logger.info("Using CPU for segmentation")
    amount_of_processes = widget.parent.get_process_limit()
    logger.debug("Amount of processes: " + str(amount_of_processes))

    if data_squeezed.ndim == 2:
        if 0 in exclude_set:
            mask = np.zeros_like(data_squeezed, dtype=np.int32)
        else:
            mask = segment_slice_cpu(data_squeezed, parameters)
            yield 0
    else:
        n_t = data_squeezed.shape[0]
        mask = np.zeros((n_t, *data_squeezed.shape[1:]), dtype=np.int32)
        indices_to_run = [i for i in range(n_t) if i not in exclude_set]
        if indices_to_run:
            data_with_parameters = [
                (data_squeezed[i], parameters) for i in indices_to_run
            ]
            for task_i, layer_mask in iter_starmap_as_completed(
                segment_slice_cpu, data_with_parameters, amount_of_processes
            ):
                idx = indices_to_run[task_i]
                mask[idx] = layer_mask
                yield idx

    return _finalize_segmentation_mask(
        widget,
        mask,
        data_squeezed,
        removed_dims,
        exclude_set,
        copy_excluded_frames_from_layer,
        excluded_frames_masks_dir,
        excluded_frames_layer_prefix,
        demo,
    )


def _segment_image_gpu(
    widget,
    demo=False,
    exclude_frame_indices=None,
    copy_excluded_frames_from_layer=None,
    excluded_frames_masks_dir: Path | None = None,
    excluded_frames_layer_prefix: str | None = None,
):
    """
    GPU Cellpose. Yields once per segmented frame for ``create_worker`` progress.
    """
    logger.info("Starting segmentation")
    QApplication.setOverrideCursor(Qt.WaitCursor)

    data_squeezed, removed_dims = _load_segmentation_image_data(widget, demo)
    exclude_set = exclude_frame_indices or frozenset()
    selected_model = widget.combobox_cellpose_model.currentText()
    parameters = _get_parameters(widget, selected_model)

    logger.info("Using GPU for segmentation")
    model = models.CellposeModel(
        gpu=True, pretrained_model=parameters.pop("model_path")
    )
    if data_squeezed.ndim == 2:
        if 0 in exclude_set:
            mask = np.zeros_like(data_squeezed, dtype=np.int32)
        else:
            mask, _, _ = model.eval(data_squeezed, **parameters)
            yield 0
    else:
        n_t = data_squeezed.shape[0]
        mask = np.zeros((n_t, *data_squeezed.shape[1:]), dtype=np.int32)
        frames_to_run = [i for i in range(n_t) if i not in exclude_set]
        for i in frames_to_run:
            layer_mask, _, _ = model.eval(data_squeezed[i], **parameters)
            mask[i] = layer_mask
            yield i

    return _finalize_segmentation_mask(
        widget,
        mask,
        data_squeezed,
        removed_dims,
        exclude_set,
        copy_excluded_frames_from_layer,
        excluded_frames_masks_dir,
        excluded_frames_layer_prefix,
        demo,
    )


def _get_parameters(widget, model: str):
    """
    Get the parameters for the selected model

    Parameters
    ----------
    model : String
        The selected model

    Returns
    -------
    dict
        a dictionary of all the parameters based on selected model

    Raises
    ------
    ValueError
        If ``model`` is not a known hardcoded or registered custom model.
    """
    models_root = package_models_dir()

    if model == "Neutrophil_granulocytes":
        return {
            "model_path": str(models_root / model),
            "diameter": 15,
            "channels": [0, 0],
            "flow_threshold": 0.4,
            "cellprob_threshold": 0,
        }
    if model == "cpsam":
        return {
            "model_path": str(models_root / model),
            "flow_threshold": 0.4,
            "cellprob_threshold": 0,
        }
    if model.startswith(CUSTOM_MODEL_PREFIX):
        key = model[len(CUSTOM_MODEL_PREFIX) :]
        if key in widget.custom_models:
            entry = widget.custom_models[key]
            params = dict(entry["params"])
            params["model_path"] = str(
                get_custom_model_store().weights_path(entry["filename"])
            )
            return params

    raise ValueError(f"Unknown model: {model!r}")


def _track_segmentation(widget, on_returned=None):
    """
    Start coordinate-based tracking with side-channel dock progress.

    Returns a started napari worker (no ``_progress`` / no generator yields for
    progress — see ``DockProgressReporter``). If ``on_returned`` is set it is
    called with the tracks array after progress is torn down.
    """
    parent = widget.parent
    try:
        data = _get_segmentation_data(widget)
    except ValueError as exc:
        handle_exception(exc)
        return None

    n_frames = int(len(data))
    n_pairs = max(0, n_frames - 1)
    total = n_frames + n_pairs
    return run_with_dock_progress(
        parent,
        _worker_track_segmentation,
        widget,
        data,
        desc="Coordinate tracking",
        total=total,
        on_returned=on_returned,
    )


def _worker_track_segmentation(widget, data, reporter: DockProgressReporter):
    """
    Coordinate tracking. Reports progress via ``reporter`` (side channel);
    returns the tracks array.
    """
    starttime = time.time()
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        n_workers = widget.parent.get_process_limit()
        n_frames = int(len(data))

        extended_centroids = [None] * n_frames
        for task_i, result in iter_map_as_completed(
            calculate_centroids, data, n_workers
        ):
            extended_centroids[task_i] = result
            reporter.increment()
        time3 = time.time()
        logger.info(f"calculating centroids took {time3 - starttime} seconds")

        slice_pairs = [
            (extended_centroids[i - 1], extended_centroids[i])
            for i in range(1, n_frames)
        ]
        matches = [None] * len(slice_pairs)
        for task_i, result in iter_map_as_completed(
            match_centroids, slice_pairs, n_workers
        ):
            matches[task_i] = result
            reporter.increment()
        time4 = time.time()
        logger.info(f"matching centroids took {time4 - time3} seconds")

        tracks = _process_matches(matches)
        time5 = time.time()
        logger.info(f"processing matches took {time5 - time4} seconds")
        return tracks
    finally:
        QApplication.restoreOverrideCursor()


def _get_segmentation_data(widget):
    """
    Get the segmentation data from the viewer

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    array
        the segmentation data as a numpy array
    """
    try:
        label_layer = widget.parent.selected_labels_layer()
    except ValueError as exc:
        raise ValueError("Segmentation layer not found in viewer") from exc

    return layer_as_numpy(label_layer)


def _check_for_tracks_layer(widget):
    """
    Check if there is a tracks layer in the viewer

    Parameters
    ----------
    widget : QWidget
        the widget containing the viewer and the comboboxes

    Returns
    -------
    tracks_name : String
        the name of the tracks layer
    collision : Boolean
        whether or not there is a tracks layer in the viewer
    """
    tracks_name = DEFAULT_TRACKS_LAYER_NAME
    collision = True
    try:
        tracks_layer = widget.parent.selected_tracks_layer()
    except ValueError:
        collision = False
    else:
        tracks_name = tracks_layer.name
    return tracks_name, collision


def _calculate_centroids_parallel(widget, data):
    """
    Calculate the centroids of objects in a 2D slice.
    """
    return map_parallel(
        calculate_centroids, data, widget.parent.get_process_limit()
    )


def _match_centroids_parallel(widget, extended_centroids):
    """
    Match centroids between two slices.
    """
    slice_pairs = [
        (extended_centroids[i - 1], extended_centroids[i])
        for i in range(1, len(extended_centroids))
    ]
    return map_parallel(
        match_centroids, slice_pairs, widget.parent.get_process_limit()
    )


def _process_matches(matches):
    """
    Process the matches to create the tracks

    Parameters
    ----------
    matches : list
        the list of matches

    Returns
    -------
    array
        the tracks
    """
    # Initialize variables to store tracks, unique ID and visited cells
    tracks = np.array([])
    next_id = 0
    visited = [[0] * len(matches[i]) for i in range(len(matches))]

    # Helper function to append entry to tracks
    def process_entry(entry, tracks):
        """
        Process an entry to add to the tracks"""
        try:
            tracks = np.append(tracks, np.array([entry]), axis=0)
        except ValueError:
            tracks = np.array([entry])
        return tracks

    # Create an iterator to traverse through all slices and cells
    iterator = iter(
        ((i, j) for i in range(len(visited)) for j in range(len(visited[i])))
    )

    # Iterate through slices and cells
    for slice_id, cell_id in iterator:
        if visited[slice_id][cell_id]:
            continue

        # Extract centroid information for parent and child cells
        entry = [
            next_id,
            slice_id,
            int(matches[slice_id][cell_id]["parent"]["centroid"][0]),
            int(matches[slice_id][cell_id]["parent"]["centroid"][1]),
        ]
        tracks = process_entry(entry, tracks)

        entry = [
            next_id,
            slice_id + 1,
            int(matches[slice_id][cell_id]["child"]["centroid"][0]),
            int(matches[slice_id][cell_id]["child"]["centroid"][1]),
        ]
        tracks = process_entry(entry, tracks)

        visited[slice_id][cell_id] = 1
        label = matches[slice_id][cell_id]["child"]["id"]

        # Iterate through subsequent slices to complete the track
        while True:
            if slice_id + 1 >= len(matches):
                break
            labels = [
                matches[slice_id + 1][matched_cell]["parent"]["id"]
                for matched_cell in range(len(matches[slice_id + 1]))
            ]

            if label not in labels:
                break

            match_number = labels.index(label)
            visited[slice_id + 1][match_number] = 1
            entry = [
                next_id,
                slice_id + 2,
                int(matches[slice_id + 1][match_number]["child"]["centroid"][0]),
                int(matches[slice_id + 1][match_number]["child"]["centroid"][1]),
            ]
            tracks = process_entry(entry, tracks)
            label = matches[slice_id + 1][match_number]["child"]["id"]

            slice_id += 1

        next_id += 1

    return tracks.astype(int)


def scan_mmvh4tracks_training_temp_on_startup(main_widget) -> None:
    """
    On plugin load: clean stale training temp dirs; offer to resume interrupted training.
    """
    from ._train import (
        _safe_rmtree,
        classify_mmvh4tracks_training_dir,
        iter_mmvh4tracks_train_directories,
        parse_layer_prefix_and_frames_from_masks_dir,
        parse_model_fragment_from_train_dir_name,
    )

    seg = main_widget.segmentation_window
    dirs = list(iter_mmvh4tracks_train_directories())
    interrupted: list[Path] = []
    for d in dirs:
        kind = classify_mmvh4tracks_training_dir(d)
        if kind in ("empty", "masks_only", "incomplete"):
            _safe_rmtree(d)
        elif kind == "interrupted":
            interrupted.append(d)

    for d in interrupted:
        mtime = datetime.fromtimestamp(d.stat().st_mtime)
        day_str = mtime.strftime("%Y-%m-%d %H:%M")
        fragment = parse_model_fragment_from_train_dir_name(d.name) or d.name
        with awaiting_user_dialog(main_widget):
            reply = QMessageBox.question(
                main_widget,
                "napari",
                f"It seems training on data {fragment} on {day_str} was interrupted.\n\n"
                "Would you like to try again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
        if reply != QMessageBox.Yes:
            _safe_rmtree(d)
            continue
        model_name = custom_model_weights_basename(fragment)
        if is_custom_model_display_name_taken(seg, model_name):
            notify(
                f"A custom model named {model_name!r} already exists. "
                "Removing interrupted export."
            )
            _safe_rmtree(d)
            continue
        parsed = parse_layer_prefix_and_frames_from_masks_dir(d)
        if not parsed:
            _safe_rmtree(d)
            continue
        layer_prefix, train_frames = parsed
        start_cellpose_training_worker(
            seg,
            d,
            on_returned=lambda r, mn=model_name, tf=train_frames, lp=layer_prefix: (
                seg._complete_cellpose_training_after_worker(r, mn, tf, lp)
            ),
        )
        return


def remove_frame_from_track(tracks, track_entry):
    """Handles the logic for removing a frame from a track.
    Includes splitting the track if necessary."""
    # get index, track id and frame of track entry
    index = np.where(np.all(tracks == track_entry, axis=1))[0][0]
    track_id, frame, _, _ = track_entry
    indices_to_remove = [index]

    # get all track entries with the same track id
    track = tracks[tracks[:, 0] == track_id]

    # if entries with lower or higher frame exist:
    # connection(s) must be removed
    if np.any(track[:, 1] < frame) or np.any(track[:, 1] > frame):

        # get the entries with lower and higher frame
        lower_entries = track[track[:, 1] < frame]
        lower_indices = np.where(
            np.any(np.all(tracks[:, None] == lower_entries, axis=2), axis=1)
        )[0]
        higher_entries = track[track[:, 1] > frame]
        higher_indices = np.where(
            np.any(np.all(tracks[:, None] == higher_entries, axis=2), axis=1)
        )[0]

        # if only one entry with either lower or higher exists, find index
        # and queue for removal
        if len(lower_indices) == 1:
            indices_to_remove.append(lower_indices[0])
        if len(higher_indices) == 1:
            indices_to_remove.append(higher_indices[0])

    # if more than the one entry needs to be removed no splitting
    # is necessary
    if len(indices_to_remove) < 2:
        # get new track id
        new_track_id = lowest_missing_int(tracks[:, 0])
        # remove only the one entry, relabel the following frames
        tracks[higher_indices, 0] = new_track_id
        tracks = np.delete(tracks, indices_to_remove, axis=0)
    else:
        tracks = np.delete(tracks, indices_to_remove, axis=0)

    return tracks


def lowest_missing_int(arr):
    num_set = set(arr) if not isinstance(arr, set) else arr
    i = 1
    while i in num_set:
        i += 1
    return i


def split_noncontinuous_tracks(tracks):
    """Split tracks that have frame gaps into separate track IDs.

    For each track ID, rows are ordered by frame. Contiguous frame runs keep the
    original ID for the first run; each later run gets a new ID. Indexing uses
    the actual row positions in ``tracks`` (not an assumed contiguous block).
    """
    if tracks is None or len(tracks) == 0:
        return tracks

    tracks = np.asarray(tracks)
    for track_id in np.unique(tracks[:, 0]):
        idxs = np.where(tracks[:, 0] == track_id)[0]
        if len(idxs) <= 1:
            continue

        idxs = idxs[np.argsort(tracks[idxs, 1], kind="stable")]
        frames = tracks[idxs, 1]
        gap_starts = np.where(np.diff(frames) != 1)[0] + 1
        if gap_starts.size == 0:
            continue

        boundaries = np.concatenate(([0], gap_starts, [len(idxs)]))
        used_ids = set(np.unique(tracks[:, 0]).tolist())
        for seg in range(1, len(boundaries) - 1):
            seg_idxs = idxs[boundaries[seg] : boundaries[seg + 1]]
            new_id = lowest_missing_int(used_ids)
            tracks[seg_idxs, 0] = new_id
            used_ids.add(new_id)

    return tracks


def remove_dot_tracks(tracks):
    """Remove tracks that are only one frame long."""
    # get unique track ids
    unique_track_ids = np.unique(tracks[:, 0])

    # iterate through all unique track ids
    for track_id in unique_track_ids:
        # get all entries with the current track id
        current_track = tracks[tracks[:, 0] == track_id]

        # if the track is only one frame long, remove it
        if len(current_track) == 1:
            tracks = np.delete(tracks, np.where(tracks[:, 0] == track_id), axis=0)

    return tracks
