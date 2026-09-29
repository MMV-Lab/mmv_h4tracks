"""Shared pytest fixtures for mmv_h4tracks tests."""

from __future__ import annotations

import logging

import pytest
from qtpy.QtWidgets import QMessageBox

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._custom_models import CustomModelStore, set_custom_model_store
from mmv_h4tracks._tests.fixture_helpers import reset_widget

_NOISY_PRODUCTION_LOGGER_NAMES = ("mmv_h4tracks._batch", "mmv_h4tracks._processing")


@pytest.fixture(scope="session", autouse=True)
def _quiet_production_loggers():
    """
    Route _batch.py/_processing.py's log output through pytest's own log
    capture instead of straight to the terminal.

    Both attach their own StreamHandler with propagate=False (so their
    output doesn't collide with Cellpose's root-logger setup in production -
    see _train.py's logging.basicConfig(force=True) during training). That
    same isolation means pytest's log-capturing (which hooks the root
    logger) never sees their records, so real batch-processing/segmentation
    runs in the test suite print straight to the console for every test,
    pass or fail. Safe to flip here since conftest.py never loads outside
    pytest - production behavior (and the reason for propagate=False there)
    is unaffected.
    """
    originals = {}
    for name in _NOISY_PRODUCTION_LOGGER_NAMES:
        logger = logging.getLogger(name)
        originals[name] = (logger.propagate, list(logger.handlers))
        logger.propagate = True
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    yield
    for name, (propagate, handlers) in originals.items():
        logger = logging.getLogger(name)
        logger.propagate = propagate
        for handler in handlers:
            logger.addHandler(handler)


@pytest.fixture(scope="session", autouse=True)
def _isolate_custom_model_store(tmp_path_factory):
    """Keep tests off the real ``~/.mmv_h4tracks`` registry."""
    store = CustomModelStore(tmp_path_factory.mktemp("mmv_custom_models"))
    set_custom_model_store(store)
    yield store
    set_custom_model_store(None)


@pytest.fixture(scope="session", autouse=True)
def _no_blocking_message_boxes():
    """
    Prevent modal QMessageBox dialogs from hanging the suite.

    Session-scoped so module-scoped widget setup (warm-up / resume-training
    scan) is covered before any function-scoped fixtures run.
    ``notify`` / ``choice_dialog`` and raw ``QMessageBox`` usage all call
    ``exec`` / ``question``; stub them so tests never wait for a click.
    Per-test ``@patch(...notify)`` still overrides where asserted.
    """
    original_exec = getattr(QMessageBox, "exec", None)
    original_exec_ = getattr(QMessageBox, "exec_", None)
    original_question = QMessageBox.question

    def _exec_ok(self, *args, **kwargs):
        return QMessageBox.Ok

    def _question_no(*args, **kwargs):
        return QMessageBox.No

    QMessageBox.exec = _exec_ok
    if original_exec_ is not None:
        QMessageBox.exec_ = _exec_ok
    QMessageBox.question = staticmethod(_question_no)
    try:
        yield
    finally:
        if original_exec is not None:
            QMessageBox.exec = original_exec
        if original_exec_ is not None:
            QMessageBox.exec_ = original_exec_
        QMessageBox.question = original_question


@pytest.fixture(scope="module")
def module_napari_viewer(qapp):
    """One napari viewer per test module (reused across that file's tests)."""
    import napari

    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


@pytest.fixture(scope="module")
def module_widget(module_napari_viewer):
    """One MMVH4TRACKS instance per test module."""
    from qtpy.QtWidgets import QApplication

    import mmv_h4tracks._processing as processing

    def _instant_warmup(*, on_finished):
        on_finished(False)

    previous = processing.start_cellpose_warmup
    processing.start_cellpose_warmup = _instant_warmup
    try:
        widget = MMVH4TRACKS(module_napari_viewer)
        # Flush QTimer warm-up (and post-warm-up training-temp scan).
        QApplication.processEvents()
        yield widget
    finally:
        processing.start_cellpose_warmup = previous


@pytest.fixture
def create_widget(module_widget):
    """
    Function-scoped clean widget: same module instance, reset before each test.

    Teardown reset is skipped — the next test's setup clears again.
    """
    reset_widget(module_widget)
    yield module_widget
