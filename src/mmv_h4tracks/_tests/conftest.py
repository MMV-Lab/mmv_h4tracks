"""Shared pytest fixtures for mmv_h4tracks tests."""

from __future__ import annotations

import pytest

from mmv_h4tracks import MMVH4TRACKS
from mmv_h4tracks._custom_models import CustomModelStore, set_custom_model_store
from mmv_h4tracks._tests.fixture_helpers import reset_widget


@pytest.fixture(scope="session", autouse=True)
def _isolate_custom_model_store(tmp_path_factory):
    """Keep tests off the real ``~/.mmv_h4tracks`` registry."""
    store = CustomModelStore(tmp_path_factory.mktemp("mmv_custom_models"))
    set_custom_model_store(store)
    yield store
    set_custom_model_store(None)


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
    widget = MMVH4TRACKS(module_napari_viewer)
    yield widget


@pytest.fixture
def create_widget(module_widget):
    """
    Function-scoped clean widget: same module instance, reset before each test.

    Teardown reset is skipped — the next test's setup clears again.
    """
    reset_widget(module_widget)
    yield module_widget
