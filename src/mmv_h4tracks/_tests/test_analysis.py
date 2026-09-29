"""Module providing tests for the analysis tab's plot window"""

import numpy as np
import pytest

from qtpy.QtWidgets import QAbstractButton, QPushButton


@pytest.fixture
def widget(create_widget):
    yield create_widget
    # _plot() shows plot_window as an unparented top-level window; the shared
    # cleanup in fixture_helpers only closes it as *setup* for the next test,
    # so a test that's the last in this module would otherwise leave it open.
    plot_window = getattr(create_widget, "plot_window", None)
    if plot_window is not None:
        plot_window.close()
        create_widget.plot_window = None


def test_plot_adds_reset_view_and_apply_buttons(widget):
    """The plot window gets a smaller Reset view button next to a larger Apply button"""
    plot_dict = {
        "Name": "Speed [px/frame]",
        "Description": "Scatterplot Standard Deviation vs Average: Speed",
        "x_label": "Average",
        "y_label": "Standard Deviation",
        "Results": np.array([[0, 1, 1], [1, 5, 5], [2, 9, 9]]),
    }

    widget.analysis_window._plot(plot_dict)

    buttons = widget.plot_window.findChildren(QPushButton)
    labels = [button.text() for button in buttons]
    assert "Reset view" in labels
    assert "Apply" in labels

    reset_view = next(button for button in buttons if button.text() == "Reset view")
    apply_btn = next(button for button in buttons if button.text() == "Apply")
    # Apply has layout stretch and grows to fill the row; Reset view has none
    # and stays at its natural (smaller) size.
    row_layout = reset_view.parent().layout()
    assert row_layout.stretch(row_layout.indexOf(reset_view)) == 0
    assert row_layout.stretch(row_layout.indexOf(apply_btn)) > 0


def test_reset_view_button_resets_the_view(widget):
    """Clicking Reset view restores the view after zooming"""
    plot_dict = {
        "Name": "Speed [px/frame]",
        "Description": "Scatterplot Standard Deviation vs Average: Speed",
        "x_label": "Average",
        "y_label": "Standard Deviation",
        "Results": np.array([[0, 1, 1], [1, 5, 5], [2, 9, 9]]),
    }

    widget.analysis_window._plot(plot_dict)
    selector = widget.analysis_window.selector
    home_xlim = selector.ax.get_xlim()

    selector.ax.set_xlim(home_xlim[0] + 1, home_xlim[1] + 1)
    assert selector.ax.get_xlim() != home_xlim

    buttons = widget.plot_window.findChildren(QPushButton)
    reset_view = next(button for button in buttons if button.text() == "Reset view")
    reset_view.click()

    assert selector.ax.get_xlim() == home_xlim


@pytest.fixture
def action_cam_button(widget):
    buttons = widget.analysis_window.findChildren(QPushButton)
    return next(b for b in buttons if b.text() == "Open Action Cam")


def _close_action_cam(widget):
    if getattr(widget, "action_cam_window", None) is not None:
        widget.viewer.window.remove_dock_widget(widget.action_cam_window)
        widget.action_cam_window = None
        widget._action_cam_dock = None


def test_action_cam_button_opens_the_dock_widget(widget, action_cam_button):
    """Clicking "Open Action Cam" must create and dock an ActionCamWindow,
    not just switch to an existing tab - there is no tab anymore."""
    assert widget.action_cam_window is None

    try:
        action_cam_button.click()
        assert widget.action_cam_window is not None
        assert widget._action_cam_dock is not None
    finally:
        _close_action_cam(widget)


def test_action_cam_button_reuses_the_panel_on_a_second_click(widget, action_cam_button):
    """Reopening must not lose a previously loaded track - reuse the same
    instance instead of reconstructing it (unlike the Plot window, which is
    deliberately rebuilt fresh each time)."""
    try:
        action_cam_button.click()
        first_instance = widget.action_cam_window

        action_cam_button.click()
        assert widget.action_cam_window is first_instance
    finally:
        _close_action_cam(widget)


def test_action_cam_opens_as_a_floating_window(widget, action_cam_button):
    """Not merged into a dock edge, so it can't be mistaken for part of the
    main layout."""
    try:
        action_cam_button.click()
        assert widget._action_cam_dock.isFloating() is True
    finally:
        _close_action_cam(widget)


def test_open_action_cam_recovers_if_the_previous_window_was_destroyed(
    widget, monkeypatch
):
    """A third click (etc.) must never error, whether the panel is still
    open, was hidden/closed, or its underlying Qt object was actually
    destroyed (e.g. by however the OS/napari cleaned up its close button)."""
    try:
        widget.open_action_cam()
        first_instance = widget.action_cam_window

        def _raise_deleted(*args, **kwargs):
            raise RuntimeError(
                "wrapped C/C++ object of type QtViewerDockWidget has been deleted"
            )

        monkeypatch.setattr(widget._action_cam_dock, "show", _raise_deleted)

        widget.open_action_cam()  # must not raise

        assert widget.action_cam_window is not None
        assert widget.action_cam_window is not first_instance
    finally:
        _close_action_cam(widget)


def test_action_cam_button_is_disabled_while_the_plugin_is_busy(widget, action_cam_button):
    """The button must follow the same convention as every other button:
    set_plugin_busy sweeps QAbstractButton children of the main widget."""
    assert action_cam_button in widget.findChildren(QAbstractButton)

    widget.set_plugin_busy(True)
    assert not action_cam_button.isEnabled()

    widget.set_plugin_busy(False)
    assert action_cam_button.isEnabled()
