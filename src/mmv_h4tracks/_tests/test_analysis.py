"""Module providing tests for the analysis tab's plot window"""

import numpy as np
import pytest

from qtpy.QtWidgets import QPushButton


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
