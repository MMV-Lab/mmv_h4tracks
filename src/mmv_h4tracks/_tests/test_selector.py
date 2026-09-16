"""Module providing tests for the plot's zoom/pan selector"""

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mmv_h4tracks._selector import (
    Selector,
    compute_offscreen_counts,
    badge_size_for_count,
    OFFSCREEN_BADGE_MIN_SIZE,
    OFFSCREEN_BADGE_MAX_SIZE,
    OFFSCREEN_BADGE_REF_COUNT,
)
from mmv_h4tracks._analysis import (
    PLOT_TITLE_PAD,
    PLOT_LABELPAD,
    PLOT_MARGINS,
    PLOT_YLABEL_AXES_X,
)


def event(**kwargs):
    """A minimal stand-in for a matplotlib ``MouseEvent``"""
    defaults = {"inaxes": None, "xdata": None, "ydata": None, "x": 0, "y": 0, "button": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def selector():
    """A Selector on a headless figure with a few scattered points"""
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    results = np.array([[0, 1, 1], [1, 5, 5], [2, 9, 9]])
    return Selector(parent=SimpleNamespace(), ax=ax, results=results)


@pytest.fixture
def selector_with_chrome():
    """
    A Selector on a figure replicating the app's actual plot chrome: title/
    x-label/y-label at their real (larger than default) font sizes, and the
    fixed margin/pinned y-label position ``_plot()`` sets up. A first version
    of the arrows only overlapped these because a fixture without any of
    this couldn't have caught it.
    """
    fig = Figure(figsize=(8, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Speed [px/frame]", {"fontsize": 22}, pad=PLOT_TITLE_PAD)
    ax.set_xlabel("Average", fontsize=15, labelpad=PLOT_LABELPAD)
    ax.set_ylabel("Standard Deviation", fontsize=15, labelpad=PLOT_LABELPAD)
    fig.subplots_adjust(**PLOT_MARGINS)
    ax.yaxis.set_label_coords(PLOT_YLABEL_AXES_X, 0.5)
    results = np.array([[0, 1, 1], [1, 5, 5], [2, 9, 9]])
    return Selector(parent=SimpleNamespace(), ax=ax, results=results)


def test_scroll_up_zooms_in_around_cursor(selector):
    """Scrolling up shrinks the visible range and keeps the cursor's data point fixed"""
    selector._on_scroll(event(inaxes=selector.ax, xdata=5, ydata=5, button="up"))

    xlim = selector.ax.get_xlim()
    ylim = selector.ax.get_ylim()
    assert xlim[1] - xlim[0] < 10
    assert ylim[1] - ylim[0] < 10
    # cursor was at the center, so it should still be centered after zooming
    assert xlim[0] + xlim[1] == pytest.approx(10)
    assert ylim[0] + ylim[1] == pytest.approx(10)


def test_scroll_down_zooms_out(selector):
    """Scrolling down grows the visible range"""
    selector._on_scroll(event(inaxes=selector.ax, xdata=5, ydata=5, button="down"))

    xlim = selector.ax.get_xlim()
    assert xlim[1] - xlim[0] > 10


def test_scroll_off_axis_is_ignored(selector):
    """Scrolling outside the axes does not change the view"""
    selector._on_scroll(event(inaxes=None, xdata=5, ydata=5, button="up"))

    assert selector.ax.get_xlim() == (0, 10)
    assert selector.ax.get_ylim() == (0, 10)


def test_scroll_keeps_off_center_cursor_point_fixed(selector):
    """Zooming around a non-centered cursor position keeps that data point under it"""
    selector._on_scroll(event(inaxes=selector.ax, xdata=2, ydata=2, button="up"))

    xlim = selector.ax.get_xlim()
    ylim = selector.ax.get_ylim()
    relx = (2 - xlim[0]) / (xlim[1] - xlim[0])
    rely = (2 - ylim[0]) / (ylim[1] - ylim[0])
    assert relx == pytest.approx(0.2)
    assert rely == pytest.approx(0.2)


def test_right_drag_pans_the_view(selector):
    """Dragging right-click moves the visible range by the dragged amount"""
    selector._on_button_press(event(inaxes=selector.ax, button=3, x=100, y=100))
    # Drag 50 pixels right and 20 pixels up across a ~640x480 default axes bbox
    selector._on_motion(event(x=150, y=120))

    xlim = selector.ax.get_xlim()
    ylim = selector.ax.get_ylim()
    # Dragging right pans the view left (content follows the cursor)
    assert xlim[0] < 0
    assert ylim[0] < 0
    # Width/height are preserved, only the view shifted
    assert xlim[1] - xlim[0] == pytest.approx(10)
    assert ylim[1] - ylim[0] == pytest.approx(10)


def test_left_drag_does_not_pan(selector):
    """Only the right mouse button starts a pan"""
    selector._on_button_press(event(inaxes=selector.ax, button=1, x=100, y=100))
    selector._on_motion(event(x=200, y=200))

    assert selector.ax.get_xlim() == (0, 10)
    assert selector.ax.get_ylim() == (0, 10)


def test_button_release_stops_panning(selector):
    """Releasing the right button ends the pan, further motion has no effect"""
    selector._on_button_press(event(inaxes=selector.ax, button=3, x=100, y=100))
    selector._on_button_release(event(button=3))
    selector._on_motion(event(x=200, y=200))

    assert selector.ax.get_xlim() == (0, 10)
    assert selector.ax.get_ylim() == (0, 10)


def test_pan_without_press_is_a_noop(selector):
    """Motion events before any button press don't move the view"""
    selector._on_motion(event(x=200, y=200))

    assert selector.ax.get_xlim() == (0, 10)
    assert selector.ax.get_ylim() == (0, 10)


def test_reset_view_restores_the_initial_limits(selector):
    """reset_view undoes any zooming/panning back to the view at construction"""
    home_xlim, home_ylim = selector.ax.get_xlim(), selector.ax.get_ylim()

    selector._on_scroll(event(inaxes=selector.ax, xdata=5, ydata=5, button="up"))
    selector._on_button_press(event(inaxes=selector.ax, button=3, x=100, y=100))
    selector._on_motion(event(x=150, y=120))
    assert selector.ax.get_xlim() != home_xlim

    selector.reset_view()

    assert selector.ax.get_xlim() == home_xlim
    assert selector.ax.get_ylim() == home_ylim


class TestComputeOffscreenCounts:
    def test_counts_points_beyond_each_edge(self):
        """Each direction only counts points beyond that specific edge"""
        xys = [
            [5, 5],  # visible
            [-1, 5],  # left
            [11, 5],  # right
            [5, -1],  # bottom
            [5, 11],  # top
        ]
        counts = compute_offscreen_counts(xys, (0, 10), (0, 10))
        assert counts == {"left": 1, "right": 1, "bottom": 1, "top": 1}

    def test_corner_point_counts_toward_both_edges(self):
        """A point off-screen diagonally counts toward both edges it exceeds"""
        counts = compute_offscreen_counts([[-1, 11]], (0, 10), (0, 10))
        assert counts == {"left": 1, "right": 0, "bottom": 0, "top": 1}

    def test_no_points_offscreen(self):
        """All points visible: every direction is zero"""
        counts = compute_offscreen_counts([[5, 5], [1, 1]], (0, 10), (0, 10))
        assert counts == {"left": 0, "right": 0, "bottom": 0, "top": 0}

    def test_empty_points(self):
        """An empty point set never reports any off-screen points"""
        counts = compute_offscreen_counts(np.empty((0, 2)), (0, 10), (0, 10))
        assert counts == {"left": 0, "right": 0, "bottom": 0, "top": 0}


class TestBadgeSizeForCount:
    def test_single_point_is_minimum_size(self):
        """One off-screen point always gets the minimum badge size"""
        assert badge_size_for_count(1) == OFFSCREEN_BADGE_MIN_SIZE

    def test_reference_count_is_maximum_size(self):
        """Reaching the reference count saturates to the maximum badge size"""
        assert badge_size_for_count(OFFSCREEN_BADGE_REF_COUNT) == pytest.approx(
            OFFSCREEN_BADGE_MAX_SIZE
        )

    def test_beyond_reference_count_stays_capped(self):
        """More than the reference count never exceeds the maximum badge size"""
        assert badge_size_for_count(OFFSCREEN_BADGE_REF_COUNT * 100) == pytest.approx(
            OFFSCREEN_BADGE_MAX_SIZE
        )

    def test_size_grows_monotonically_with_count(self):
        """Badge size never shrinks as the off-screen count grows"""
        sizes = [badge_size_for_count(n) for n in (1, 2, 5, 10, 25, 50)]
        assert sizes == sorted(sizes)


class TestOffscreenIndicators:
    def test_badge_appears_for_offscreen_direction(self, selector):
        """Zooming in until points fall outside the view creates a badge"""
        # Fixture points are (1,1), (5,5), (9,9); narrowing x only pushes
        # (1,1) and (9,9) out left/right while all y values stay in view.
        selector.ax.set_xlim(4, 6)

        assert "left" in selector._offscreen_artists
        assert "right" in selector._offscreen_artists
        assert "bottom" not in selector._offscreen_artists
        assert "top" not in selector._offscreen_artists

    def test_badge_disappears_once_back_in_view(self, selector):
        """Badges are removed again once their direction has no off-screen points"""
        selector.ax.set_xlim(4, 6)
        assert "left" in selector._offscreen_artists

        selector.reset_view()

        assert selector._offscreen_artists == {}

    def test_no_badges_when_everything_is_visible(self, selector):
        """A fresh selector with all points in view starts with no badges"""
        assert selector._offscreen_artists == {}


def _artist_bbox_axes_fraction(selector, artist):
    """Bounding box of a rendered artist (marker or text), in axes-fraction."""
    renderer = selector.canvas.get_renderer()
    return artist.get_window_extent(renderer).transformed(
        selector.ax.transAxes.inverted()
    )


class TestOffscreenBadgesClearChrome:
    """
    Regression coverage for arrows overlapping the title/x-label/y-label.
    They sit in the pad/labelpad _plot() reserves for exactly this - between
    the axes and the label, not beyond it - so these use
    ``selector_with_chrome``, which replicates that padding.
    """

    def test_top_badge_clears_the_title(self, selector_with_chrome):
        """The arrow above the axes doesn't sit on top of the title"""
        # Fixture points are (1,1), (5,5), (9,9); narrowing the top of ylim
        # only pushes (9,9) off-screen at the top.
        selector = selector_with_chrome
        selector.ax.set_ylim(-1, 6)
        selector._update_offscreen_indicators()

        marker = selector._offscreen_artists["top"]
        title_bbox = _artist_bbox_axes_fraction(selector, selector.ax.title)
        marker_bbox = _artist_bbox_axes_fraction(selector, marker)
        assert marker_bbox.ymax <= title_bbox.ymin

    def test_bottom_badge_clears_the_xlabel(self, selector_with_chrome):
        """The arrow below the axes doesn't sit on top of 'Average'"""
        # Narrowing the bottom of ylim only pushes (1,1) off-screen at the bottom.
        selector = selector_with_chrome
        selector.ax.set_ylim(4, 11)
        selector._update_offscreen_indicators()

        marker = selector._offscreen_artists["bottom"]
        xlabel_bbox = _artist_bbox_axes_fraction(selector, selector.ax.xaxis.label)
        marker_bbox = _artist_bbox_axes_fraction(selector, marker)
        assert marker_bbox.ymin >= xlabel_bbox.ymax

    def test_left_badge_clears_the_ylabel(self, selector_with_chrome):
        """The arrow left of the axes doesn't sit on top of 'Standard Deviation'"""
        # Narrowing the left of xlim only pushes (1,1) off-screen to the left.
        selector = selector_with_chrome
        selector.ax.set_xlim(2, 10)
        selector._update_offscreen_indicators()

        marker = selector._offscreen_artists["left"]
        ylabel_bbox = _artist_bbox_axes_fraction(selector, selector.ax.yaxis.label)
        marker_bbox = _artist_bbox_axes_fraction(selector, marker)
        assert marker_bbox.xmin >= ylabel_bbox.xmax

    def test_left_badge_clears_the_ylabel_with_wide_tick_labels(
        self, selector_with_chrome
    ):
        """
        The y-label shifts further left as tick numbers get wider (more
        digits); the reserved labelpad needs to keep clearing it even so.
        """
        selector = selector_with_chrome
        selector.ax.set_xlim(2, 10)
        selector.ax.set_ylim(-123456, 10)  # wide (6-digit) tick labels
        selector._update_offscreen_indicators()

        marker = selector._offscreen_artists["left"]
        ylabel_bbox = _artist_bbox_axes_fraction(selector, selector.ax.yaxis.label)
        marker_bbox = _artist_bbox_axes_fraction(selector, marker)
        assert marker_bbox.xmin >= ylabel_bbox.xmax


def _artist_bbox_figure_fraction(selector, artist):
    """Bounding box of a rendered artist, in figure-fraction (the visible canvas)."""
    renderer = selector.canvas.get_renderer()
    figure = selector.ax.figure
    return artist.get_window_extent(renderer).transformed(figure.transFigure.inverted())


class TestYLabelStaysOnCanvas:
    """
    Regression coverage: a narrow decimal y-range (e.g. -0.75 to 0.75) makes
    matplotlib's default tick formatter emit longer-looking labels
    ("-0.75", not "0"), which - combined with the labelpad reserved for the
    off-screen arrow - could push the y-label past the figure's left edge
    entirely (a more subtle version of the wide-integer-tick-label case,
    since narrow float ranges aren't "wide" in digit count).
    """

    @pytest.mark.parametrize(
        "ylim",
        [
            (0, 10),  # normal
            (-0.75, 0.75),  # the reported case: decimal ticks look long
            (-99999, 10),  # 5-digit integer ticks
            (-999999, 10),  # 6-digit integer ticks
            (-12345.6789, 10),  # many decimal digits
            (-123456789, 10),  # large enough to trigger offset (x1eN) notation
        ],
    )
    def test_ylabel_is_not_clipped_past_the_figure_edge(self, selector_with_chrome, ylim):
        selector = selector_with_chrome
        selector.ax.set_ylim(*ylim)
        selector._update_offscreen_indicators()

        ylabel_bbox = _artist_bbox_figure_fraction(selector, selector.ax.yaxis.label)
        assert ylabel_bbox.xmin >= 0


class TestFrameNeverMoves:
    """
    Regression coverage for the frame "snapping": an earlier version
    recomputed the layout (tight_layout()) on every zoom/pan, which visibly
    shifted the axes box - and everything anchored to it - as tick labels
    changed width mid-interaction. The fixed margin/label position ``_plot()``
    sets up must make the axes box (and the y-label's position within it)
    completely independent of content, for any number of updates.
    """

    def test_axes_position_is_identical_across_tick_label_widths(
        self, selector_with_chrome
    ):
        """The axes box itself never moves, regardless of tick label content"""
        selector = selector_with_chrome
        initial_bounds = selector.ax.get_position().bounds

        for ylim in [(-0.75, 0.75), (-999999, 10), (-12345.6789, 10), (0, 10)]:
            selector.ax.set_ylim(*ylim)
            selector._update_offscreen_indicators()
            assert selector.ax.get_position().bounds == initial_bounds

    def test_ylabel_position_is_identical_across_tick_label_widths(
        self, selector_with_chrome
    ):
        """The y-label's anchor position never moves, regardless of tick content"""
        selector = selector_with_chrome
        initial_coords = selector.ax.yaxis.get_label().get_position()

        for ylim in [(-0.75, 0.75), (-999999, 10), (-12345.6789, 10), (0, 10)]:
            selector.ax.set_ylim(*ylim)
            selector._update_offscreen_indicators()
            assert selector.ax.yaxis.get_label().get_position() == initial_coords

    def test_repeated_updates_do_not_drift_the_axes_position(
        self, selector_with_chrome
    ):
        """Calling _update_offscreen_indicators many times in a row is a no-op for layout"""
        selector = selector_with_chrome
        initial_bounds = selector.ax.get_position().bounds

        for _ in range(20):
            selector._update_offscreen_indicators()

        assert selector.ax.get_position().bounds == initial_bounds
