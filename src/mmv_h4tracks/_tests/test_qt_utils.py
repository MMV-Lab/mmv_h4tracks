"""Tests for Qt / napari helpers."""

import numpy as np
import pytest

from mmv_h4tracks._qt_utils import layer_as_numpy
from mmv_h4tracks._reader import build_multiscale


class _FakeLayer:
    def __init__(self, data, *, multiscale=False, name="fake"):
        self.data = data
        self.multiscale = multiscale
        self.name = name


class _FakeMultiScaleData:
    """Napari-like pyramid: ``.shape`` is the displayed (often coarsest) level."""

    def __init__(self, levels, displayed_index=-1):
        self._levels = list(levels)
        self._displayed = displayed_index

    def __len__(self):
        return len(self._levels)

    def __getitem__(self, i):
        return self._levels[i]

    @property
    def shape(self):
        return self._levels[self._displayed].shape

    def __array__(self, dtype=None):
        arr = np.asarray(self._levels[self._displayed])
        return arr if dtype is None else arr.astype(dtype)


@pytest.mark.unit
def test_layer_as_numpy_ignores_multiscale_flag_on_shaped_array():
    """Truthy ``multiscale`` must not treat a TZYX/ZYX ndarray as pyramid levels."""
    arr = np.arange(2 * 4 * 5).reshape(2, 4, 5)
    out = layer_as_numpy(_FakeLayer(arr, multiscale=True))
    assert out.shape == (2, 4, 5)
    assert np.array_equal(out, arr)


@pytest.mark.unit
def test_layer_as_numpy_single_scale():
    arr = np.arange(12).reshape(3, 4)
    out = layer_as_numpy(_FakeLayer(arr))
    assert out.shape == (3, 4)
    assert np.array_equal(out, arr)


@pytest.mark.unit
def test_layer_as_numpy_multiscale_picks_largest_even_if_not_first():
    fine = np.zeros((2, 64, 80), dtype=np.uint16)
    fine[0, 10:20, 10:20] = 1
    # Coarse-first order (would break naive data[0] selection)
    levels = [
        fine[..., ::8, ::8],
        fine[..., ::4, ::4],
        fine[..., ::2, ::2],
        fine,
    ]
    out = layer_as_numpy(_FakeLayer(levels, multiscale=True))
    assert out.shape == fine.shape
    assert int(out.max()) == 1


@pytest.mark.unit
def test_layer_as_numpy_build_multiscale_order():
    image = np.random.randint(0, 255, (10, 768, 1280), dtype=np.uint16)
    levels = build_multiscale(image)
    assert levels[0].shape == (10, 768, 1280)
    assert levels[-1].shape == (10, 96, 160)
    out = layer_as_numpy(_FakeLayer(levels, multiscale=True))
    assert out.shape == (10, 768, 1280)


@pytest.mark.unit
def test_layer_as_numpy_multiscaledata_shape_is_coarse_displayed_level():
    """np.asarray(MultiScaleData) follows .shape (coarse); we must still pick finest."""
    image = np.arange(10 * 768 * 1280, dtype=np.uint16).reshape(10, 768, 1280)
    levels = build_multiscale(image)
    data = _FakeMultiScaleData(levels, displayed_index=-1)
    assert data.shape == (10, 96, 160)
    assert np.asarray(data).shape == (10, 96, 160)
    out = layer_as_numpy(_FakeLayer(data, multiscale=True))
    assert out.shape == (10, 768, 1280)
    assert np.array_equal(out, image)
