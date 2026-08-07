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
