"""Unit tests for shared concurrency helpers."""

import time

from mmv_h4tracks._concurrency import (
    iter_map_as_completed,
    iter_starmap_as_completed,
    map_parallel,
    should_use_pool,
    starmap_parallel,
)


def test_should_use_pool_thresholds():
    assert not should_use_pool(1, 100)
    assert not should_use_pool(8, 2)
    assert not should_use_pool(8, 0)
    assert should_use_pool(8, 3)


def test_map_parallel_serial_path():
    assert map_parallel(lambda x: x * 2, [1, 2, 3], n_workers=1) == [2, 4, 6]
    assert map_parallel(lambda x: x + 1, [1, 2], n_workers=8) == [2, 3]


def test_starmap_parallel_serial_path():
    def add(a, b):
        return a + b

    assert starmap_parallel(add, [(1, 2), (3, 4)], n_workers=1) == [3, 7]
    assert starmap_parallel(add, [(1, 1)], n_workers=8) == [2]


def test_iter_starmap_as_completed_serial_path():
    def add(a, b):
        return a + b

    pairs = list(iter_starmap_as_completed(add, [(1, 2), (3, 4)], n_workers=1))
    assert pairs == [(0, 3), (1, 7)]


def _slow_add(a, b):
    """Module-level so Windows spawn can pickle it."""
    time.sleep(0.05 if a == 1 else 0.01)
    return a + b


def test_iter_starmap_as_completed_streams_results():
    results = list(
        iter_starmap_as_completed(
            _slow_add, [(1, 0), (2, 0), (3, 0)], n_workers=2
        )
    )
    assert sorted(results) == [(0, 1), (1, 2), (2, 3)]


def test_iter_map_as_completed_serial_path():
    pairs = list(iter_map_as_completed(lambda x: x * 2, [1, 2, 3], n_workers=1))
    assert pairs == [(0, 2), (1, 4), (2, 6)]


def _slow_double(x):
    """Module-level so Windows spawn can pickle it."""
    time.sleep(0.02)
    return x * 2


def test_iter_map_as_completed_pool_path():
    results = list(
        iter_map_as_completed(_slow_double, [1, 2, 3], n_workers=2)
    )
    assert sorted(results) == [(0, 2), (1, 4), (2, 6)]
