"""Unit tests for shared concurrency helpers."""

from mmv_h4tracks._concurrency import (
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
