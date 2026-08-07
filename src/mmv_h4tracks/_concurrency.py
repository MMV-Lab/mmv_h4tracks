"""Shared CPU process-budget and map/starmap helpers.

Keeps napari ``@thread_worker`` for UI responsiveness; this module only
decides whether work inside a worker uses ``multiprocessing.Pool`` or runs
in-process (avoiding Windows spawn cost for tiny / single-worker jobs).
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import Pool
from typing import Any, TypeVar

T = TypeVar("T")

# Match evaluation's previous threshold: pools are not worth it for 1–2 tasks.
DEFAULT_SERIAL_MAX_TASKS = 2
_POLL_INTERVAL_S = 0.05


def process_limit_from_eco(eco: bool) -> int:
    """Return worker count from the main widget eco/full radio state."""
    fraction = 0.4 if eco else 0.8
    return max(1, int(multiprocessing.cpu_count() * fraction))


def should_use_pool(
    n_workers: int,
    n_tasks: int,
    *,
    serial_max_tasks: int = DEFAULT_SERIAL_MAX_TASKS,
) -> bool:
    """True when a process pool is cheaper than serial execution."""
    return n_workers > 1 and n_tasks > serial_max_tasks


def map_parallel(
    func: Callable[[Any], T],
    tasks: Iterable[Any],
    n_workers: int,
    *,
    serial_max_tasks: int = DEFAULT_SERIAL_MAX_TASKS,
) -> list[T]:
    """``Pool.map`` or serial ``list(map(...))`` depending on budget/size."""
    task_list = list(tasks)
    if not should_use_pool(
        n_workers, len(task_list), serial_max_tasks=serial_max_tasks
    ):
        return list(map(func, task_list))
    with Pool(n_workers) as pool:
        return pool.map(func, task_list)


def starmap_parallel(
    func: Callable[..., T],
    tasks: Iterable[Iterable[Any]],
    n_workers: int,
    *,
    serial_max_tasks: int = DEFAULT_SERIAL_MAX_TASKS,
) -> list[T]:
    """``Pool.starmap`` or serial comprehension depending on budget/size."""
    task_list = [tuple(args) for args in tasks]
    if not should_use_pool(
        n_workers, len(task_list), serial_max_tasks=serial_max_tasks
    ):
        return [func(*args) for args in task_list]
    with Pool(n_workers) as pool:
        return pool.starmap(func, task_list)


def iter_starmap_as_completed(
    func: Callable[..., T],
    tasks: Iterable[Iterable[Any]],
    n_workers: int,
    *,
    serial_max_tasks: int = DEFAULT_SERIAL_MAX_TASKS,
) -> Iterator[tuple[int, T]]:
    """Yield ``(task_index, result)`` as each task finishes.

    Unlike ``starmap_parallel``, this streams results for live progress. The
    pool is not opened with ``with Pool`` around yields (that pattern breaks
    napari generator workers). Cleanup uses ``terminate``/``join`` in
    ``finally`` so cancel still tears workers down.
    """
    task_list = [tuple(args) for args in tasks]
    if not task_list:
        return
    if not should_use_pool(
        n_workers, len(task_list), serial_max_tasks=serial_max_tasks
    ):
        for i, args in enumerate(task_list):
            yield i, func(*args)
        return

    pool = Pool(n_workers)
    try:
        pending = [
            (i, pool.apply_async(func, args)) for i, args in enumerate(task_list)
        ]
        pool.close()
        while pending:
            remaining = []
            progressed = False
            for i, async_result in pending:
                if async_result.ready():
                    yield i, async_result.get()
                    progressed = True
                else:
                    remaining.append((i, async_result))
            pending = remaining
            if pending and not progressed:
                time.sleep(_POLL_INTERVAL_S)
    finally:
        pool.terminate()
        pool.join()


def iter_map_as_completed(
    func: Callable[[Any], T],
    tasks: Iterable[Any],
    n_workers: int,
    *,
    serial_max_tasks: int = DEFAULT_SERIAL_MAX_TASKS,
) -> Iterator[tuple[int, T]]:
    """Yield ``(task_index, result)`` as each single-arg task finishes."""
    task_list = list(tasks)
    if not task_list:
        return
    if not should_use_pool(
        n_workers, len(task_list), serial_max_tasks=serial_max_tasks
    ):
        for i, task in enumerate(task_list):
            yield i, func(task)
        return

    pool = Pool(n_workers)
    try:
        pending = [
            (i, pool.apply_async(func, (task,))) for i, task in enumerate(task_list)
        ]
        pool.close()
        while pending:
            remaining = []
            progressed = False
            for i, async_result in pending:
                if async_result.ready():
                    yield i, async_result.get()
                    progressed = True
                else:
                    remaining.append((i, async_result))
            pending = remaining
            if pending and not progressed:
                time.sleep(_POLL_INTERVAL_S)
    finally:
        pool.terminate()
        pool.join()