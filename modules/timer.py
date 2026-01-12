# modules\timer.py

from __future__ import annotations

import time
from functools import wraps

class Timer:
    """
    Optional context manager for ad-hoc timing:

        with Timer() as t:
            ... do work ...
        print(t.elapsed)
    """
    def __init__(self, label: str | None = None):
        self.label = label
        self._t0: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is not None:
            self.elapsed = time.perf_counter() - self._t0
        return False


def timer(func):
    """
    Decorator that returns (result, elapsed_seconds), and cleanly forwards *args/**kwargs.

    Usage:
        @timer
        def foo(x, y, *, z=1):
            ...

        out, dt = foo(1, 2, z=3)
    """
    @wraps(func)
    def _wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        return result, dt

    return _wrapper

