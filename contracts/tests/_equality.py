"""Deep equality with strict array semantics (exact values + exact dtype).

Plain ``==`` between BaseModel instances breaks when ndarray fields are involved (ndarray
equality returns an array, not a bool). This helper walks the dumped dicts and asserts:
- ndarray vs ndarray: identical shape, identical dtype, ``np.array_equal``
- dict vs dict: same keys, recursive equality on values
- list/tuple vs same: equal length, recursive on items
- everything else: ``==``
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


def assert_models_equal(a: BaseModel, b: BaseModel) -> None:
    if type(a) is not type(b):
        raise AssertionError(f"different types: {type(a).__name__} vs {type(b).__name__}")
    _assert_equal(a.model_dump(mode="python"), b.model_dump(mode="python"))


def _assert_equal(a: object, b: object, path: str = "$") -> None:
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            raise AssertionError(f"{path}: expected ndarray, got {type(b).__name__}")
        if a.dtype != b.dtype:
            raise AssertionError(f"{path}: dtype mismatch {a.dtype} vs {b.dtype}")
        if a.shape != b.shape:
            raise AssertionError(f"{path}: shape mismatch {a.shape} vs {b.shape}")
        if not np.array_equal(a, b):
            raise AssertionError(f"{path}: values differ")
        return

    if isinstance(a, dict):
        if not isinstance(b, dict):
            raise AssertionError(f"{path}: expected dict, got {type(b).__name__}")
        if set(a) != set(b):
            raise AssertionError(f"{path}: keys differ — left={sorted(a)}, right={sorted(b)}")
        for k in a:
            _assert_equal(a[k], b[k], f"{path}.{k}")
        return

    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            raise AssertionError(f"{path}: expected list/tuple, got {type(b).__name__}")
        if len(a) != len(b):
            raise AssertionError(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{path}[{i}]")
        return

    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")
