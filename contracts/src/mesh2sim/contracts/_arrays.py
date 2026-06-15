"""Internal array helpers for pydantic v2 validators.

Two design rules:

1. **Never silently change dtype.** A validator that needs a specific dtype (e.g. mesh
   vertices in float32) casts explicitly; every other validator preserves whatever dtype
   the caller passed in. Round-trip must keep dtype intact.
2. **Coerce lists/tuples to ndarray on input**, so users can write
   ``Landmark(pos_3d=[0.1, 0.2, 0.3])`` instead of forcing them through ``np.asarray``.
"""

from __future__ import annotations

from typing import Annotated, Callable

import numpy as np
from pydantic import BeforeValidator


def _shape_check(
    expected: tuple[int, ...],
    *,
    dtype: np.dtype | None = None,
    allow_none: bool = False,
) -> Callable[[object], np.ndarray | None]:
    """Build a validator that coerces input to ndarray and checks its shape.

    If ``dtype`` is None, the input dtype is preserved (no silent casting).
    """

    def _check(v: object) -> np.ndarray | None:
        if v is None:
            if allow_none:
                return None
            raise ValueError("required array is None")
        if isinstance(v, np.ndarray):
            arr = v if dtype is None else v.astype(dtype, copy=False)
        else:
            arr = np.asarray(v) if dtype is None else np.asarray(v, dtype=dtype)
        if arr.shape != expected:
            raise ValueError(f"expected shape {expected}, got {arr.shape}")
        return arr

    return _check


def _ndim_check(
    ndim: int,
    *,
    trailing: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    allow_none: bool = False,
) -> Callable[[object], np.ndarray | None]:
    """Build a validator that coerces and checks ``ndim`` and optionally trailing dims.

    Use when leading dims are variable (e.g. T frames in a trajectory).
    """

    def _check(v: object) -> np.ndarray | None:
        if v is None:
            if allow_none:
                return None
            raise ValueError("required array is None")
        if isinstance(v, np.ndarray):
            arr = v if dtype is None else v.astype(dtype, copy=False)
        else:
            arr = np.asarray(v) if dtype is None else np.asarray(v, dtype=dtype)
        if arr.ndim != ndim:
            raise ValueError(f"expected ndim {ndim}, got {arr.ndim} (shape {arr.shape})")
        if trailing is not None and arr.shape[-len(trailing) :] != trailing:
            raise ValueError(
                f"expected trailing dims {trailing}, got {arr.shape[-len(trailing) :]}"
            )
        return arr

    return _check


# ---------------------------------------------------------------------------
# Public reusable annotated types (preserve caller dtype unless noted)
# ---------------------------------------------------------------------------

Vec2 = Annotated[np.ndarray, BeforeValidator(_shape_check((2,)))]
Vec3 = Annotated[np.ndarray, BeforeValidator(_shape_check((3,)))]
Mat3 = Annotated[np.ndarray, BeforeValidator(_shape_check((3, 3)))]

Vec2Opt = Annotated[np.ndarray | None, BeforeValidator(_shape_check((2,), allow_none=True))]
Vec3Opt = Annotated[np.ndarray | None, BeforeValidator(_shape_check((3,), allow_none=True))]
Mat3Opt = Annotated[np.ndarray | None, BeforeValidator(_shape_check((3, 3), allow_none=True))]

# Mesh-specific: spec requires float32 vertices (memory budget for large meshes).
VerticesF32 = Annotated[
    np.ndarray,
    BeforeValidator(_ndim_check(2, trailing=(3,), dtype=np.dtype(np.float32))),
]
# Faces: int dtype preserved (no spec on exact int width); shape (F, 3).
FacesOpt = Annotated[
    np.ndarray | None,
    BeforeValidator(_ndim_check(2, trailing=(3,), allow_none=True)),
]

# Distortion vector (variable length).
Vec1DOpt = Annotated[
    np.ndarray | None,
    BeforeValidator(_ndim_check(1, allow_none=True)),
]
