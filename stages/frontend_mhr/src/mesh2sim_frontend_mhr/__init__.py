"""Mesh2Sim frontend stage — wraps the vendored sam_3d_body inference and emits
``BodyEstimate`` contracts.

Public API:
- ``MeshEstimator``                 : Protocol (per-frame estimator interface).
- ``FastSAM3DBodyEstimator``        : implementation backed by the vendored core.
- ``sam3db_output_to_body_estimate``: pure adapter, useful for tests and tools.
- MHR topology constants            : see ``mhr_topology``.
- ``DEFAULT_ESTIMATOR_ID``          : pinned ``estimator_id`` string.
"""

from __future__ import annotations

from .adapter import sam3db_output_to_body_estimate
from .estimator import DEFAULT_ESTIMATOR_ID, FastSAM3DBodyEstimator
from .interface import MeshEstimator
from .mhr_topology import (
    MHR_JOINT_NAMES,
    MHR_N_JOINTS,
    MHR_N_KEYPOINTS_2D,
    MHR_N_MODEL_PARAMETERS,
    MHR_N_VERTICES,
    MHR_RIG_FINGERPRINT,
    MHR_TOPOLOGY_ID,
)
from .shape_lock import (
    SHAPE_LOCK_VERSION,
    ShapeLockRegenerator,
    aggregate_shape,
    lock_shape_and_regenerate,
)

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_ESTIMATOR_ID",
    "FastSAM3DBodyEstimator",
    "MHR_JOINT_NAMES",
    "MHR_N_JOINTS",
    "MHR_N_KEYPOINTS_2D",
    "MHR_N_MODEL_PARAMETERS",
    "MHR_N_VERTICES",
    "MHR_RIG_FINGERPRINT",
    "MHR_TOPOLOGY_ID",
    "MeshEstimator",
    "SHAPE_LOCK_VERSION",
    "ShapeLockRegenerator",
    "aggregate_shape",
    "lock_shape_and_regenerate",
    "sam3db_output_to_body_estimate",
]
