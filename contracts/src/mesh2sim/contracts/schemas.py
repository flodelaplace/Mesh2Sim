"""The five contracts that travel between stages.

See ``docs/contracts_spec.md`` at repo root for the spec.

Conventions are uniform across all five:
- units: m / pixels / radians
- world frame: Y-up, right-handed (OpenSim-compatible)
- rotations: 3x3 matrices, never Euler in the contracts themselves
- every contract carries a ``schema_version`` field
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import (
    CameraParams,
    Capabilities,
    Keypoints2D,
    Landmark,
    MeshData,
    Mode,
    Pos3DFrame,
    Provenance,
    ShapeDescriptor,
    SkeletonState,
    Task,
)
from .version import SCHEMA_VERSION
from .vocab import validate_landmark

_npy_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Schéma 1: BodyEstimate (per frame, per view — estimator-native output)
# ---------------------------------------------------------------------------


class BodyEstimate(BaseModel):
    """Estimator-native body estimate (one frame, one view), before standardisation.

    Capabilities flags that DO apply: has_mesh, has_skeleton, has_2d_keypoints,
    has_native_params. The other flags (has_segment_frames, has_shape_descriptor) are not
    enforced here.
    """

    model_config = _npy_config

    schema_version: str = Field(default=SCHEMA_VERSION)
    estimator_id: str
    frame_id: int
    view_id: str
    timestamp: float | None = None
    capabilities: Capabilities
    native_params: dict | None = None
    mesh: MeshData | None = None
    skeleton_state: SkeletonState | None = None
    keypoints_2d: Keypoints2D | None = None
    camera: CameraParams | None = None
    frame_shape: tuple[int, int] | None = None  # (H, W)

    @model_validator(mode="after")
    def _check_capabilities(self) -> BodyEstimate:
        pairs = {
            "has_mesh": self.mesh,
            "has_skeleton": self.skeleton_state,
            "has_2d_keypoints": self.keypoints_2d,
            "has_native_params": self.native_params,
        }
        for flag, field in pairs.items():
            present = field is not None
            if getattr(self.capabilities, flag) != present:
                raise ValueError(
                    f"capability {flag}={getattr(self.capabilities, flag)} "
                    f"but field {'present' if present else 'absent'}"
                )
        return self


# ---------------------------------------------------------------------------
# Schéma 3: AnatomicalObservation (THE key contract — per frame, per view)
# ---------------------------------------------------------------------------


class AnatomicalObservation(BaseModel):
    """Standardised anatomical observation, one frame, one view.

    Capabilities flags that DO apply here: has_mesh (refers to ``dense_surface``),
    has_segment_frames, has_shape_descriptor. The other flags are not enforced here.
    """

    model_config = _npy_config

    schema_version: str = Field(default=SCHEMA_VERSION)
    frame_id: int
    view_id: str
    timestamp: float | None = None
    landmarks: dict[str, Landmark]
    pos3d_frame: Pos3DFrame = Pos3DFrame.none
    segment_frames: dict[str, np.ndarray] | None = None  # name -> (3,3)
    shape_descriptor: ShapeDescriptor | None = None
    joint_centers_init: dict[str, np.ndarray] | None = None  # name -> (3,)
    dense_surface: MeshData | None = None
    capabilities: Capabilities
    provenance: Provenance

    @field_validator("landmarks")
    @classmethod
    def _check_landmark_names(cls, v: dict[str, Landmark]) -> dict[str, Landmark]:
        for name in v:
            validate_landmark(name)
        return v

    @field_validator("segment_frames", mode="before")
    @classmethod
    def _coerce_segment_frames(cls, v: dict[str, object] | None) -> dict[str, np.ndarray] | None:
        if v is None:
            return None
        out = {}
        for name, mat in v.items():
            arr = mat if isinstance(mat, np.ndarray) else np.asarray(mat)
            if arr.shape != (3, 3):
                raise ValueError(f"segment_frames[{name!r}] must be (3, 3), got {arr.shape}")
            out[name] = arr
        return out

    @field_validator("joint_centers_init", mode="before")
    @classmethod
    def _coerce_joint_centers(cls, v: dict[str, object] | None) -> dict[str, np.ndarray] | None:
        if v is None:
            return None
        out = {}
        for name, arr_in in v.items():
            arr = arr_in if isinstance(arr_in, np.ndarray) else np.asarray(arr_in)
            if arr.shape != (3,):
                raise ValueError(f"joint_centers_init[{name!r}] must be (3,), got {arr.shape}")
            out[name] = arr
        return out

    @model_validator(mode="after")
    def _check_capabilities(self) -> AnatomicalObservation:
        pairs = {
            "has_segment_frames": self.segment_frames,
            "has_shape_descriptor": self.shape_descriptor,
            "has_mesh": self.dense_surface,
        }
        for flag, field in pairs.items():
            present = field is not None
            if getattr(self.capabilities, flag) != present:
                raise ValueError(
                    f"capability {flag}={getattr(self.capabilities, flag)} "
                    f"but field {'present' if present else 'absent'}"
                )
        return self


# ---------------------------------------------------------------------------
# Schéma 4: AnatomicalTrajectory (per trial, fused or assembled)
# ---------------------------------------------------------------------------


class AnatomicalTrajectory(BaseModel):
    """Per-trial trajectory in the world frame. Shape descriptor is unique (shape-lock)."""

    model_config = _npy_config

    schema_version: str = Field(default=SCHEMA_VERSION)
    subject_id: str
    trial_id: str
    task: Task
    mode: Mode
    fps: float
    landmark_names: list[str]
    positions: np.ndarray  # (T, L, 3) world frame, m
    confidence: np.ndarray  # (T, L)
    shape_descriptor: ShapeDescriptor
    views_used: list[str]
    uncertainty: np.ndarray | None = None  # (T, L, 3)
    provenance: Provenance

    @field_validator("landmark_names")
    @classmethod
    def _check_names(cls, v: list[str]) -> list[str]:
        for name in v:
            validate_landmark(name)
        return v

    @field_validator("positions", "confidence", mode="before")
    @classmethod
    def _coerce_arr(cls, v: object) -> np.ndarray:
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @field_validator("uncertainty", mode="before")
    @classmethod
    def _coerce_uncertainty(cls, v: object) -> np.ndarray | None:
        if v is None:
            return None
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @model_validator(mode="after")
    def _check_shapes(self) -> AnatomicalTrajectory:
        ell = len(self.landmark_names)
        if self.positions.ndim != 3 or self.positions.shape[1:] != (ell, 3):
            raise ValueError(f"positions must be (T, {ell}, 3), got {self.positions.shape}")
        t = self.positions.shape[0]
        if self.confidence.shape != (t, ell):
            raise ValueError(f"confidence must be ({t}, {ell}), got {self.confidence.shape}")
        if self.uncertainty is not None and self.uncertainty.shape != (t, ell, 3):
            raise ValueError(f"uncertainty must be ({t}, {ell}, 3), got {self.uncertainty.shape}")
        return self


# ---------------------------------------------------------------------------
# Schéma 5: BiomechFit (per subject/trial — bridge output)
# ---------------------------------------------------------------------------


class BiomechFit(BaseModel):
    """Joint angle output from the biomech bridge.

    At least one of ``angles`` or ``motion_path`` must be present.
    """

    model_config = _npy_config

    schema_version: str = Field(default=SCHEMA_VERSION)
    subject_id: str
    trial_id: str
    model_id: str  # e.g. Rajagopal2016, LaiUhlrich, sportfx
    scaled_model_path: str
    dof_names: list[str]
    angles: np.ndarray | None = None  # (T, D) radians
    motion_path: str | None = None  # path to a .mot file
    marker_offsets: dict[str, np.ndarray]  # landmark name -> (3,) offset, m
    marker_residuals: np.ndarray  # (T,)
    uncertainty: np.ndarray | None = None  # (T, D, 2) — CI per DoF
    provenance: Provenance

    @field_validator("marker_offsets", mode="before")
    @classmethod
    def _coerce_marker_offsets(cls, v: dict[str, object]) -> dict[str, np.ndarray]:
        if not isinstance(v, dict):
            raise ValueError("marker_offsets must be a dict")
        out = {}
        for name, arr_in in v.items():
            validate_landmark(name)
            arr = arr_in if isinstance(arr_in, np.ndarray) else np.asarray(arr_in)
            if arr.shape != (3,):
                raise ValueError(f"marker_offsets[{name!r}] must be (3,), got {arr.shape}")
            out[name] = arr
        return out

    @field_validator("angles", "uncertainty", mode="before")
    @classmethod
    def _coerce_optional_array(cls, v: object) -> np.ndarray | None:
        if v is None:
            return None
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @field_validator("marker_residuals", mode="before")
    @classmethod
    def _coerce_residuals(cls, v: object) -> np.ndarray:
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @model_validator(mode="after")
    def _check_consistency(self) -> BiomechFit:
        if self.angles is None and self.motion_path is None:
            raise ValueError("BiomechFit needs at least one of angles or motion_path")
        if self.marker_residuals.ndim != 1:
            raise ValueError(f"marker_residuals must be (T,), got {self.marker_residuals.shape}")
        t = self.marker_residuals.shape[0]
        d = len(self.dof_names)
        if self.angles is not None:
            if self.angles.shape != (t, d):
                raise ValueError(f"angles must be ({t}, {d}), got {self.angles.shape}")
            if self.uncertainty is not None and self.uncertainty.shape != (t, d, 2):
                raise ValueError(f"uncertainty must be ({t}, {d}, 2), got {self.uncertainty.shape}")
        return self


__all__ = [
    "AnatomicalObservation",
    "AnatomicalTrajectory",
    "BiomechFit",
    "BodyEstimate",
]
