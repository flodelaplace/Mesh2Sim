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

from ._arrays import Mat3, Vec3
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
from .vocab import SEGMENTS_BY_MODEL, validate_landmark

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
    model_id: str  # slug of the OpenSim model (e.g. Pose2Sim_Wholebody)
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


# ---------------------------------------------------------------------------
# Schéma 6: CorrespondenceMap (Mesh2Marker output — MHR vertices → OpenSim markers)
# ---------------------------------------------------------------------------


class FrameAlignment(BaseModel):
    """Rigid-plus-uniform-scale alignment between MHR mesh frame and OpenSim world frame."""

    model_config = _npy_config

    rotation: Mat3  # (3, 3) — applied first, must be a proper rotation in practice
    translation: Vec3  # (3,) — meters, applied after rotation
    scale: float = 1.0  # uniform scale; non-uniform handled via shape descriptors elsewhere


class CorrespondenceMarker(BaseModel):
    """One entry of a CorrespondenceMap: which MHR vertices map to which OpenSim marker."""

    model_config = _npy_config

    name: str  # validated against LANDMARKS
    mhr_vertices: list[int]  # MHR mesh vertex indices to regress / average for this marker
    opensim_body: str  # validated at CorrespondenceMap level against SEGMENTS[opensim_model]
    local_offset: Vec3  # (3,) meters, in the parent body's local frame
    fixed: bool  # True = bony landmark (rigid attachment), False = soft tissue
    synthpose_index: int | None = None  # cross-reference with SynthPose 2D detection layout

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        validate_landmark(v)
        return v

    @field_validator("mhr_vertices", mode="before")
    @classmethod
    def _check_vertices(cls, v: object) -> list[int]:
        if not isinstance(v, list) or not all(isinstance(i, int) and i >= 0 for i in v):
            raise ValueError("mhr_vertices must be a list of non-negative ints")
        if not v:
            raise ValueError("mhr_vertices must not be empty")
        return v


class CorrespondenceMap(BaseModel):
    """Frozen correspondence between MHR mesh topology and an OpenSim marker set.

    Produced offline by the Mesh2Marker tool, then loaded by the biomech bridge to drive
    per-subject scaling + IK in OpenSim. Validates that every entry references a known
    landmark name (LANDMARKS) and a known body (SEGMENTS[opensim_model]).
    """

    model_config = _npy_config

    schema_version: str = Field(default=SCHEMA_VERSION)
    mhr_topology_id: str  # e.g. "mhr_v1"; binds the map to a specific MHR mesh topology
    opensim_model: str  # slug — must be a key of SEGMENTS_BY_MODEL
    marker_set: str  # human-readable identifier for the marker set version
    frame_alignment: FrameAlignment
    markers: list[CorrespondenceMarker]

    @model_validator(mode="after")
    def _check_model_and_bodies(self) -> CorrespondenceMap:
        segs = SEGMENTS_BY_MODEL.get(self.opensim_model)
        if segs is None:
            raise ValueError(
                f"unknown opensim_model {self.opensim_model!r}; known: {sorted(SEGMENTS_BY_MODEL)}"
            )
        for m in self.markers:
            if m.opensim_body not in segs:
                raise ValueError(
                    f"marker {m.name!r} bound to unknown body {m.opensim_body!r} "
                    f"in model {self.opensim_model!r}"
                )
        # Unique marker names — duplicates would silently drop earlier entries.
        names = [m.name for m in self.markers]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate marker names in CorrespondenceMap: {dup}")
        return self


__all__ = [
    "AnatomicalObservation",
    "AnatomicalTrajectory",
    "BiomechFit",
    "BodyEstimate",
    "CorrespondenceMap",
    "CorrespondenceMarker",
    "FrameAlignment",
]
