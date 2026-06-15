"""Enums and transverse value objects.

Kept separate from ``schemas.py`` so the five main schemas can read at a glance.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._arrays import (
    FacesOpt,
    Mat3,
    Mat3Opt,
    Vec1DOpt,
    Vec2Opt,
    Vec3Opt,
    VerticesF32,
)
from .version import SCHEMA_VERSION

_npy_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Source(str, Enum):
    """How a landmark was inferred."""

    bony = "bony"
    soft = "soft"
    unknown = "unknown"


class Pos3DFrame(str, Enum):
    """Reference frame of a landmark's pos_3d."""

    none = "none"
    camera = "camera"
    world = "world"


class Task(str, Enum):
    """Motor task being captured."""

    gait = "gait"
    sts = "sts"  # sit-to-stand
    other = "other"


class Mode(str, Enum):
    """Capture mode."""

    mono = "mono"
    multi = "multi"


class ShapeRepresentation(str, Enum):
    """How shape information is encoded in a ShapeDescriptor."""

    per_segment_scale = "per_segment_scale"
    opaque = "opaque"


# ---------------------------------------------------------------------------
# Transverse value objects
# ---------------------------------------------------------------------------


class Capabilities(BaseModel):
    """Capability flags. Must stay in sync with the corresponding optional fields.

    Coherence (``flag == (field is not None)``) is enforced per-schema in
    ``schemas.py`` via ``model_validator``; not all flags apply to every schema.
    """

    has_mesh: bool = False
    has_skeleton: bool = False
    has_2d_keypoints: bool = False
    has_native_params: bool = False
    has_segment_frames: bool = False
    has_shape_descriptor: bool = False


class Provenance(BaseModel):
    """Trace of who produced a contract and with what map/adapter."""

    estimator_id: str | None = None
    adapter_id: str | None = None
    correspondence_map_id: str | None = None
    created_at: str | None = None  # ISO-8601 — left to producer to set
    schema_version: str = Field(default=SCHEMA_VERSION)
    extra: dict = Field(default_factory=dict)


class MeshData(BaseModel):
    """A triangle mesh with explicit topology id."""

    model_config = _npy_config

    vertices: VerticesF32  # (V, 3) float32 — cast enforced by spec
    topology_id: str
    faces: FacesOpt = None  # (F, 3) int dtype preserved


class Landmark(BaseModel):
    """One anatomical landmark observation."""

    model_config = _npy_config

    pos_3d: Vec3Opt = None  # (3,) dtype preserved
    pos_2d: Vec2Opt = None  # (2,) dtype preserved
    confidence: float = 0.0
    visibility: float = 0.0
    source: Source = Source.unknown


class SkeletonState(BaseModel):
    """Joint chain: positions (J,3) and orientations (J,3,3) under shared name list."""

    model_config = _npy_config

    joint_positions: np.ndarray  # (J, 3)
    joint_orientations: np.ndarray  # (J, 3, 3)
    joint_names: list[str]

    @field_validator("joint_positions", mode="before")
    @classmethod
    def _coerce_positions(cls, v: object) -> np.ndarray:
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @field_validator("joint_orientations", mode="before")
    @classmethod
    def _coerce_orientations(cls, v: object) -> np.ndarray:
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @model_validator(mode="after")
    def _check_shapes(self) -> SkeletonState:
        j = len(self.joint_names)
        if self.joint_positions.shape != (j, 3):
            raise ValueError(f"joint_positions must be ({j}, 3), got {self.joint_positions.shape}")
        if self.joint_orientations.shape != (j, 3, 3):
            raise ValueError(
                f"joint_orientations must be ({j}, 3, 3), got {self.joint_orientations.shape}"
            )
        return self


class Keypoints2D(BaseModel):
    """2D keypoints with confidences, indexed by name list."""

    model_config = _npy_config

    names: list[str]
    xy: np.ndarray  # (K, 2)
    confidence: np.ndarray  # (K,)

    @field_validator("xy", "confidence", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> np.ndarray:
        return v if isinstance(v, np.ndarray) else np.asarray(v)

    @model_validator(mode="after")
    def _check_shapes(self) -> Keypoints2D:
        k = len(self.names)
        if self.xy.shape != (k, 2):
            raise ValueError(f"xy must be ({k}, 2), got {self.xy.shape}")
        if self.confidence.shape != (k,):
            raise ValueError(f"confidence must be ({k},), got {self.confidence.shape}")
        return self


class ShapeDescriptor(BaseModel):
    """Per-segment scale or opaque shape vector.

    For ``per_segment_scale``, each segment name maps to a (3,) scale vector. The keys are
    OpenSim segment names of ``source_model``.
    """

    model_config = _npy_config

    representation: ShapeRepresentation
    data: dict[str, np.ndarray]
    source_model: str

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_values(cls, v: object) -> dict:
        if not isinstance(v, dict):
            raise ValueError("data must be a dict[str, ndarray]")
        out = {}
        for k, val in v.items():
            arr = val if isinstance(val, np.ndarray) else np.asarray(val)
            out[k] = arr
        return out

    @model_validator(mode="after")
    def _check_per_segment(self) -> ShapeDescriptor:
        if self.representation == ShapeRepresentation.per_segment_scale:
            for seg, arr in self.data.items():
                if arr.shape != (3,):
                    raise ValueError(f"per-segment scale for {seg!r} must be (3,), got {arr.shape}")
        return self


class CameraParams(BaseModel):
    """Intrinsics, optional extrinsics, resolution, and sync offset for one view."""

    model_config = _npy_config

    view_id: str
    K: Mat3
    distortion: Vec1DOpt = None
    R: Mat3Opt = None
    t: Vec3Opt = None
    resolution: tuple[int, int]
    time_offset: float | None = None


__all__ = [
    "Capabilities",
    "CameraParams",
    "Keypoints2D",
    "Landmark",
    "MeshData",
    "Mode",
    "Pos3DFrame",
    "Provenance",
    "ShapeDescriptor",
    "ShapeRepresentation",
    "SkeletonState",
    "Source",
    "Task",
]
