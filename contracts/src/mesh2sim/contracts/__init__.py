"""Mesh2Sim data contracts.

Shared schemas exchanged between stages. Light dependencies only (pydantic + numpy).
See ``docs/contracts_spec.md`` at the repo root for the full specification.
"""

from .io import NPZ_KEY, load, save
from .schemas import (
    AnatomicalObservation,
    AnatomicalTrajectory,
    BiomechFit,
    BodyEstimate,
    CorrespondenceMap,
    CorrespondenceMarker,
    FrameAlignment,
)
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
    ShapeRepresentation,
    SkeletonState,
    Source,
    Task,
)
from .version import SCHEMA_VERSION, check_compatible, is_compatible
from .vocab import (
    COORDINATES_BY_MODEL,
    DEFAULT_MODEL_ID,
    LANDMARKS,
    MARKER_PARENT_BODY,
    MODELS_METADATA,
    SEGMENTS_BY_MODEL,
    is_valid_coordinate,
    is_valid_landmark,
    is_valid_segment,
    marker_parent_body,
    validate_coordinate,
    validate_landmark,
    validate_segment,
)

__version__ = SCHEMA_VERSION

__all__ = [
    # version
    "SCHEMA_VERSION",
    "check_compatible",
    "is_compatible",
    # vocab
    "DEFAULT_MODEL_ID",
    "LANDMARKS",
    "SEGMENTS_BY_MODEL",
    "COORDINATES_BY_MODEL",
    "MARKER_PARENT_BODY",
    "MODELS_METADATA",
    "is_valid_landmark",
    "is_valid_segment",
    "is_valid_coordinate",
    "validate_landmark",
    "validate_segment",
    "validate_coordinate",
    "marker_parent_body",
    # enums
    "Mode",
    "Pos3DFrame",
    "ShapeRepresentation",
    "Source",
    "Task",
    # transverse types
    "Capabilities",
    "CameraParams",
    "Keypoints2D",
    "Landmark",
    "MeshData",
    "Provenance",
    "ShapeDescriptor",
    "SkeletonState",
    # main schemas
    "AnatomicalObservation",
    "AnatomicalTrajectory",
    "BiomechFit",
    "BodyEstimate",
    "CorrespondenceMap",
    "CorrespondenceMarker",
    "FrameAlignment",
    # io
    "NPZ_KEY",
    "load",
    "save",
]
