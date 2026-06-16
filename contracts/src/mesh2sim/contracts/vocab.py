"""Frozen anatomical vocabularies, loaded verbatim from the bundled OpenSim model snapshot.

**No name is typed in this file.** Everything is derived from
``contracts/src/mesh2sim/contracts/data/opensim_model.json``, which is a verbatim
extraction from the production OpenSim model (``Pose2Sim_WithMusclesAndConstraints``).
Case is preserved exactly: ``Abdomen`` capitalized as a body, ``RWrist_hand`` / ``LWrist_hand``
mixed case markers, ``RFAradius`` and ``RFAulna`` both attached to ``radius_r``, etc.

Public surface:
- ``LANDMARKS``                — frozen set of marker names (the 73 of the model)
- ``SEGMENTS_BY_MODEL[model]`` — frozen set of body names per model
- ``COORDINATES_BY_MODEL[m]``  — frozen set of coordinate names per model
- ``MARKER_PARENT_BODY[m][n]`` — for each marker name, its parent body name (verbatim)
- ``MODELS_METADATA[model]``   — model identity, units, axes
- ``DEFAULT_MODEL_ID``         — slug of the production model

When the OpenSim model changes: re-extract the JSON to that path and rebuild — vocab.py
does not need editing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from types import MappingProxyType

_REF_FILENAME = "opensim_model.json"


def _load_reference() -> dict:
    """Read the bundled OpenSim-model snapshot. Errors loudly if absent or malformed."""
    pkg = files(__package__) / "data" / _REF_FILENAME
    if not pkg.is_file():
        raise FileNotFoundError(
            f"missing reference {pkg}; the contracts package must ship the OpenSim model "
            f"snapshot under data/{_REF_FILENAME}"
        )
    return json.loads(pkg.read_text(encoding="utf-8"))


_REF = _load_reference()
_IDENTITY = _REF["model_identity"]
_UNITS = _REF["units_and_axes"]

DEFAULT_MODEL_ID: str = _IDENTITY["slug"]
"""Slug identifying the production OpenSim model used as default everywhere."""


# ---------------------------------------------------------------------------
# Bodies / segments
# ---------------------------------------------------------------------------

SEGMENTS_BY_MODEL: Mapping[str, frozenset[str]] = MappingProxyType(
    {DEFAULT_MODEL_ID: frozenset(_REF["bodies"])}
)
"""Segment (body) names per supported OpenSim model. Keys are model slugs."""


def is_valid_segment(name: str, model_id: str = DEFAULT_MODEL_ID) -> bool:
    """Return True iff ``name`` is a known segment for ``model_id``."""
    segs = SEGMENTS_BY_MODEL.get(model_id)
    return bool(segs) and name in segs


def validate_segment(name: str, model_id: str = DEFAULT_MODEL_ID) -> None:
    """Raise ValueError if ``name`` is not a valid segment for ``model_id``."""
    segs = SEGMENTS_BY_MODEL.get(model_id)
    if segs is None:
        raise ValueError(
            f"unknown OpenSim model_id {model_id!r}; known: {sorted(SEGMENTS_BY_MODEL)}"
        )
    if name not in segs:
        raise ValueError(f"unknown segment {name!r} for model {model_id!r}; known: {sorted(segs)}")


# ---------------------------------------------------------------------------
# Coordinates (DoF names)
# ---------------------------------------------------------------------------

COORDINATES_BY_MODEL: Mapping[str, frozenset[str]] = MappingProxyType(
    {DEFAULT_MODEL_ID: frozenset(c["name"] for c in _REF["coordinates"])}
)
"""Coordinate (DoF) names per supported model. Keys are model slugs."""


def is_valid_coordinate(name: str, model_id: str = DEFAULT_MODEL_ID) -> bool:
    coords = COORDINATES_BY_MODEL.get(model_id)
    return bool(coords) and name in coords


def validate_coordinate(name: str, model_id: str = DEFAULT_MODEL_ID) -> None:
    coords = COORDINATES_BY_MODEL.get(model_id)
    if coords is None:
        raise ValueError(
            f"unknown OpenSim model_id {model_id!r}; known: {sorted(COORDINATES_BY_MODEL)}"
        )
    if name not in coords:
        raise ValueError(f"unknown coordinate {name!r} for model {model_id!r}")


# ---------------------------------------------------------------------------
# Landmarks (markers)
# ---------------------------------------------------------------------------

_MARKERSET = _REF["markerset"]
_MARKERS = _MARKERSET["markers"]

LANDMARKS: frozenset[str] = frozenset(m["name"] for m in _MARKERS)
"""Closed vocabulary of marker names. Verbatim from the bundled OpenSim markerset."""


MARKER_PARENT_BODY: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {DEFAULT_MODEL_ID: MappingProxyType({m["name"]: m["parent_body"] for m in _MARKERS})}
)
"""For each model, a name → parent_body mapping (e.g. ``RFAradius`` → ``radius_r``)."""


def is_valid_landmark(name: str) -> bool:
    """Return True iff ``name`` is in the closed LANDMARKS vocabulary."""
    return name in LANDMARKS


def validate_landmark(name: str) -> None:
    """Raise ValueError if ``name`` is not in LANDMARKS."""
    if name not in LANDMARKS:
        raise ValueError(
            f"unknown landmark {name!r}; "
            f"see mesh2sim.contracts.vocab.LANDMARKS ({len(LANDMARKS)} entries)"
        )


def marker_parent_body(name: str, model_id: str = DEFAULT_MODEL_ID) -> str:
    """Look up the parent_body of a marker for a given model. Raises on unknown name."""
    table = MARKER_PARENT_BODY.get(model_id)
    if table is None:
        raise ValueError(
            f"unknown OpenSim model_id {model_id!r}; known: {sorted(MARKER_PARENT_BODY)}"
        )
    if name not in table:
        raise ValueError(f"unknown marker {name!r} for model {model_id!r}")
    return table[name]


# ---------------------------------------------------------------------------
# Model metadata (identity, units, axes)
# ---------------------------------------------------------------------------

MODELS_METADATA: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        DEFAULT_MODEL_ID: MappingProxyType(
            {
                "slug": _IDENTITY["slug"],
                "model_name_verbatim": _IDENTITY["model_name_verbatim"],
                "opensim_document_version": _IDENTITY["opensim_document_version"],
                "derived_family": _IDENTITY["derived_family"],
                "credits_verbatim": _IDENTITY["credits_verbatim"],
                "publications_verbatim": _IDENTITY["publications_verbatim"],
                "length_units": _UNITS["length_units"],
                "force_units": _UNITS["force_units"],
                "gravity": _UNITS["gravity"],
                "axis_convention": _UNITS["axis_convention"],
            }
        )
    }
)
"""Identity + units + axes per supported model, verbatim from the JSON snapshot."""


__all__ = [
    "COORDINATES_BY_MODEL",
    "DEFAULT_MODEL_ID",
    "LANDMARKS",
    "MARKER_PARENT_BODY",
    "MODELS_METADATA",
    "SEGMENTS_BY_MODEL",
    "is_valid_coordinate",
    "is_valid_landmark",
    "is_valid_segment",
    "marker_parent_body",
    "validate_coordinate",
    "validate_landmark",
    "validate_segment",
]
