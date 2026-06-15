"""Frozen anatomical vocabularies.

LANDMARKS is the shared vocabulary used everywhere a landmark name appears in a contract
(SynthPose outputs, MHR-to-anatomy correspondence map, OpenSim marker registration). It is
inspired by the OpenCap 43-marker set (lower body focus, with upper-body and trunk
references). Names use lowercase snake_case with side prefix ``r_`` / ``l_``.

SEGMENTS_BY_MODEL lists the OpenSim segment names for each supported musculoskeletal model.
The biomech bridge uses these as the targets of marker registration (scaling + offsets).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Landmarks (anatomical markers)
# ---------------------------------------------------------------------------

# Pelvis
_PELVIS = frozenset(
    {
        "r_asis",
        "l_asis",
        "r_psis",
        "l_psis",
    }
)

# Thigh (cluster markers)
_THIGH = frozenset(
    {
        "r_thigh_1",
        "r_thigh_2",
        "r_thigh_3",
        "r_thigh_4",
        "l_thigh_1",
        "l_thigh_2",
        "l_thigh_3",
        "l_thigh_4",
    }
)

# Knee
_KNEE = frozenset(
    {
        "r_knee_med",
        "r_knee_lat",
        "l_knee_med",
        "l_knee_lat",
    }
)

# Shank (cluster markers)
_SHANK = frozenset(
    {
        "r_shank_1",
        "r_shank_2",
        "r_shank_3",
        "r_shank_4",
        "l_shank_1",
        "l_shank_2",
        "l_shank_3",
        "l_shank_4",
    }
)

# Ankle
_ANKLE = frozenset(
    {
        "r_ankle_med",
        "r_ankle_lat",
        "l_ankle_med",
        "l_ankle_lat",
    }
)

# Foot
_FOOT = frozenset(
    {
        "r_calc",
        "r_toe_1",
        "r_toe_5",
        "l_calc",
        "l_toe_1",
        "l_toe_5",
    }
)

# Trunk
_TRUNK = frozenset(
    {
        "c7",
        "sternum_notch",
        "sternum_xiphoid",
    }
)

# Upper limb (kept light — full upper-body capture is not the primary use case but the names
# exist for completeness with OpenCap-style sets)
_UPPER = frozenset(
    {
        "r_acromion",
        "l_acromion",
        "r_humerus_lat_epicondyle",
        "l_humerus_lat_epicondyle",
        "r_radius_styloid",
        "l_radius_styloid",
        "r_ulna_styloid",
        "l_ulna_styloid",
    }
)

LANDMARKS: frozenset[str] = _PELVIS | _THIGH | _KNEE | _SHANK | _ANKLE | _FOOT | _TRUNK | _UPPER
"""Closed vocabulary of anatomical landmark names. Inspired by OpenCap (43 markers)."""


def is_valid_landmark(name: str) -> bool:
    """Return True iff ``name`` is in the closed LANDMARKS vocabulary."""
    return name in LANDMARKS


def validate_landmark(name: str) -> None:
    """Raise ValueError if ``name`` is not in LANDMARKS."""
    if not is_valid_landmark(name):
        raise ValueError(
            f"unknown landmark {name!r}; "
            f"see mesh2sim.contracts.vocab.LANDMARKS ({len(LANDMARKS)} entries)"
        )


# ---------------------------------------------------------------------------
# OpenSim segments per model
# ---------------------------------------------------------------------------

# Rajagopal 2016 full-body model: lower limbs + torso + arms.
_RAJAGOPAL2016_SEGMENTS = frozenset(
    {
        "pelvis",
        "femur_r",
        "femur_l",
        "tibia_r",
        "tibia_l",
        "talus_r",
        "talus_l",
        "calcn_r",
        "calcn_l",
        "toes_r",
        "toes_l",
        "torso",
        "humerus_r",
        "humerus_l",
        "ulna_r",
        "ulna_l",
        "radius_r",
        "radius_l",
        "hand_r",
        "hand_l",
    }
)

SEGMENTS_BY_MODEL: dict[str, frozenset[str]] = {
    "Rajagopal2016": _RAJAGOPAL2016_SEGMENTS,
}
"""Segment names per supported OpenSim model. Add a new key per model as they come online."""


def is_valid_segment(name: str, model_id: str) -> bool:
    """Return True iff ``name`` is a known segment for ``model_id``."""
    segs = SEGMENTS_BY_MODEL.get(model_id)
    if segs is None:
        return False
    return name in segs


def validate_segment(name: str, model_id: str) -> None:
    """Raise ValueError if ``name`` is not a valid segment for ``model_id``."""
    segs = SEGMENTS_BY_MODEL.get(model_id)
    if segs is None:
        raise ValueError(
            f"unknown OpenSim model_id {model_id!r}; known: {sorted(SEGMENTS_BY_MODEL)}"
        )
    if name not in segs:
        raise ValueError(f"unknown segment {name!r} for model {model_id!r}; known: {sorted(segs)}")
