"""Verbatim-casing lock for the OpenSim model snapshot.

These tests pin tricky names down to the exact spelling and parent_body in the bundled
reference. If anyone normalizes the vocabulary (lowercases, snake_cases, splits
``RWrist_hand`` etc.), these tests fail loudly. The point isn't pickiness — anatomy
exports must round-trip into OpenSim with byte-identical names or scaling/IK silently
falls over.
"""

from __future__ import annotations

from mesh2sim.contracts import (
    DEFAULT_MODEL_ID,
    LANDMARKS,
    SEGMENTS_BY_MODEL,
    is_valid_landmark,
    is_valid_segment,
    marker_parent_body,
)

# ---------------------------------------------------------------------------
# Default model identity
# ---------------------------------------------------------------------------


def test_default_model_id_is_pose2sim_wholebody():
    assert DEFAULT_MODEL_ID == "Pose2Sim_Wholebody"


def test_segments_count_is_30():
    assert len(SEGMENTS_BY_MODEL[DEFAULT_MODEL_ID]) == 30


def test_landmarks_count_is_73():
    assert len(LANDMARKS) == 73


# ---------------------------------------------------------------------------
# Casing traps the user called out explicitly
# ---------------------------------------------------------------------------


def test_abdomen_segment_is_capitalized():
    """The body is ``Abdomen`` (capital A), NOT ``abdomen``."""
    segs = SEGMENTS_BY_MODEL[DEFAULT_MODEL_ID]
    assert "Abdomen" in segs
    assert "abdomen" not in segs


def test_wrist_hand_markers_keep_mixed_case():
    """``RWrist_hand`` / ``LWrist_hand`` use mixed case with an underscore.
    Don't lowercase, don't camel-case, don't snake_case the whole thing."""
    assert is_valid_landmark("RWrist_hand")
    assert is_valid_landmark("LWrist_hand")
    for wrong in ("rwrist_hand", "RWristHand", "r_wrist_hand", "RWRIST_HAND"):
        assert not is_valid_landmark(wrong), f"variant {wrong!r} must NOT pass"


def test_rfa_markers_both_live_on_radius_r():
    """``RFAradius`` and ``RFAulna`` are both attached to ``radius_r`` (yes, both)."""
    assert is_valid_landmark("RFAradius")
    assert is_valid_landmark("RFAulna")
    assert marker_parent_body("RFAradius") == "radius_r"
    assert marker_parent_body("RFAulna") == "radius_r"


def test_lfa_markers_both_live_on_radius_l():
    """Mirror sanity check on the left side."""
    assert is_valid_landmark("LFAradius")
    assert is_valid_landmark("LFAulna")
    assert marker_parent_body("LFAradius") == "radius_l"
    assert marker_parent_body("LFAulna") == "radius_l"


# ---------------------------------------------------------------------------
# A representative sample to catch any accidental normalisation
# ---------------------------------------------------------------------------


def test_uppercase_anatomical_landmarks_pass():
    # The Pose2Sim markerset uses uppercase short codes (RASI, LASI, C7, ...).
    for name in ("RASI", "LASI", "RPSI", "LPSI", "C7", "RACR", "LACR", "RCAL", "LCAL"):
        assert is_valid_landmark(name), name


def test_lowercase_segments_pass():
    # Body/segment names are lowercase with side suffix (femur_r, tibia_l, ...).
    for name in ("pelvis", "femur_r", "femur_l", "tibia_r", "calcn_l", "torso", "head"):
        assert is_valid_segment(name), name


def test_segment_lookup_with_explicit_model():
    assert is_valid_segment("Abdomen", "Pose2Sim_Wholebody")
    assert not is_valid_segment("Abdomen", "Rajagopal2016")  # old default no longer registered
