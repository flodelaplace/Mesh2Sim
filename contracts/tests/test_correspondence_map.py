"""Tests for the CorrespondenceMap schema (6th contract).

A CorrespondenceMap is the artefact produced by the Mesh2Marker tool — for each OpenSim
marker name, which MHR mesh vertices contribute, on which body, at what local offset,
bony vs soft, and (optionally) which SynthPose 2D index it pairs with. Validation has to
catch the two failure modes that would corrupt the downstream OpenSim scaling silently:

1. Marker name not in the closed LANDMARKS vocabulary
2. Body name not in SEGMENTS for the specified opensim_model
"""

from __future__ import annotations

import numpy as np
import pytest
from mesh2sim.contracts import (
    CorrespondenceMap,
    CorrespondenceMarker,
    FrameAlignment,
    load,
    save,
)
from pydantic import ValidationError

from ._equality import assert_models_equal


def _make_fake_map() -> CorrespondenceMap:
    """Small but valid CorrespondenceMap covering several body types and a soft marker."""
    return CorrespondenceMap(
        mhr_topology_id="mhr_v1",
        opensim_model="Pose2Sim_Wholebody",
        marker_set="pose2sim_v0",
        frame_alignment=FrameAlignment(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            scale=1.0,
        ),
        markers=[
            CorrespondenceMarker(
                name="RASI",
                mhr_vertices=[120, 121, 122],
                opensim_body="pelvis",
                local_offset=np.array([0.005, 0.0, 0.0], dtype=np.float64),
                fixed=True,
                synthpose_index=11,
            ),
            CorrespondenceMarker(
                name="LASI",
                mhr_vertices=[150, 151],
                opensim_body="pelvis",
                local_offset=np.array([-0.005, 0.0, 0.0], dtype=np.float64),
                fixed=True,
                synthpose_index=12,
            ),
            CorrespondenceMarker(
                name="RFAradius",
                mhr_vertices=[8800],
                opensim_body="radius_r",
                local_offset=np.array([0.0005, -0.229515, 0.05], dtype=np.float64),
                fixed=True,
                synthpose_index=None,
            ),
            CorrespondenceMarker(
                name="RFAulna",
                mhr_vertices=[8900],
                opensim_body="radius_r",  # NB: ulna marker bound to radius_r (verbatim)
                local_offset=np.array([-0.037, -0.218, -0.012], dtype=np.float64),
                fixed=False,
                synthpose_index=None,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Construction + round-trip
# ---------------------------------------------------------------------------


def test_construct_valid_map():
    m = _make_fake_map()
    assert len(m.markers) == 4
    assert m.opensim_model == "Pose2Sim_Wholebody"


def test_roundtrip_correspondence_map(tmp_path):
    cm = _make_fake_map()
    out = save(cm, tmp_path / "cm")
    back = load(CorrespondenceMap, out)
    assert_models_equal(cm, back)


def test_roundtrip_preserves_local_offset_dtype(tmp_path):
    """local_offset is (3,) float; we don't cast, dtype must survive save/load."""
    cm = CorrespondenceMap(
        mhr_topology_id="mhr_v1",
        opensim_model="Pose2Sim_Wholebody",
        marker_set="pose2sim_v0",
        frame_alignment=FrameAlignment(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        ),
        markers=[
            CorrespondenceMarker(
                name="RASI",
                mhr_vertices=[0],
                opensim_body="pelvis",
                local_offset=np.array([0.001, 0.002, 0.003], dtype=np.float32),
                fixed=True,
            )
        ],
    )
    out = save(cm, tmp_path / "cm")
    back = load(CorrespondenceMap, out)
    assert back.markers[0].local_offset.dtype == np.float32


# ---------------------------------------------------------------------------
# Validation: rejection paths
# ---------------------------------------------------------------------------


def test_rejects_unknown_marker_name():
    with pytest.raises(ValidationError, match="unknown landmark"):
        CorrespondenceMarker(
            name="not_a_real_marker",
            mhr_vertices=[1],
            opensim_body="pelvis",
            local_offset=np.zeros(3),
            fixed=True,
        )


def test_rejects_unknown_body_for_known_model():
    with pytest.raises(ValidationError, match="unknown body"):
        CorrespondenceMap(
            mhr_topology_id="mhr_v1",
            opensim_model="Pose2Sim_Wholebody",
            marker_set="pose2sim_v0",
            frame_alignment=FrameAlignment(
                rotation=np.eye(3),
                translation=np.zeros(3),
            ),
            markers=[
                CorrespondenceMarker(
                    name="RASI",
                    mhr_vertices=[1],
                    opensim_body="not_a_real_body",
                    local_offset=np.zeros(3),
                    fixed=True,
                )
            ],
        )


def test_rejects_unknown_opensim_model():
    with pytest.raises(ValidationError, match="unknown opensim_model"):
        CorrespondenceMap(
            mhr_topology_id="mhr_v1",
            opensim_model="Rajagopal2016",  # no longer registered
            marker_set="pose2sim_v0",
            frame_alignment=FrameAlignment(
                rotation=np.eye(3),
                translation=np.zeros(3),
            ),
            markers=[],
        )


def test_rejects_duplicate_marker_names():
    with pytest.raises(ValidationError, match="duplicate marker names"):
        CorrespondenceMap(
            mhr_topology_id="mhr_v1",
            opensim_model="Pose2Sim_Wholebody",
            marker_set="pose2sim_v0",
            frame_alignment=FrameAlignment(
                rotation=np.eye(3),
                translation=np.zeros(3),
            ),
            markers=[
                CorrespondenceMarker(
                    name="RASI",
                    mhr_vertices=[1],
                    opensim_body="pelvis",
                    local_offset=np.zeros(3),
                    fixed=True,
                ),
                CorrespondenceMarker(
                    name="RASI",  # duplicate
                    mhr_vertices=[2],
                    opensim_body="pelvis",
                    local_offset=np.zeros(3),
                    fixed=True,
                ),
            ],
        )


def test_rejects_empty_mhr_vertices():
    with pytest.raises(ValidationError, match="mhr_vertices must not be empty"):
        CorrespondenceMarker(
            name="RASI",
            mhr_vertices=[],
            opensim_body="pelvis",
            local_offset=np.zeros(3),
            fixed=True,
        )


def test_rejects_negative_mhr_vertex_index():
    with pytest.raises(ValidationError, match="mhr_vertices must be a list of non-negative"):
        CorrespondenceMarker(
            name="RASI",
            mhr_vertices=[10, -1, 20],
            opensim_body="pelvis",
            local_offset=np.zeros(3),
            fixed=True,
        )


# ---------------------------------------------------------------------------
# Sanity: the verbatim-casing constraints survive a roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_tricky_casing(tmp_path):
    """RFAulna on radius_r and RWrist_hand on hand_r — both must come back byte-identical."""
    cm = CorrespondenceMap(
        mhr_topology_id="mhr_v1",
        opensim_model="Pose2Sim_Wholebody",
        marker_set="pose2sim_v0",
        frame_alignment=FrameAlignment(
            rotation=np.eye(3),
            translation=np.zeros(3),
        ),
        markers=[
            CorrespondenceMarker(
                name="RFAulna",
                mhr_vertices=[1],
                opensim_body="radius_r",
                local_offset=np.zeros(3),
                fixed=True,
            ),
            CorrespondenceMarker(
                name="RWrist_hand",
                mhr_vertices=[2],
                opensim_body="hand_r",
                local_offset=np.zeros(3),
                fixed=True,
            ),
        ],
    )
    out = save(cm, tmp_path / "cm")
    back = load(CorrespondenceMap, out)
    names = [m.name for m in back.markers]
    bodies = [m.opensim_body for m in back.markers]
    assert names == ["RFAulna", "RWrist_hand"]
    assert bodies == ["radius_r", "hand_r"]
