"""Adapter unit tests — pure, no GPU.

Coverage requested by the ticket:
- 3 calculation cases: single-vertex bony (RLMAL), soft-tissue source (RFLT),
  multi-vertex centroid (RWrist_hand).
- 3 error cases: topology mismatch, vertex index out of bounds, missing mesh.
- Round-trip through the contracts IO.
- Plus: vocab rejection at CorrespondenceMap construction (the adapter relies on this
  layer, but we add a smoke test to catch a contract regression).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mesh2sim.contracts import (
    AnatomicalObservation,
    BodyEstimate,
    Capabilities,
    CorrespondenceMap,
    CorrespondenceMarker,
    FrameAlignment,
    MeshData,
    Pos3DFrame,
    Source,
    load,
    save,
)
from mesh2sim_anatomical_adapter import (
    ADAPTER_ID,
    TopologyMismatchError,
    VertexOutOfBoundsError,
    body_estimate_to_anatomical_observation,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Calculation cases
# ---------------------------------------------------------------------------


def test_emits_one_landmark_per_marker(body_estimate, correspondence_map):
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    expected_names = {m.name for m in correspondence_map.markers}
    assert set(obs.landmarks) == expected_names
    assert len(obs.landmarks) == 7


def test_single_vertex_bony_marker_position_is_byte_exact(body_estimate, correspondence_map):
    """RLMAL: bony, single vertex 17393. The synthetic mesh has vertex i = (i, 2i, 3i),
    so RLMAL must come out exactly as (17393, 34786, 52179)."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    lm = obs.landmarks["RLMAL"]
    assert lm.source == Source.bony
    expected = np.array([17393.0, 2 * 17393.0, 3 * 17393.0])
    np.testing.assert_array_equal(lm.pos_3d, expected)


def test_soft_tissue_marker_has_source_soft(body_estimate, correspondence_map):
    """RFLT: thigh cluster, fixed=False → Source.soft."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    lm = obs.landmarks["RFLT"]
    assert lm.source == Source.soft


def test_multi_vertex_marker_is_centroid_of_mhr_vertices(body_estimate, correspondence_map):
    """RWrist_hand: mhr_vertices = [12001, 12002, 12003]. With the synthetic mesh
    (vertex i = (i, 2i, 3i)), centroid = ((12001+12002+12003)/3, *2, *3) = (12002, 24004, 36006)."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    lm = obs.landmarks["RWrist_hand"]
    assert lm.source == Source.bony  # fixture flags it as fixed=True
    expected = np.array([12002.0, 2 * 12002.0, 3 * 12002.0])
    np.testing.assert_allclose(lm.pos_3d, expected, atol=1e-9)


def test_pos3d_frame_is_camera_not_world(body_estimate, correspondence_map):
    """The adapter does NOT transform to world; T6 does."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    assert obs.pos3d_frame == Pos3DFrame.camera


def test_provenance_is_correctly_threaded(body_estimate, correspondence_map):
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    assert obs.provenance.estimator_id == body_estimate.estimator_id
    assert obs.provenance.adapter_id == ADAPTER_ID
    assert obs.provenance.correspondence_map_id == correspondence_map.marker_set


def test_frame_id_view_id_timestamp_passed_through(body_estimate, correspondence_map):
    """The adapter inherits the frame_id/view_id/timestamp from the BodyEstimate."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    assert obs.frame_id == body_estimate.frame_id
    assert obs.view_id == body_estimate.view_id
    assert obs.timestamp == body_estimate.timestamp


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_topology_mismatch_raises(body_estimate, correspondence_map):
    """The topology guard MUST refuse a mesh whose topology_id differs from the map's.
    Without this guard, the vertex indices would silently point to the wrong points."""
    # Forge a BodyEstimate with a different topology id.
    bad_mesh = MeshData(
        vertices=body_estimate.mesh.vertices,
        topology_id="some_other_topology_v2",
        faces=None,
    )
    bad_be = body_estimate.model_copy(update={"mesh": bad_mesh})
    with pytest.raises(TopologyMismatchError, match="mhr_topology_id mismatch"):
        body_estimate_to_anatomical_observation(bad_be, correspondence_map)


def test_vertex_out_of_bounds_raises():
    """An out-of-range index must surface a clear error, not a silent garbage position."""
    fa = FrameAlignment(rotation=np.eye(3), translation=np.zeros(3))
    # Map references vertex 99_999, our synthetic mesh has only 18_439.
    bad_map = CorrespondenceMap(
        mhr_topology_id="mhr_v1",
        opensim_model="Pose2Sim_Wholebody",
        marker_set="oob_test",
        frame_alignment=fa,
        markers=[
            CorrespondenceMarker(
                name="RASI",
                mhr_vertices=[99_999],
                opensim_body="pelvis",
                local_offset=np.zeros(3),
                fixed=True,
            )
        ],
    )
    tiny_mesh = MeshData(
        vertices=np.zeros((100, 3), dtype=np.float32),
        topology_id="mhr_v1",
    )
    be = BodyEstimate(
        estimator_id="x",
        frame_id=0,
        view_id="mono",
        capabilities=Capabilities(has_mesh=True),
        mesh=tiny_mesh,
    )
    with pytest.raises(VertexOutOfBoundsError, match=r"out of range \[0, 100\)"):
        body_estimate_to_anatomical_observation(be, bad_map)


def test_negative_vertex_index_rejected_at_construction():
    """The contract itself rejects negative indices in CorrespondenceMarker.mhr_vertices
    (validated at construction). The adapter doesn't even see them. This test pins that
    behaviour so a contract loosening doesn't unwittingly bypass it."""
    with pytest.raises(ValidationError, match="non-negative"):
        CorrespondenceMarker(
            name="RASI",
            mhr_vertices=[10, -1, 20],
            opensim_body="pelvis",
            local_offset=np.zeros(3),
            fixed=True,
        )


def test_missing_mesh_raises(correspondence_map):
    """If the upstream estimator didn't emit mesh data (has_mesh=False), the adapter
    must refuse — there's nothing to look vertices up in."""
    be = BodyEstimate(
        estimator_id="x",
        frame_id=0,
        view_id="mono",
        capabilities=Capabilities(has_mesh=False),
        mesh=None,
    )
    with pytest.raises(ValueError, match="no mesh"):
        body_estimate_to_anatomical_observation(be, correspondence_map)


def test_unknown_landmark_name_rejected_at_map_construction():
    """Vocab guard lives in the contract itself; we lock that this still holds, since the
    adapter relies on it (no name check in the adapter — it trusts the map)."""
    fa = FrameAlignment(rotation=np.eye(3), translation=np.zeros(3))
    with pytest.raises(ValidationError, match="unknown landmark"):
        CorrespondenceMap(
            mhr_topology_id="mhr_v1",
            opensim_model="Pose2Sim_Wholebody",
            marker_set="bad_name_test",
            frame_alignment=fa,
            markers=[
                CorrespondenceMarker(
                    name="not_a_real_marker",
                    mhr_vertices=[10],
                    opensim_body="pelvis",
                    local_offset=np.zeros(3),
                    fixed=True,
                )
            ],
        )


# ---------------------------------------------------------------------------
# Round-trip through contracts IO
# ---------------------------------------------------------------------------


def test_anatomical_observation_roundtrips_through_contracts_io(
    tmp_path: Path, body_estimate, correspondence_map
):
    """The produced AnatomicalObservation must save → load byte-identical."""
    obs = body_estimate_to_anatomical_observation(body_estimate, correspondence_map)
    out = save(obs, tmp_path / "obs")
    back = load(AnatomicalObservation, out)

    # Same set of landmarks, same source flags, byte-equal pos_3d arrays.
    assert set(back.landmarks) == set(obs.landmarks)
    for name in obs.landmarks:
        orig = obs.landmarks[name]
        rt = back.landmarks[name]
        assert orig.source == rt.source
        np.testing.assert_array_equal(orig.pos_3d, rt.pos_3d)
        assert orig.confidence == rt.confidence
        assert orig.visibility == rt.visibility
    assert back.pos3d_frame == obs.pos3d_frame
    assert back.provenance.adapter_id == ADAPTER_ID
    assert back.provenance.correspondence_map_id == correspondence_map.marker_set


# ---------------------------------------------------------------------------
# Marker count sanity — the 7-fixture coverage
# ---------------------------------------------------------------------------


def test_fixture_covers_the_three_calculation_cases(correspondence_map):
    """The fixture is the contract between us and the test suite. If someone touches
    example_correspondence_map.json, this test reminds them which cases must remain
    represented."""
    names_to_role = {
        m.name: ("bony" if m.fixed else "soft", len(m.mhr_vertices))
        for m in correspondence_map.markers
    }

    # Single-vertex bony (with the pinned vertex index 17393).
    assert names_to_role["RLMAL"] == ("bony", 1)
    rlmal = next(m for m in correspondence_map.markers if m.name == "RLMAL")
    assert rlmal.mhr_vertices == [17393]

    # Soft tissue.
    assert names_to_role["RFLT"][0] == "soft"

    # Multi-vertex centroid.
    assert names_to_role["RWrist_hand"][1] >= 2
