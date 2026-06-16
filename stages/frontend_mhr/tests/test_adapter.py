"""Adapter unit tests — fully mocked, no GPU.

We feed the adapter the per-person dict shape produced by
``SAM3DBodyEstimator.process_one_image`` and validate that the resulting
``BodyEstimate``:

1. Passes contract validation (constructed via the pydantic models).
2. Has coherent ``capabilities`` flags vs the optional fields actually present.
3. Round-trips through the contracts IO byte-equal (the test we trust most:
   if the adapter silently mishandles a dtype, the round-trip will catch it).
4. Routes pipeline metadata correctly: ``bbox`` / ``mask`` are NOT in
   ``BodyEstimate``, they are dropped at the adapter level.
5. Keeps the MHR-opaque parameters in ``native_params``, NOTHING else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mesh2sim.contracts import BodyEstimate, load, save
from mesh2sim_frontend_mhr import (
    MHR_JOINT_NAMES,
    MHR_N_JOINTS,
    MHR_TOPOLOGY_ID,
    sam3db_output_to_body_estimate,
)


def _adapt(output: dict, **overrides) -> BodyEstimate:
    defaults = dict(
        estimator_id="fast-sam-3d-body@test",
        frame_id=0,
        view_id="mono",
        timestamp=0.0,
        camera=None,
        frame_shape=(480, 640),
    )
    defaults.update(overrides)
    return sam3db_output_to_body_estimate(output, **defaults)


# ---------------------------------------------------------------------------
# 1. Basic shape & capability conformance
# ---------------------------------------------------------------------------


def test_adapter_returns_valid_body_estimate(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    assert be.estimator_id == "fast-sam-3d-body@test"
    assert be.frame_id == 0
    assert be.view_id == "mono"
    assert be.timestamp == 0.0
    assert be.frame_shape == (480, 640)


def test_mesh_has_expected_topology_and_dtype(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    assert be.mesh is not None
    assert be.mesh.topology_id == MHR_TOPOLOGY_ID
    assert be.mesh.vertices.shape == (18439, 3)
    # Contract enforces float32 for vertices; the adapter must not silently
    # double it.
    assert be.mesh.vertices.dtype == np.float32
    assert be.mesh.faces is None


def test_skeleton_state_has_127_joints(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    assert be.skeleton_state is not None
    assert be.skeleton_state.joint_positions.shape == (MHR_N_JOINTS, 3)
    assert be.skeleton_state.joint_orientations.shape == (MHR_N_JOINTS, 3, 3)
    assert be.skeleton_state.joint_names == MHR_JOINT_NAMES
    assert len(be.skeleton_state.joint_names) == MHR_N_JOINTS


def test_keypoints_2d_has_70_entries(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    assert be.keypoints_2d is not None
    assert len(be.keypoints_2d.names) == 70
    assert be.keypoints_2d.xy.shape == (70, 2)
    assert be.keypoints_2d.confidence.shape == (70,)
    # SAM3DBody doesn't expose confidence; adapter MUST surface uniform 1.0.
    assert np.allclose(be.keypoints_2d.confidence, 1.0)


def test_capabilities_match_present_fields(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    # All four fields are populated in the standard case → all four flags True.
    assert be.capabilities.has_mesh is True
    assert be.capabilities.has_skeleton is True
    assert be.capabilities.has_2d_keypoints is True
    assert be.capabilities.has_native_params is True


# ---------------------------------------------------------------------------
# 2. native_params is the MHR opaque set ONLY
# ---------------------------------------------------------------------------


def test_native_params_contains_mhr_opaque_keys(sam3db_one_person):
    be = _adapt(sam3db_one_person)
    assert be.native_params is not None
    expected = {
        "shape_params", "expr_params", "scale_params", "global_rot",
        "body_pose_params", "hand_pose_params", "pred_pose_raw",
        "pred_cam_t", "pred_keypoints_3d", "focal_length",
    }
    assert set(be.native_params) == expected


def test_native_params_does_not_leak_pipeline_metadata(sam3db_one_person):
    """bbox / mask / hand bboxes are PIPELINE-side and must not survive."""
    be = _adapt(sam3db_one_person)
    assert be.native_params is not None
    forbidden = {"bbox", "mask", "lhand_bbox", "rhand_bbox"}
    assert not (forbidden & set(be.native_params)), (
        f"pipeline metadata leaked into native_params: "
        f"{forbidden & set(be.native_params)}"
    )


def test_native_params_excludes_none_values():
    """Optional fields absent in the upstream output must be dropped, not stored as None."""
    output = {
        "pred_vertices": np.zeros((18439, 3), dtype=np.float32),
        "pred_joint_coords": np.zeros((127, 3), dtype=np.float32),
        "pred_global_rots": np.broadcast_to(np.eye(3, dtype=np.float32), (127, 3, 3)).copy(),
        "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
        "shape_params": np.zeros(45, dtype=np.float32),
        "expr_params": None,            # absent in this run
        "scale_params": np.zeros(28, dtype=np.float32),
        "global_rot": np.zeros(6, dtype=np.float32),
        "body_pose_params": np.zeros(260, dtype=np.float32),
        "hand_pose_params": np.zeros(108, dtype=np.float32),
        "pred_pose_raw": None,
        "pred_cam_t": np.zeros(3, dtype=np.float32),
        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
        "focal_length": np.float32(1200.0),
    }
    be = _adapt(output)
    assert be.native_params is not None
    assert "expr_params" not in be.native_params
    assert "pred_pose_raw" not in be.native_params
    assert "shape_params" in be.native_params


# ---------------------------------------------------------------------------
# 3. Round-trip through contracts IO
# ---------------------------------------------------------------------------


def test_roundtrip_through_contracts_io(tmp_path: Path, sam3db_one_person):
    """If the adapter mishandles a dtype/shape, the round-trip catches it."""
    be = _adapt(sam3db_one_person)
    out = save(be, tmp_path / "be")
    back = load(BodyEstimate, out)

    # The IO layer goes through pydantic re-validation, so the loaded BodyEstimate
    # is contract-valid by construction. We additionally verify byte-equality of
    # the heavy arrays — that's the dtype/shape audit.
    assert back.mesh is not None and be.mesh is not None
    assert back.mesh.vertices.dtype == be.mesh.vertices.dtype == np.float32
    assert np.array_equal(back.mesh.vertices, be.mesh.vertices)
    assert back.skeleton_state.joint_orientations.dtype == np.float32
    assert np.array_equal(
        back.skeleton_state.joint_orientations, be.skeleton_state.joint_orientations
    )


# ---------------------------------------------------------------------------
# 4. Missing-input guardrails
# ---------------------------------------------------------------------------


def test_adapter_rejects_missing_required_key(sam3db_one_person):
    bad = dict(sam3db_one_person)
    del bad["pred_vertices"]
    with pytest.raises(KeyError, match="missing required keys"):
        _adapt(bad)


def test_adapter_rejects_none_required_value(sam3db_one_person):
    bad = dict(sam3db_one_person)
    bad["pred_keypoints_2d"] = None
    with pytest.raises(KeyError, match="missing required keys"):
        _adapt(bad)


# ---------------------------------------------------------------------------
# 5. Optional camera + frame_shape pass through verbatim
# ---------------------------------------------------------------------------


def test_camera_and_frame_shape_pass_through(sam3db_one_person):
    from mesh2sim.contracts import CameraParams

    K = np.array(
        [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cam = CameraParams(view_id="cam0", K=K, resolution=(640, 480))
    be = _adapt(sam3db_one_person, camera=cam, frame_shape=(720, 1280), view_id="cam0")
    assert be.camera is not None
    assert be.camera.view_id == "cam0"
    assert np.array_equal(be.camera.K, K)
    assert be.frame_shape == (720, 1280)
    assert be.view_id == "cam0"
