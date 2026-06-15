"""Validation rules: vocab + capability coherence + array shapes."""

from __future__ import annotations

import numpy as np
import pytest
from mesh2sim.contracts import (
    LANDMARKS,
    AnatomicalObservation,
    AnatomicalTrajectory,
    BiomechFit,
    BodyEstimate,
    CameraParams,
    Capabilities,
    Landmark,
    MeshData,
    Mode,
    Pos3DFrame,
    Provenance,
    ShapeDescriptor,
    ShapeRepresentation,
    Task,
    is_valid_landmark,
    is_valid_segment,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocab_contains_opencap_bony_landmarks():
    # A few core bony landmarks that must be in any OpenCap-like set.
    for must_have in ("r_asis", "l_asis", "r_knee_lat", "l_knee_lat", "c7"):
        assert is_valid_landmark(must_have), must_have


def test_vocab_rajagopal2016_segments():
    for must_have in ("pelvis", "femur_r", "tibia_r", "calcn_r"):
        assert is_valid_segment(must_have, "Rajagopal2016")


def test_anatomical_observation_rejects_unknown_landmark():
    with pytest.raises(ValidationError, match="unknown landmark"):
        AnatomicalObservation(
            frame_id=0,
            view_id="cam0",
            landmarks={"not_a_real_marker": Landmark()},
            pos3d_frame=Pos3DFrame.none,
            capabilities=Capabilities(),
            provenance=Provenance(),
        )


def test_trajectory_rejects_unknown_landmark():
    with pytest.raises(ValidationError, match="unknown landmark"):
        AnatomicalTrajectory(
            subject_id="S",
            trial_id="t",
            task=Task.gait,
            mode=Mode.mono,
            fps=30.0,
            landmark_names=["bogus_marker"],
            positions=np.zeros((1, 1, 3)),
            confidence=np.zeros((1, 1)),
            shape_descriptor=ShapeDescriptor(
                representation=ShapeRepresentation.opaque,
                data={"g": np.zeros(2)},
                source_model="Rajagopal2016",
            ),
            views_used=["mono"],
            provenance=Provenance(),
        )


def test_biomech_fit_rejects_unknown_marker():
    with pytest.raises(ValidationError, match="unknown landmark"):
        BiomechFit(
            subject_id="S",
            trial_id="t",
            model_id="Rajagopal2016",
            scaled_model_path="/tmp/x.osim",
            dof_names=["d0"],
            angles=np.zeros((2, 1)),
            marker_offsets={"bogus_marker": np.zeros(3)},
            marker_residuals=np.zeros(2),
            provenance=Provenance(),
        )


# ---------------------------------------------------------------------------
# Capability ↔ field coherence
# ---------------------------------------------------------------------------


def test_body_estimate_has_mesh_flag_must_match_field():
    # flag True but no mesh
    with pytest.raises(ValidationError, match="has_mesh"):
        BodyEstimate(
            estimator_id="x",
            frame_id=0,
            view_id="cam0",
            capabilities=Capabilities(has_mesh=True),
        )

    # mesh present but flag False
    with pytest.raises(ValidationError, match="has_mesh"):
        BodyEstimate(
            estimator_id="x",
            frame_id=0,
            view_id="cam0",
            capabilities=Capabilities(has_mesh=False),
            mesh=MeshData(
                vertices=np.zeros((3, 3), dtype=np.float32),
                topology_id="mhr_v1",
            ),
        )


def test_anatomical_observation_capabilities_must_match():
    with pytest.raises(ValidationError, match="has_segment_frames"):
        AnatomicalObservation(
            frame_id=0,
            view_id="cam0",
            landmarks={},
            pos3d_frame=Pos3DFrame.none,
            capabilities=Capabilities(has_segment_frames=True),  # but no segment_frames
            provenance=Provenance(),
        )

    with pytest.raises(ValidationError, match="has_shape_descriptor"):
        AnatomicalObservation(
            frame_id=0,
            view_id="cam0",
            landmarks={},
            pos3d_frame=Pos3DFrame.none,
            capabilities=Capabilities(),  # flag False
            shape_descriptor=ShapeDescriptor(  # but present
                representation=ShapeRepresentation.opaque,
                data={"g": np.zeros(2)},
                source_model="Rajagopal2016",
            ),
            provenance=Provenance(),
        )


# ---------------------------------------------------------------------------
# Array shapes
# ---------------------------------------------------------------------------


def test_camera_K_must_be_3x3():
    with pytest.raises(ValidationError, match="expected shape"):
        CameraParams(view_id="v", K=np.eye(2), resolution=(640, 480))


def test_mesh_vertices_must_be_Vx3():
    with pytest.raises(ValidationError, match="expected"):
        MeshData(vertices=np.zeros((10, 2), dtype=np.float32), topology_id="x")


def test_segment_frames_must_be_3x3():
    with pytest.raises(ValidationError, match=r"segment_frames\[.+\] must be \(3, 3\)"):
        AnatomicalObservation(
            frame_id=0,
            view_id="cam0",
            landmarks={},
            pos3d_frame=Pos3DFrame.none,
            segment_frames={"pelvis": np.eye(2)},
            capabilities=Capabilities(has_segment_frames=True),
            provenance=Provenance(),
        )


def test_trajectory_shape_consistency():
    # confidence shape doesn't match landmark_names / positions
    with pytest.raises(ValidationError, match="confidence must be"):
        AnatomicalTrajectory(
            subject_id="S",
            trial_id="t",
            task=Task.gait,
            mode=Mode.mono,
            fps=30.0,
            landmark_names=["r_asis", "l_asis"],
            positions=np.zeros((4, 2, 3)),
            confidence=np.zeros((4, 3)),  # wrong L
            shape_descriptor=ShapeDescriptor(
                representation=ShapeRepresentation.opaque,
                data={"g": np.zeros(2)},
                source_model="Rajagopal2016",
            ),
            views_used=["mono"],
            provenance=Provenance(),
        )


def test_biomech_fit_needs_angles_or_motion_path():
    with pytest.raises(ValidationError, match="at least one of angles or motion_path"):
        BiomechFit(
            subject_id="S",
            trial_id="t",
            model_id="Rajagopal2016",
            scaled_model_path="/tmp/x.osim",
            dof_names=["d0"],
            marker_offsets={"r_asis": np.zeros(3)},
            marker_residuals=np.zeros(4),
            provenance=Provenance(),
        )


def test_vocab_size_is_reasonable():
    # Sanity: not empty, not absurdly small/large.
    assert 30 <= len(LANDMARKS) <= 100, f"unexpected LANDMARKS size: {len(LANDMARKS)}"
