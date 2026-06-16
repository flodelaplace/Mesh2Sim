"""Test fixtures: factories for each contract.

Each factory returns a fully-valid instance with deterministic data (seeded RNG), small
enough to be cheap but non-trivial (no all-zero arrays).
"""

from __future__ import annotations

import numpy as np
import pytest
from mesh2sim.contracts import (
    AnatomicalObservation,
    AnatomicalTrajectory,
    BiomechFit,
    BodyEstimate,
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


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Transverse value-object factories
# ---------------------------------------------------------------------------


def make_camera(rng: np.random.Generator) -> CameraParams:
    return CameraParams(
        view_id="cam0",
        K=np.array(
            [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros(5, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        t=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        resolution=(640, 480),
        time_offset=0.0,
    )


def make_mesh(rng: np.random.Generator, v: int = 12, f: int = 6) -> MeshData:
    return MeshData(
        vertices=rng.standard_normal((v, 3)).astype(np.float32),
        topology_id="mhr_v1",
        faces=rng.integers(0, v, size=(f, 3), dtype=np.int32),
    )


def make_skeleton(rng: np.random.Generator) -> SkeletonState:
    names = ["pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l"]
    j = len(names)
    return SkeletonState(
        joint_positions=rng.standard_normal((j, 3)).astype(np.float64),
        joint_orientations=np.tile(np.eye(3, dtype=np.float64), (j, 1, 1)),
        joint_names=names,
    )


def make_keypoints(rng: np.random.Generator) -> Keypoints2D:
    # Keypoint names are estimator-specific (e.g. SynthPose) and NOT validated against
    # LANDMARKS — they're a separate vocabulary. Kept lowercase here on purpose.
    names = ["rhip", "rknee", "rankle"]
    k = len(names)
    return Keypoints2D(
        names=names,
        xy=(rng.random((k, 2)) * 640).astype(np.float32),
        confidence=rng.random(k).astype(np.float32),
    )


def make_shape_descriptor(rng: np.random.Generator) -> ShapeDescriptor:
    return ShapeDescriptor(
        representation=ShapeRepresentation.per_segment_scale,
        data={
            "pelvis": np.array([1.00, 1.00, 1.00], dtype=np.float64),
            "femur_r": np.array([1.05, 1.00, 1.05], dtype=np.float64),
            "femur_l": np.array([1.04, 1.00, 1.04], dtype=np.float64),
        },
        source_model="Pose2Sim_Wholebody",
    )


def make_provenance() -> Provenance:
    return Provenance(
        estimator_id="fast-sam-3d-body@0.1",
        adapter_id="mhr-to-anatomy@v0",
        correspondence_map_id="map_v0_2026-06-15",
        created_at="2026-06-15T10:00:00Z",
        extra={"notes": "fixture"},
    )


# ---------------------------------------------------------------------------
# Five-contract factories (pytest fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture
def body_estimate(rng: np.random.Generator) -> BodyEstimate:
    return BodyEstimate(
        estimator_id="fast-sam-3d-body@0.1",
        frame_id=0,
        view_id="cam0",
        timestamp=0.0,
        capabilities=Capabilities(
            has_mesh=True,
            has_skeleton=True,
            has_2d_keypoints=True,
            has_native_params=True,
        ),
        native_params={"identity_len": 45, "model_parameters_len": 204},
        mesh=make_mesh(rng),
        skeleton_state=make_skeleton(rng),
        keypoints_2d=make_keypoints(rng),
        camera=make_camera(rng),
        frame_shape=(480, 640),
    )


@pytest.fixture
def anatomical_observation(rng: np.random.Generator) -> AnatomicalObservation:
    return AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        timestamp=0.0,
        landmarks={
            "RASI": Landmark(
                pos_3d=np.array([0.10, 1.00, 0.0]),
                pos_2d=np.array([320.0, 240.0]),
                confidence=0.95,
                visibility=1.0,
                source=Source.bony,
            ),
            "LASI": Landmark(
                pos_3d=np.array([-0.10, 1.00, 0.0]),
                confidence=0.94,
                source=Source.bony,
            ),
            "RLFC": Landmark(
                pos_3d=np.array([0.12, 0.50, 0.0]),
                confidence=0.85,
                source=Source.soft,
            ),
        },
        pos3d_frame=Pos3DFrame.world,
        segment_frames={
            "pelvis": np.eye(3, dtype=np.float64),
            "femur_r": np.eye(3, dtype=np.float64),
        },
        shape_descriptor=make_shape_descriptor(rng),
        joint_centers_init={"RHJC": np.array([0.10, 0.95, 0.0])},
        dense_surface=make_mesh(rng),
        capabilities=Capabilities(
            has_mesh=True,
            has_segment_frames=True,
            has_shape_descriptor=True,
        ),
        provenance=make_provenance(),
    )


@pytest.fixture
def anatomical_trajectory(rng: np.random.Generator) -> AnatomicalTrajectory:
    names = ["RASI", "LASI", "RLFC", "LLFC"]
    t, ell = 8, len(names)
    return AnatomicalTrajectory(
        subject_id="S001",
        trial_id="gait_01",
        task=Task.gait,
        mode=Mode.multi,
        fps=60.0,
        landmark_names=names,
        positions=rng.standard_normal((t, ell, 3)).astype(np.float64),
        confidence=rng.random((t, ell)).astype(np.float32),
        shape_descriptor=make_shape_descriptor(rng),
        views_used=["cam0", "cam1"],
        uncertainty=(rng.random((t, ell, 3)) * 0.01).astype(np.float64),
        provenance=make_provenance(),
    )


@pytest.fixture
def biomech_fit(rng: np.random.Generator) -> BiomechFit:
    dofs = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "hip_flexion_r", "knee_angle_r"]
    t, d = 8, len(dofs)
    return BiomechFit(
        subject_id="S001",
        trial_id="gait_01",
        model_id="Pose2Sim_Wholebody",
        scaled_model_path="/tmp/scaled.osim",
        dof_names=dofs,
        angles=rng.standard_normal((t, d)).astype(np.float64),
        motion_path=None,
        marker_offsets={
            "RASI": np.array([0.005, 0.0, 0.0]),
            "LASI": np.array([-0.005, 0.0, 0.0]),
        },
        marker_residuals=(rng.random(t) * 0.01).astype(np.float32),
        uncertainty=(rng.random((t, d, 2)) * 0.05).astype(np.float64),
        provenance=make_provenance(),
    )


@pytest.fixture
def all_contracts(
    body_estimate: BodyEstimate,
    anatomical_observation: AnatomicalObservation,
    anatomical_trajectory: AnatomicalTrajectory,
    biomech_fit: BiomechFit,
) -> dict[str, object]:
    return {
        "BodyEstimate": body_estimate,
        "AnatomicalObservation": anatomical_observation,
        "AnatomicalTrajectory": anatomical_trajectory,
        "BiomechFit": biomech_fit,
    }
