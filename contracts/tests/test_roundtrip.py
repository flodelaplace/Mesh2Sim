"""Round-trip ``save → load`` must be exact for every contract.

Exact means: ``np.array_equal`` AND identical ``dtype`` for every ndarray field, and
``==`` for everything else. We assert per-schema and across a dtype matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
from mesh2sim.contracts import (
    AnatomicalObservation,
    AnatomicalTrajectory,
    BiomechFit,
    BodyEstimate,
    Capabilities,
    Landmark,
    MeshData,
    Mode,
    Pos3DFrame,
    Provenance,
    ShapeDescriptor,
    ShapeRepresentation,
    Source,
    Task,
    load,
    save,
)

from ._equality import assert_models_equal

# ---------------------------------------------------------------------------
# Per-schema round-trip (uses conftest fixtures)
# ---------------------------------------------------------------------------


def test_roundtrip_body_estimate(tmp_path, body_estimate):
    save(body_estimate, tmp_path / "be")
    loaded = load(BodyEstimate, tmp_path / "be")
    assert_models_equal(body_estimate, loaded)


def test_roundtrip_anatomical_observation(tmp_path, anatomical_observation):
    save(anatomical_observation, tmp_path / "obs")
    loaded = load(AnatomicalObservation, tmp_path / "obs")
    assert_models_equal(anatomical_observation, loaded)


def test_roundtrip_anatomical_trajectory(tmp_path, anatomical_trajectory):
    save(anatomical_trajectory, tmp_path / "traj")
    loaded = load(AnatomicalTrajectory, tmp_path / "traj")
    assert_models_equal(anatomical_trajectory, loaded)


def test_roundtrip_biomech_fit(tmp_path, biomech_fit):
    save(biomech_fit, tmp_path / "fit")
    loaded = load(BiomechFit, tmp_path / "fit")
    assert_models_equal(biomech_fit, loaded)


# ---------------------------------------------------------------------------
# Dtype preservation (the main thing we don't want to drift silently)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trajectory_positions_dtype_preserved(tmp_path, dtype):
    """positions has no documented cast — whatever dtype goes in must come back."""
    t, ell = 4, 3
    names = ["r_asis", "l_asis", "r_knee_lat"]
    traj = AnatomicalTrajectory(
        subject_id="S",
        trial_id="t",
        task=Task.gait,
        mode=Mode.mono,
        fps=30.0,
        landmark_names=names,
        positions=np.arange(t * ell * 3).reshape(t, ell, 3).astype(dtype),
        confidence=np.ones((t, ell), dtype=np.float32),
        shape_descriptor=ShapeDescriptor(
            representation=ShapeRepresentation.opaque,
            data={"global": np.zeros(4, dtype=np.float64)},
            source_model="Rajagopal2016",
        ),
        views_used=["mono"],
        provenance=Provenance(),
    )
    assert traj.positions.dtype == dtype  # creation preserved
    save(traj, tmp_path / "t")
    back = load(AnatomicalTrajectory, tmp_path / "t")
    assert back.positions.dtype == dtype  # round-trip preserved
    assert np.array_equal(back.positions, traj.positions)


def test_mesh_vertices_are_float32(tmp_path):
    """Vertices are documented as float32 in the spec — that cast happens at construction."""
    # Float64 input gets cast down on creation, then survives round-trip as float32.
    m_in = MeshData(
        vertices=np.random.default_rng(0).standard_normal((10, 3)).astype(np.float64),
        topology_id="mhr_v1",
    )
    assert m_in.vertices.dtype == np.float32

    # Roundtrip a contract that holds the mesh.
    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={"r_asis": Landmark()},
        pos3d_frame=Pos3DFrame.none,
        dense_surface=m_in,
        capabilities=Capabilities(has_mesh=True),
        provenance=Provenance(),
    )
    save(obs, tmp_path / "obs")
    back = load(AnatomicalObservation, tmp_path / "obs")
    assert back.dense_surface.vertices.dtype == np.float32
    assert np.array_equal(back.dense_surface.vertices, m_in.vertices)


def test_camera_intrinsics_dtype_preserved(tmp_path):
    """K and other camera arrays keep their dtype across save/load."""
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cam_be = BodyEstimate(
        estimator_id="x",
        frame_id=0,
        view_id="cam0",
        capabilities=Capabilities(),
        camera={"view_id": "cam0", "K": K, "resolution": (640, 480)},
    )
    assert cam_be.camera is not None
    assert cam_be.camera.K.dtype == np.float32

    save(cam_be, tmp_path / "be")
    back = load(BodyEstimate, tmp_path / "be")
    assert back.camera is not None
    assert back.camera.K.dtype == np.float32
    assert np.array_equal(back.camera.K, K)


def test_int_dtype_preserved_for_faces(tmp_path):
    """Face index dtype is whatever the user provided (we don't cast)."""
    faces32 = np.zeros((4, 3), dtype=np.int32)
    m = MeshData(
        vertices=np.zeros((6, 3), dtype=np.float32),
        topology_id="mhr_v1",
        faces=faces32,
    )
    assert m.faces.dtype == np.int32

    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={"r_asis": Landmark()},
        pos3d_frame=Pos3DFrame.none,
        dense_surface=m,
        capabilities=Capabilities(has_mesh=True),
        provenance=Provenance(),
    )
    save(obs, tmp_path / "obs")
    back = load(AnatomicalObservation, tmp_path / "obs")
    assert back.dense_surface.faces.dtype == np.int32


# ---------------------------------------------------------------------------
# Quickstart smoke: fabriquer un AnatomicalObservation, save, load (per task spec)
# ---------------------------------------------------------------------------


def test_quickstart_anatomical_observation(tmp_path):
    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={
            "r_asis": Landmark(
                pos_3d=np.array([0.1, 1.0, 0.0]),
                confidence=0.9,
                source=Source.bony,
            ),
        },
        pos3d_frame=Pos3DFrame.world,
        capabilities=Capabilities(),
        provenance=Provenance(),
    )
    save(obs, tmp_path / "obs")
    back = load(AnatomicalObservation, tmp_path / "obs")
    assert back.landmarks["r_asis"].pos_3d is not None
    assert np.array_equal(back.landmarks["r_asis"].pos_3d, np.array([0.1, 1.0, 0.0]))
    assert back.landmarks["r_asis"].source == Source.bony
