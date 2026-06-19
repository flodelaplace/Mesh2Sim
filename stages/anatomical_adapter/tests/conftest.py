"""Test fixtures.

- ``correspondence_map`` : built from ``example_correspondence_map.json`` (7 markers
  covering single-vertex bony, single-vertex soft, and multi-vertex centroid cases).
- ``mhr_vertices``       : a deterministic ``(18439, 3)`` synthetic mesh where vertex
  ``i`` has coordinates ``(i, 2*i, 3*i)``. Lets every assertion in ``test_adapter.py``
  pin the expected position to closed-form integers.
- ``body_estimate``      : a fully-valid ``BodyEstimate`` built around the synthetic mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mesh2sim.contracts import (
    BodyEstimate,
    Capabilities,
    CorrespondenceMap,
    CorrespondenceMarker,
    FrameAlignment,
    Keypoints2D,
    MeshData,
    SkeletonState,
)

N_VERTICES = 18439  # matches the real MHR mesh

_FIXTURE_PATH = Path(__file__).resolve().parent / "example_correspondence_map.json"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return _FIXTURE_PATH


@pytest.fixture(scope="session")
def correspondence_map() -> CorrespondenceMap:
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return CorrespondenceMap(
        mhr_topology_id=data["mhr_topology_id"],
        opensim_model=data["opensim_model"],
        marker_set=data["marker_set"],
        frame_alignment=FrameAlignment(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.asarray(data["frame_alignment"]["translation"], dtype=np.float64),
            scale=float(data["frame_alignment"]["scale"]),
        ),
        markers=[
            CorrespondenceMarker(
                name=m["name"],
                mhr_vertices=list(m["mhr_vertices"]),
                opensim_body=m["opensim_body"],
                local_offset=np.asarray(m["local_offset"], dtype=np.float64),
                fixed=bool(m["fixed"]),
                synthpose_index=m["synthpose_index"],
            )
            for m in data["markers"]
        ],
    )


@pytest.fixture(scope="session")
def mhr_vertices() -> np.ndarray:
    """A synthetic but predictable mesh: vertex i = (i, 2i, 3i)."""
    idx = np.arange(N_VERTICES, dtype=np.float32)
    return np.stack([idx, 2.0 * idx, 3.0 * idx], axis=1)


@pytest.fixture
def body_estimate(mhr_vertices: np.ndarray) -> BodyEstimate:
    """Minimal but contract-valid BodyEstimate built around the synthetic mesh.

    Skeleton and keypoints fields are filled with placeholders to satisfy the contract
    when capabilities are True. The adapter never touches them anyway.
    """
    return BodyEstimate(
        estimator_id="anatomical-adapter-test@v0",
        frame_id=0,
        view_id="mono",
        timestamp=0.0,
        capabilities=Capabilities(
            has_mesh=True,
            has_skeleton=True,
            has_2d_keypoints=True,
            has_native_params=False,
        ),
        mesh=MeshData(
            vertices=mhr_vertices,
            topology_id="mhr_v1",
            faces=None,
        ),
        skeleton_state=SkeletonState(
            joint_positions=np.zeros((127, 3), dtype=np.float32),
            joint_orientations=np.broadcast_to(np.eye(3, dtype=np.float32), (127, 3, 3)).copy(),
            joint_names=[f"j{i:03d}" for i in range(127)],
        ),
        keypoints_2d=Keypoints2D(
            names=[f"k{i:02d}" for i in range(70)],
            xy=np.zeros((70, 2), dtype=np.float32),
            confidence=np.ones(70, dtype=np.float32),
        ),
        camera=None,
        frame_shape=(480, 640),
    )
