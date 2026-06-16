"""Fixtures that fabricate per-person dicts in the **exact** shape produced by
``sam_3d_body.SAM3DBodyEstimator.process_one_image`` after ``recursive_to(out, "numpy")``.

These fixtures are the contract between the upstream output format (which we
do not control) and our adapter (which we do). If the vendored core ever
changes the output keys/shapes, these fixtures should fail loudly and we
re-bind the adapter, not the other way around.

Shapes verified against the vendored code at the pinned commit
``936894c37e51de9918012bcbc9ba2d9c20f73252``.
"""

from __future__ import annotations

import numpy as np
import pytest

# Constants matching the vendored MHR topology.
N_VERTICES = 18439
N_JOINTS = 127
N_KP70 = 70


def _xyxy_bbox(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def make_sam3db_person_output(
    *,
    seed: int = 0,
    bbox: np.ndarray | None = None,
) -> dict:
    """Build one synthetic per-person output dict (numpy, CPU)."""
    rng = np.random.default_rng(seed)
    if bbox is None:
        bbox = _xyxy_bbox(100.0, 50.0, 540.0, 470.0)
    return {
        # Pipeline metadata produced by the upstream code (NOT propagated to BodyEstimate).
        "bbox": bbox,
        "mask": None,
        # Camera / projection info.
        "focal_length": np.float32(1200.0),
        "pred_cam_t": rng.standard_normal(3).astype(np.float32),
        # MHR opaque parameters.
        "shape_params": rng.standard_normal(45).astype(np.float32),     # identity 45
        "expr_params": rng.standard_normal(72).astype(np.float32),      # expression 72
        "scale_params": rng.standard_normal(28).astype(np.float32),     # bone scales 28
        "global_rot": rng.standard_normal(6).astype(np.float32),
        "body_pose_params": rng.standard_normal(260).astype(np.float32),
        "hand_pose_params": rng.standard_normal(108).astype(np.float32),
        "pred_pose_raw": rng.standard_normal(127).astype(np.float32),
        # Mesh + skeleton.
        "pred_vertices": rng.standard_normal((N_VERTICES, 3)).astype(np.float32),
        "pred_joint_coords": rng.standard_normal((N_JOINTS, 3)).astype(np.float32),
        # joint_global_rots are (J, 3, 3) per the upstream code (sliced as
        # ``[..., 78, [1, 2], :]``). We synthesise valid rotation matrices
        # (identity here) — the adapter only cares about shape/dtype.
        "pred_global_rots": np.broadcast_to(
            np.eye(3, dtype=np.float32), (N_JOINTS, 3, 3)
        ).copy(),
        # 2D keypoints (no confidence is emitted upstream).
        "pred_keypoints_2d": (rng.random((N_KP70, 2)) * 640).astype(np.float32),
        "pred_keypoints_3d": rng.standard_normal((N_KP70, 3)).astype(np.float32),
    }


@pytest.fixture
def sam3db_one_person() -> dict:
    return make_sam3db_person_output(seed=42)


@pytest.fixture
def sam3db_two_people() -> list[dict]:
    """A larger and a smaller person; the larger should win main-subject selection."""
    small = make_sam3db_person_output(seed=1, bbox=_xyxy_bbox(10.0, 10.0, 60.0, 80.0))   # area 50*70 = 3500
    large = make_sam3db_person_output(seed=2, bbox=_xyxy_bbox(200.0, 50.0, 540.0, 460.0))  # area 340*410 = 139400
    return [small, large]


@pytest.fixture
def fake_frame_rgb() -> np.ndarray:
    """A small synthetic RGB frame. The estimator is mocked so size is cosmetic."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
