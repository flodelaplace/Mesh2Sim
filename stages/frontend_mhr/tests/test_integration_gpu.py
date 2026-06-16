"""End-to-end integration test — real model on real CUDA GPU.

Skipped automatically when CUDA or the checkpoint directory is unavailable, so
CI runs without a GPU stay green. Run locally with::

    pytest -m gpu stages/frontend_mhr/tests/test_integration_gpu.py -v -s

Override the checkpoint location via the ``SAM3DBODY_CHECKPOINT_DIR``
environment variable; default = the FastSAM3DToOpenSim production layout.

What it verifies on a real inference:
- A ``BodyEstimate`` comes out (the model didn't just crash).
- Shapes match the topology constants (mesh, skeleton, keypoints, capabilities).
- The ``BodyEstimate`` round-trips through the contracts IO byte-identical.

A synthetic 480x640 noise frame + a manually-provided bounding box is enough:
we are not testing model accuracy here, only the full pipeline glue. The
fingerprint test in ``test_mhr_topology.py`` already guards the rig identity.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


_DEFAULT_CKPT_DIR = Path(
    "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3"
)


def _find_checkpoint_dir() -> Path | None:
    explicit = os.environ.get("SAM3DBODY_CHECKPOINT_DIR")
    if explicit:
        p = Path(explicit)
        return p if (p / "model.ckpt").is_file() else None
    if (_DEFAULT_CKPT_DIR / "model.ckpt").is_file():
        return _DEFAULT_CKPT_DIR
    return None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.gpu
def test_real_inference_produces_conformant_body_estimate(tmp_path):
    if not _cuda_available():
        pytest.skip("CUDA not available")
    ckpt_dir = _find_checkpoint_dir()
    if ckpt_dir is None:
        pytest.skip(
            "checkpoints not found; set SAM3DBODY_CHECKPOINT_DIR or place "
            f"them at {_DEFAULT_CKPT_DIR}"
        )

    # Heavy imports localised to the test so module import is cheap.
    from mesh2sim.contracts import BodyEstimate, load, save
    from mesh2sim_frontend_mhr import (
        MHR_N_JOINTS,
        MHR_N_KEYPOINTS_2D,
        MHR_N_VERTICES,
        MHR_TOPOLOGY_ID,
        FastSAM3DBodyEstimator,
    )

    # Synthetic frame + manual bbox: we skip the human detector entirely.
    rng = np.random.default_rng(0)
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    bbox = np.array([[100.0, 50.0, 540.0, 470.0]], dtype=np.float32)

    est = FastSAM3DBodyEstimator.from_pretrained(
        checkpoint_dir=ckpt_dir,
        device="cuda",
        main_subject_only=True,
        process_one_image_kwargs={
            "bboxes": bbox,
            "inference_type": "body",  # skip the hand decoder (no hand bboxes available)
        },
    )

    results = est.estimate_frame(frame, frame_id=42, timestamp=1.5)
    assert len(results) == 1, "main-subject mode must emit exactly one BodyEstimate"
    be: BodyEstimate = results[0]

    # ---- Top-level fields --------------------------------------------------
    assert be.frame_id == 42
    assert be.view_id == "mono"
    assert be.timestamp == 1.5
    assert be.frame_shape == (480, 640)
    assert be.estimator_id.startswith("fast-sam-3d-body@")

    # ---- Mesh --------------------------------------------------------------
    assert be.mesh is not None
    assert be.mesh.topology_id == MHR_TOPOLOGY_ID
    assert be.mesh.vertices.shape == (MHR_N_VERTICES, 3)
    assert be.mesh.vertices.dtype == np.float32
    assert np.isfinite(be.mesh.vertices).all(), "vertices contain NaN/Inf"

    # ---- Skeleton ----------------------------------------------------------
    assert be.skeleton_state is not None
    assert be.skeleton_state.joint_positions.shape == (MHR_N_JOINTS, 3)
    assert be.skeleton_state.joint_orientations.shape == (MHR_N_JOINTS, 3, 3)
    assert len(be.skeleton_state.joint_names) == MHR_N_JOINTS
    assert np.isfinite(be.skeleton_state.joint_positions).all()
    assert np.isfinite(be.skeleton_state.joint_orientations).all()

    # ---- Keypoints 2D ------------------------------------------------------
    assert be.keypoints_2d is not None
    assert be.keypoints_2d.xy.shape == (MHR_N_KEYPOINTS_2D, 2)
    assert be.keypoints_2d.confidence.shape == (MHR_N_KEYPOINTS_2D,)
    assert len(be.keypoints_2d.names) == MHR_N_KEYPOINTS_2D

    # ---- Capabilities ------------------------------------------------------
    assert be.capabilities.has_mesh is True
    assert be.capabilities.has_skeleton is True
    assert be.capabilities.has_2d_keypoints is True
    assert be.capabilities.has_native_params is True

    # ---- Native params: MHR-opaque only, no pipeline leakage ---------------
    assert be.native_params is not None
    expected = {
        "shape_params", "expr_params", "scale_params", "global_rot",
        "body_pose_params", "hand_pose_params", "pred_pose_raw",
        "pred_cam_t", "pred_keypoints_3d", "focal_length",
    }
    leaked = set(be.native_params) - expected
    assert not leaked, f"unexpected keys in native_params: {leaked}"
    forbidden = {"bbox", "mask", "lhand_bbox", "rhand_bbox"}
    assert not (forbidden & set(be.native_params))

    # ---- Round-trip through contracts IO -----------------------------------
    out = save(be, tmp_path / "be")
    back = load(BodyEstimate, out)
    assert back.mesh is not None
    assert back.mesh.vertices.dtype == np.float32
    assert np.array_equal(back.mesh.vertices, be.mesh.vertices)
    assert np.array_equal(
        back.skeleton_state.joint_positions, be.skeleton_state.joint_positions
    )
    assert np.array_equal(
        back.skeleton_state.joint_orientations, be.skeleton_state.joint_orientations
    )

    # ---- Pretty diagnostic print (visible with pytest -s) ------------------
    print()
    print("=== Real-inference BodyEstimate shapes ===")
    print(f"  estimator_id                  : {be.estimator_id}")
    print(f"  frame_shape                   : {be.frame_shape}")
    print(f"  mesh.topology_id              : {be.mesh.topology_id}")
    print(f"  mesh.vertices                 : {be.mesh.vertices.shape} {be.mesh.vertices.dtype}")
    print(f"  skeleton.joint_positions      : {be.skeleton_state.joint_positions.shape} {be.skeleton_state.joint_positions.dtype}")
    print(f"  skeleton.joint_orientations   : {be.skeleton_state.joint_orientations.shape} {be.skeleton_state.joint_orientations.dtype}")
    print(f"  keypoints_2d.xy               : {be.keypoints_2d.xy.shape} {be.keypoints_2d.xy.dtype}")
    print(f"  native_params keys            : {sorted(be.native_params)}")
    for k, v in sorted(be.native_params.items()):
        if hasattr(v, "shape"):
            print(f"    {k:<25s}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"    {k:<25s}: {v!r}")
    print(f"  capabilities                  : {dict(be.capabilities)}")
    print(f"  round-trip via save/load OK   : True")
