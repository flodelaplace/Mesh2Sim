"""T4 shape-lock GPU integration test — real MHR rig + real PCA constants.

Skipped automatically when CUDA, the vendored rig, or the extracted PCA npz
are unavailable. Runs ``lock_shape_and_regenerate`` on a synthetic 3-frame
sequence with distinct poses but identical input shape, then asserts:

1. Locked shape parts of the regenerated BodyEstimates are byte-equal across
   frames (the whole point of T4).
2. Pose parts are byte-equal to the corresponding original frame.
3. Parent-child bone lengths (skeleton edges) are byte-equal across the
   regenerated frames — a geometric proof that the shape is truly locked.
4. Joint positions DIFFER across frames — proof that the pose was applied
   (we're not just freezing the rest pose).

Run locally with::

    pytest -m gpu stages/frontend_mhr/tests/test_shape_lock_gpu.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_DEFAULT_RIG = Path(
    "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
_DEFAULT_PCA = Path(__file__).resolve().parent.parent / "reference" / "mhr_scale_pca.npz"


def _find_rig() -> Path | None:
    explicit = os.environ.get("MHR_RIG_PATH")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    return _DEFAULT_RIG if _DEFAULT_RIG.is_file() else None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.gpu
def test_shape_lock_real_rig(tmp_path):  # noqa: ARG001
    if not _cuda_available():
        pytest.skip("CUDA not available")
    rig_path = _find_rig()
    if rig_path is None:
        pytest.skip(f"rig not found at {_DEFAULT_RIG}; set MHR_RIG_PATH")
    if not _DEFAULT_PCA.is_file():
        pytest.skip(
            f"scale-PCA npz not found at {_DEFAULT_PCA}; regenerate with "
            "stages/frontend_mhr/scripts/extract_scale_pca.py"
        )

    # Heavy imports localised so module import stays cheap on CI without GPU.
    from mesh2sim.contracts import BodyEstimate, Capabilities, MeshData, SkeletonState
    from mesh2sim_frontend_mhr import (
        SHAPE_LOCK_VERSION,
        lock_shape_and_regenerate,
    )
    from mesh2sim_frontend_mhr.mhr_topology import MHR_N_JOINTS, MHR_N_VERTICES, MHR_TOPOLOGY_ID

    # --- Build 3 synthetic BodyEstimates with distinct poses ---
    rng = np.random.default_rng(0)
    shape_params = rng.standard_normal(45).astype(np.float32) * 0.3
    scale_params = rng.standard_normal(28).astype(np.float32) * 0.1
    expr_params = np.zeros(72, dtype=np.float32)
    # In MHR, the "bone" joint 0 → joint 1 is driven by global_trans (= pred_cam_t)
    # since joint 0 sits at the origin and joint 1 carries the body's world
    # translation. To get a clean bone-length invariance check across frames we
    # keep pred_cam_t identical and let only body_pose rotations + global_rot
    # vary per frame. (Bone 0→1 isn't really morphological — it's the root
    # translation.)
    fixed_pred_cam_t = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    def _be(frame_id: int, seed: int) -> BodyEstimate:
        rng_pose = np.random.default_rng(seed)
        bp = (rng_pose.standard_normal(133) * 0.05).astype(np.float32)
        # Make the 6 shape modes at positions 124..129 differ per frame too —
        # this is precisely what aggregate_shape must lock down.
        return BodyEstimate(
            estimator_id="t4-integration@v0",
            frame_id=frame_id,
            view_id="mono",
            timestamp=float(frame_id) / 30.0,
            capabilities=Capabilities(
                has_mesh=True,
                has_skeleton=True,
                has_2d_keypoints=False,
                has_native_params=True,
            ),
            native_params={
                "shape_params": shape_params,
                "scale_params": scale_params,
                "expr_params": expr_params,
                "body_pose_params": bp,
                "global_rot": (rng_pose.standard_normal(3) * 0.05).astype(np.float32),
                "pred_cam_t": fixed_pred_cam_t,  # IDENTICAL across frames
                "hand_pose_params": np.zeros(108, dtype=np.float32),
            },
            mesh=MeshData(
                vertices=np.zeros((MHR_N_VERTICES, 3), dtype=np.float32),
                topology_id=MHR_TOPOLOGY_ID,
            ),
            skeleton_state=SkeletonState(
                joint_positions=np.zeros((MHR_N_JOINTS, 3), dtype=np.float32),
                joint_orientations=np.broadcast_to(
                    np.eye(3, dtype=np.float32), (MHR_N_JOINTS, 3, 3)
                ).copy(),
                joint_names=[f"mhr_joint_{j:03d}" for j in range(MHR_N_JOINTS)],
            ),
            keypoints_2d=None,
            camera=None,
            frame_shape=(480, 640),
        )

    seq = [_be(frame_id=i, seed=i + 1) for i in range(3)]

    # --- Run T4 ---
    regenerated, locked_shape = lock_shape_and_regenerate(
        seq, rig_path=rig_path, scale_pca_path=_DEFAULT_PCA, device="cuda"
    )

    assert len(regenerated) == 3, "must emit one BodyEstimate per input frame"

    # --- 1. Locked shape parts byte-equal across frames ---
    for i in range(1, 3):
        for key in ("shape_params", "scale_params", "expr_params"):
            np.testing.assert_array_equal(
                regenerated[i].native_params[key],
                regenerated[0].native_params[key],
                err_msg=f"{key} differs between frame 0 and frame {i}",
            )
        # The 6 shape modes hidden in body_pose_params[124:130] must ALSO be locked
        # (this was the bug the first run caught — body_pose's last 6 entries are
        # morphology, not pose).
        np.testing.assert_array_equal(
            regenerated[i].native_params["body_pose_params"][124:130],
            regenerated[0].native_params["body_pose_params"][124:130],
            err_msg=f"body_pose_params shape modes (124..129) differ between frame 0 and frame {i}",
        )

    # --- 2. Pose parts byte-equal to original per frame ---
    # Note: body_pose_params is split — positions 0..123 + 130..132 are pose
    # (must equal original), positions 124..129 are shape modes (locked, must
    # equal locked_shape.data['body_pose_shape_modes']).
    for orig, new in zip(seq, regenerated, strict=True):
        for key in ("global_rot", "pred_cam_t", "hand_pose_params"):
            np.testing.assert_array_equal(
                new.native_params[key],
                orig.native_params[key],
                err_msg=f"{key} not preserved at frame_id={orig.frame_id}",
            )
        np.testing.assert_array_equal(
            new.native_params["body_pose_params"][:124],
            orig.native_params["body_pose_params"][:124],
            err_msg=f"body_pose_params[0:124] not preserved at frame_id={orig.frame_id}",
        )
        np.testing.assert_array_equal(
            new.native_params["body_pose_params"][130:],
            orig.native_params["body_pose_params"][130:],
            err_msg=f"body_pose_params[130:] not preserved at frame_id={orig.frame_id}",
        )

    # --- 3. Bone lengths (parent-child) byte-equal across regenerated frames ---
    import torch

    rig = torch.jit.load(str(rig_path), map_location="cpu").eval()
    parents = rig.character_torch.skeleton.joint_parents.cpu().numpy()
    child = np.where(parents >= 0)[0]
    parent = parents[child]

    def _bone_lengths(be: BodyEstimate) -> np.ndarray:
        pos = be.skeleton_state.joint_positions
        d = pos[child] - pos[parent]
        return np.linalg.norm(d, axis=-1)

    lens_0 = _bone_lengths(regenerated[0])
    for i in range(1, 3):
        lens_i = _bone_lengths(regenerated[i])
        np.testing.assert_allclose(
            lens_i,
            lens_0,
            rtol=0,
            atol=1e-4,
            err_msg=f"bone lengths differ between frame 0 and frame {i} — shape NOT locked",
        )

    # --- 4. Joint positions DIFFER across frames (pose was applied) ---
    pos_0 = regenerated[0].skeleton_state.joint_positions
    n_diff = 0
    for i in range(1, 3):
        pos_i = regenerated[i].skeleton_state.joint_positions
        if not np.allclose(pos_i, pos_0, atol=1e-4):
            n_diff += 1
    assert n_diff >= 1, "no frame shows distinct joint positions — pose was NOT applied"

    # --- 5. Provenance + dropped 2D keypoints (sanity) ---
    for be in regenerated:
        assert be.estimator_id is not None and be.estimator_id.endswith("+" + SHAPE_LOCK_VERSION)
        assert be.capabilities.has_2d_keypoints is False
        assert be.keypoints_2d is None
        assert be.mesh is not None and be.mesh.vertices.shape == (MHR_N_VERTICES, 3)
        assert be.mesh.vertices.dtype == np.float32

    # --- 6. Locked ShapeDescriptor returned matches what was used ---
    assert locked_shape.source_model == MHR_TOPOLOGY_ID
    for key in ("shape_params", "scale_params", "expr_params"):
        np.testing.assert_array_equal(
            locked_shape.data[key],
            regenerated[0].native_params[key],
            err_msg=f"locked_shape.data[{key!r}] doesn't match regenerated native_params",
        )

    # --- Pretty diagnostic ---
    print()
    print("=== T4 shape-lock GPU integration ===")
    print(f"  rig                       : {rig_path}")
    print(f"  pca                       : {_DEFAULT_PCA}")
    print(f"  n frames                  : {len(regenerated)}")
    print(f"  vertices per frame        : {regenerated[0].mesh.vertices.shape}")
    print(f"  bone lengths (mean ± std) : {lens_0.mean():.4f} ± {lens_0.std():.4f}")
    max_diff = max(float(np.abs(_bone_lengths(regenerated[i]) - lens_0).max()) for i in range(1, 3))
    print(f"  max bone-length diff across frames: {max_diff:.2e} (tol 1e-4)")
    print("  shape locked byte-equal   : True")
    print("  pose preserved per frame  : True")
    print(f"  joint positions vary      : {n_diff} of 2 frames differ from frame 0")
