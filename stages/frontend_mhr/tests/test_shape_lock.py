"""T4 shape-lock unit tests — no GPU, no real rig.

- Phase-1 aggregation tests use plain numpy.
- Phase-2 regeneration is exercised against a ``FakeRig`` that records the inputs
  it received, so we can assert the 204-vector is assembled exactly per the
  documented layout (CLAUDE.md "Paramètres MHR et chaîne de scaling").

The full GPU integration test (real rig + geometric assertions on bone lengths)
lives in ``test_shape_lock_gpu.py`` and is marked ``gpu``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mesh2sim.contracts import (
    BodyEstimate,
    Capabilities,
    Keypoints2D,
    MeshData,
    ShapeDescriptor,
    ShapeRepresentation,
    SkeletonState,
)
from mesh2sim_frontend_mhr import (
    SHAPE_LOCK_VERSION,
    ShapeLockRegenerator,
    aggregate_shape,
)
from mesh2sim_frontend_mhr.mhr_topology import MHR_N_JOINTS, MHR_N_VERTICES, MHR_TOPOLOGY_ID

# ---------------------------------------------------------------------------
# Synthetic BodyEstimate factory (no GPU, no rig needed)
# ---------------------------------------------------------------------------


def _make_be(
    *,
    frame_id: int,
    shape_params: np.ndarray,
    scale_params: np.ndarray,
    expr_params: np.ndarray,
    body_pose_params: np.ndarray,
    global_rot: np.ndarray,
    pred_cam_t: np.ndarray,
    hand_pose_params: np.ndarray | None = None,
) -> BodyEstimate:
    """Build a minimal but contract-valid BodyEstimate with the native_params T4 reads."""
    native = {
        "shape_params": np.asarray(shape_params, dtype=np.float32),
        "scale_params": np.asarray(scale_params, dtype=np.float32),
        "expr_params": np.asarray(expr_params, dtype=np.float32),
        "body_pose_params": np.asarray(body_pose_params, dtype=np.float32),
        "global_rot": np.asarray(global_rot, dtype=np.float32),
        "pred_cam_t": np.asarray(pred_cam_t, dtype=np.float32),
        "hand_pose_params": np.asarray(
            hand_pose_params if hand_pose_params is not None else np.zeros(108),
            dtype=np.float32,
        ),
    }
    return BodyEstimate(
        estimator_id="fake@v0",
        frame_id=frame_id,
        view_id="mono",
        timestamp=float(frame_id) / 30.0,
        capabilities=Capabilities(
            has_mesh=True,
            has_skeleton=True,
            has_2d_keypoints=True,
            has_native_params=True,
        ),
        native_params=native,
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
        keypoints_2d=Keypoints2D(
            names=[f"k{i:02d}" for i in range(70)],
            xy=np.zeros((70, 2), dtype=np.float32),
            confidence=np.ones(70, dtype=np.float32),
        ),
        camera=None,
        frame_shape=(480, 640),
    )


# ---------------------------------------------------------------------------
# Phase 1: aggregation (pure numpy)
# ---------------------------------------------------------------------------


def test_aggregate_idempotent_on_constant_sequence():
    """N frames with identical shape params → median equals each frame, byte-equal."""
    rng = np.random.default_rng(0)
    shape = rng.standard_normal(45).astype(np.float32)
    scale = rng.standard_normal(28).astype(np.float32)
    expr = rng.standard_normal(72).astype(np.float32)
    # body_pose differs per frame (= different pose), but the 6 shape-mode
    # indices (124..129) are held constant so the locked aggregate matches.
    shape_modes = rng.standard_normal(6).astype(np.float32)
    bps = []
    for _ in range(5):
        bp = rng.standard_normal(133).astype(np.float32)
        bp[124:130] = shape_modes
        bps.append(bp)
    seq = [
        _make_be(
            frame_id=i,
            shape_params=shape,
            scale_params=scale,
            expr_params=expr,
            body_pose_params=bps[i],
            global_rot=rng.standard_normal(3).astype(np.float32),
            pred_cam_t=rng.standard_normal(3).astype(np.float32),
        )
        for i in range(5)
    ]
    locked = aggregate_shape(seq)
    assert isinstance(locked, ShapeDescriptor)
    assert locked.representation == ShapeRepresentation.opaque
    assert locked.source_model == MHR_TOPOLOGY_ID
    np.testing.assert_array_equal(locked.data["shape_params"], shape)
    np.testing.assert_array_equal(locked.data["scale_params"], scale)
    np.testing.assert_array_equal(locked.data["expr_params"], expr)
    # The 6 shape modes hidden in body_pose are also aggregated
    np.testing.assert_array_equal(locked.data["body_pose_shape_modes"], shape_modes)


def test_aggregate_median_robust_to_one_outlier():
    """9 frames around value v, 1 outlier at 100v → median ≈ v, NOT pulled by the outlier."""
    v = np.full(45, 0.1, dtype=np.float32)
    outlier = np.full(45, 100.0, dtype=np.float32)
    seq = [
        _make_be(
            frame_id=i,
            shape_params=v if i < 9 else outlier,
            scale_params=np.zeros(28, dtype=np.float32),
            expr_params=np.zeros(72, dtype=np.float32),
            body_pose_params=np.zeros(133, dtype=np.float32),
            global_rot=np.zeros(3, dtype=np.float32),
            pred_cam_t=np.zeros(3, dtype=np.float32),
        )
        for i in range(10)
    ]
    locked = aggregate_shape(seq)
    # Median over 10 values, 9 at v, 1 at 100v → median = v (between sample 4 and 5, both v)
    np.testing.assert_allclose(locked.data["shape_params"], v, atol=1e-6)
    # By contrast, the mean would be (9*v + 100v) / 10 = (0.9 + 10) / 10 = 1.09 — far from v.


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        aggregate_shape([])


def test_aggregate_rejects_missing_shape_key():
    """A BodyEstimate without shape_params in native_params fails loudly."""
    be = _make_be(
        frame_id=0,
        shape_params=np.zeros(45, dtype=np.float32),
        scale_params=np.zeros(28, dtype=np.float32),
        expr_params=np.zeros(72, dtype=np.float32),
        body_pose_params=np.zeros(133, dtype=np.float32),
        global_rot=np.zeros(3, dtype=np.float32),
        pred_cam_t=np.zeros(3, dtype=np.float32),
    )
    # Strip shape_params from native_params
    bad_np = dict(be.native_params)
    del bad_np["shape_params"]
    be = be.model_copy(update={"native_params": bad_np})
    with pytest.raises(KeyError, match="shape_params"):
        aggregate_shape([be])


# ---------------------------------------------------------------------------
# Phase 2: regeneration (FakeRig)
# ---------------------------------------------------------------------------


class _FakeRig:
    """Stand-in for the TorchScript MHR rig. Records the forward inputs so the
    test can assert the 204-vec layout. Returns deterministic outputs whose
    shape matches the real rig."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
        apply_correctives: bool,
    ):
        self.calls.append(
            {
                "identity_coeffs": identity_coeffs.cpu().numpy().copy(),
                "model_parameters": model_parameters.cpu().numpy().copy(),
                "face_expr_coeffs": face_expr_coeffs.cpu().numpy().copy(),
                "apply_correctives": apply_correctives,
            }
        )
        batch = identity_coeffs.shape[0]
        # Verts: deterministic vertex i = (i, 2i, 3i) shifted by global_trans
        idx = torch.arange(
            MHR_N_VERTICES, device=identity_coeffs.device, dtype=identity_coeffs.dtype
        )
        verts_one = torch.stack([idx, 2 * idx, 3 * idx], dim=-1)  # (V, 3)
        verts = verts_one.unsqueeze(0).expand(batch, -1, -1).clone()
        # Add global_trans (in same units) to make the meshes per-frame distinct
        global_trans = model_parameters[:, 0:3] / 10.0  # undo the ×10 the rig wrapper applies
        verts = verts + global_trans.unsqueeze(1)
        # Skel: (B, J, 8) — identity quat + scale=1
        skel = torch.zeros(
            batch, MHR_N_JOINTS, 8, device=identity_coeffs.device, dtype=identity_coeffs.dtype
        )
        skel[:, :, 6] = 1.0  # qw = 1 (identity quaternion)
        skel[:, :, 7] = 1.0  # scale = 1
        # Add a per-joint offset that depends on global_rot so we can prove pose was applied
        skel[:, :, 0] = global_trans[:, 0:1]
        skel[:, :, 1] = global_trans[:, 1:2]
        skel[:, :, 2] = global_trans[:, 2:3]
        return verts, skel


def _make_regenerator() -> tuple[ShapeLockRegenerator, _FakeRig]:
    fake_rig = _FakeRig()
    # Synthetic PCA constants. The numeric values don't matter for the layout test.
    scale_mean = torch.full((68,), 0.7, dtype=torch.float32)
    scale_comps = torch.eye(28, 68, dtype=torch.float32)  # (28, 68); first 28 cols of identity
    return (
        ShapeLockRegenerator(
            rig=fake_rig,
            scale_mean=scale_mean,
            scale_comps=scale_comps,
            device="cpu",
        ),
        fake_rig,
    )


def test_regenerate_constructs_correct_204_layout():
    """Build a BodyEstimate with known per-frame pose + locked shape, feed it to the
    FakeRig, assert the 204-vec layout matches CLAUDE.md."""
    regen, fake_rig = _make_regenerator()

    shape_locked = np.full(45, 0.11, dtype=np.float32)
    scale_locked = np.zeros(28, dtype=np.float32)
    scale_locked[3] = 1.0  # the 4th PCA component is +1
    expr_locked = np.full(72, 0.22, dtype=np.float32)
    bp_shape_modes_locked = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": shape_locked,
            "scale_params": scale_locked,
            "expr_params": expr_locked,
            "body_pose_shape_modes": bp_shape_modes_locked,
        },
        source_model=MHR_TOPOLOGY_ID,
    )

    body_pose = np.zeros(133, dtype=np.float32)
    body_pose[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]  # leading 5 entries so we can verify slicing [:130]
    body_pose[124:130] = 0.0  # original shape modes — will be overwritten by locked values
    global_rot = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    pred_cam_t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    be = _make_be(
        frame_id=0,
        shape_params=np.full(45, 999.0, dtype=np.float32),  # ignored — locked is used
        scale_params=np.full(28, 999.0, dtype=np.float32),  # ignored
        expr_params=np.full(72, 999.0, dtype=np.float32),  # ignored
        body_pose_params=body_pose,
        global_rot=global_rot,
        pred_cam_t=pred_cam_t,
    )

    regen.regenerate_batch([be], locked)

    # Inspect what the FakeRig saw
    assert len(fake_rig.calls) == 1
    call = fake_rig.calls[0]
    mp = call["model_parameters"]  # (1, 204)
    assert mp.shape == (1, 204)
    np.testing.assert_allclose(mp[0, 0:3], pred_cam_t * 10.0, atol=1e-6)  # global_trans × 10
    np.testing.assert_allclose(mp[0, 3:6], global_rot, atol=1e-6)
    # body_pose passes through at positions 0..123 (and 130..132 would be too, but
    # the rig slice cuts those off). Positions 124..129 are OVERWRITTEN by the
    # locked shape modes.
    np.testing.assert_allclose(mp[0, 6:130], body_pose[:124], atol=1e-6)  # per-frame body pose
    np.testing.assert_allclose(
        mp[0, 130:136], bp_shape_modes_locked, atol=1e-6
    )  # locked shape modes
    # Scales = scale_mean + scale_params @ scale_comps. With scale_locked[3]=1 and
    # scale_comps = identity(28, 68), the 4th row of identity is at position 3, so
    # scale_params @ scale_comps gives [0,0,0,1,0,...] (68-long), thus scales[3] = 0.7+1.
    expected_scales = np.full(68, 0.7, dtype=np.float32)
    expected_scales[3] += 1.0
    np.testing.assert_allclose(mp[0, 136:204], expected_scales, atol=1e-6)
    # Identity coeffs (45) = locked shape_params
    np.testing.assert_allclose(call["identity_coeffs"][0], shape_locked, atol=1e-6)
    # Face expression (72) = locked expr_params
    np.testing.assert_allclose(call["face_expr_coeffs"][0], expr_locked, atol=1e-6)
    # apply_correctives must be False (faster, and we don't have a corrective signal)
    assert call["apply_correctives"] is False


def test_regenerate_locked_shape_byte_equal_across_frames():
    """Three frames with DIFFERENT poses but SAME locked shape → the locked-shape
    parts of each regenerated BodyEstimate's native_params must be byte-equal."""
    regen, _ = _make_regenerator()

    shape_locked = np.array([1.0] * 45, dtype=np.float32)
    scale_locked = np.array([2.0] * 28, dtype=np.float32)
    expr_locked = np.array([3.0] * 72, dtype=np.float32)
    bp_shape_modes_locked = np.array([4.0] * 6, dtype=np.float32)
    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": shape_locked,
            "scale_params": scale_locked,
            "expr_params": expr_locked,
            "body_pose_shape_modes": bp_shape_modes_locked,
        },
        source_model=MHR_TOPOLOGY_ID,
    )

    rng = np.random.default_rng(42)
    seq = [
        _make_be(
            frame_id=i,
            shape_params=rng.standard_normal(45).astype(np.float32),  # ignored
            scale_params=rng.standard_normal(28).astype(np.float32),  # ignored
            expr_params=rng.standard_normal(72).astype(np.float32),  # ignored
            body_pose_params=rng.standard_normal(133).astype(np.float32),
            global_rot=rng.standard_normal(3).astype(np.float32),
            pred_cam_t=rng.standard_normal(3).astype(np.float32),
        )
        for i in range(3)
    ]
    regenerated = regen.regenerate_batch(seq, locked)
    assert len(regenerated) == 3

    for i in range(1, 3):
        for key in ("shape_params", "scale_params", "expr_params"):
            np.testing.assert_array_equal(
                regenerated[i].native_params[key],
                regenerated[0].native_params[key],
                err_msg=f"{key} differs between frame 0 and frame {i}",
            )
        # body_pose_params positions 124..129 hold the 6 shape modes — they must
        # be locked too. Positions 0..123 and 130..132 are pose and differ per frame.
        np.testing.assert_array_equal(
            regenerated[i].native_params["body_pose_params"][124:130],
            regenerated[0].native_params["body_pose_params"][124:130],
            err_msg=f"body_pose shape modes (124..129) differ between frame 0 and frame {i}",
        )


def test_regenerate_pose_preserved_per_frame():
    """The pose parts of native_params survive verbatim through regeneration."""
    regen, _ = _make_regenerator()

    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": np.zeros(45, dtype=np.float32),
            "scale_params": np.zeros(28, dtype=np.float32),
            "expr_params": np.zeros(72, dtype=np.float32),
            "body_pose_shape_modes": np.zeros(6, dtype=np.float32),
        },
        source_model=MHR_TOPOLOGY_ID,
    )

    rng = np.random.default_rng(7)
    bp = rng.standard_normal(133).astype(np.float32)
    gr = rng.standard_normal(3).astype(np.float32)
    ct = rng.standard_normal(3).astype(np.float32)
    hp = rng.standard_normal(108).astype(np.float32)
    be = _make_be(
        frame_id=42,
        shape_params=np.zeros(45, dtype=np.float32),
        scale_params=np.zeros(28, dtype=np.float32),
        expr_params=np.zeros(72, dtype=np.float32),
        body_pose_params=bp,
        global_rot=gr,
        pred_cam_t=ct,
        hand_pose_params=hp,
    )

    regen_out = regen.regenerate_batch([be], locked)
    assert len(regen_out) == 1
    out_np = regen_out[0].native_params
    # Pose passthrough at body_pose[0:124] and 130:133 — preserved verbatim.
    # Shape modes at 124..129 are OVERWRITTEN with the locked values (zeros here).
    np.testing.assert_array_equal(out_np["body_pose_params"][:124], bp[:124])
    np.testing.assert_array_equal(
        out_np["body_pose_params"][124:130], np.zeros(6, dtype=np.float32)
    )
    np.testing.assert_array_equal(out_np["body_pose_params"][130:], bp[130:])
    np.testing.assert_array_equal(out_np["global_rot"], gr)
    np.testing.assert_array_equal(out_np["pred_cam_t"], ct)
    np.testing.assert_array_equal(out_np["hand_pose_params"], hp)


def test_regenerate_drops_keypoints_2d_and_3d():
    """After shape-lock, the camera_head's 2D projections are stale → must be dropped."""
    regen, _ = _make_regenerator()
    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": np.zeros(45, dtype=np.float32),
            "scale_params": np.zeros(28, dtype=np.float32),
            "expr_params": np.zeros(72, dtype=np.float32),
            "body_pose_shape_modes": np.zeros(6, dtype=np.float32),
        },
        source_model=MHR_TOPOLOGY_ID,
    )
    be = _make_be(
        frame_id=0,
        shape_params=np.zeros(45, dtype=np.float32),
        scale_params=np.zeros(28, dtype=np.float32),
        expr_params=np.zeros(72, dtype=np.float32),
        body_pose_params=np.zeros(133, dtype=np.float32),
        global_rot=np.zeros(3, dtype=np.float32),
        pred_cam_t=np.zeros(3, dtype=np.float32),
    )
    out = regen.regenerate_batch([be], locked)[0]
    assert out.keypoints_2d is None
    assert out.capabilities.has_2d_keypoints is False
    assert "pred_keypoints_3d" not in (out.native_params or {})


def test_regenerate_estimator_id_carries_shape_lock_tag():
    """Provenance: the regenerated estimator_id must carry the shape-lock version tag."""
    regen, _ = _make_regenerator()
    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": np.zeros(45, dtype=np.float32),
            "scale_params": np.zeros(28, dtype=np.float32),
            "expr_params": np.zeros(72, dtype=np.float32),
            "body_pose_shape_modes": np.zeros(6, dtype=np.float32),
        },
        source_model=MHR_TOPOLOGY_ID,
    )
    be = _make_be(
        frame_id=0,
        shape_params=np.zeros(45, dtype=np.float32),
        scale_params=np.zeros(28, dtype=np.float32),
        expr_params=np.zeros(72, dtype=np.float32),
        body_pose_params=np.zeros(133, dtype=np.float32),
        global_rot=np.zeros(3, dtype=np.float32),
        pred_cam_t=np.zeros(3, dtype=np.float32),
    )
    out = regen.regenerate_batch([be], locked)[0]
    assert out.estimator_id is not None
    assert out.estimator_id.endswith("+" + SHAPE_LOCK_VERSION)
    assert "fake@v0" in out.estimator_id


def test_regenerate_empty_returns_empty():
    regen, _ = _make_regenerator()
    locked = ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": np.zeros(45, dtype=np.float32),
            "scale_params": np.zeros(28, dtype=np.float32),
            "expr_params": np.zeros(72, dtype=np.float32),
            "body_pose_shape_modes": np.zeros(6, dtype=np.float32),
        },
        source_model=MHR_TOPOLOGY_ID,
    )
    assert regen.regenerate_batch([], locked) == []


def test_regenerator_rejects_wrong_pca_shapes():
    fake_rig = _FakeRig()
    with pytest.raises(ValueError, match=r"scale_mean must be \(68,\)"):
        ShapeLockRegenerator(
            rig=fake_rig,
            scale_mean=torch.zeros(67),
            scale_comps=torch.zeros(28, 68),
            device="cpu",
        )
    with pytest.raises(ValueError, match=r"scale_comps must be \(28, 68\)"):
        ShapeLockRegenerator(
            rig=fake_rig,
            scale_mean=torch.zeros(68),
            scale_comps=torch.zeros(27, 68),
            device="cpu",
        )
