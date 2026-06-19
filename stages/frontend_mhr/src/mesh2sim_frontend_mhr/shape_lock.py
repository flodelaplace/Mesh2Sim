"""T4 shape-lock — aggregate the subject's MHR shape across an essai and regenerate
each frame's mesh with that locked shape (pose preserved per frame).

Two phases, kept separate so each is testable on its own:

- Phase 1, :func:`aggregate_shape`: pure-numpy median over the time axis of the
  shape-related entries of ``BodyEstimate.native_params`` (``shape_params``,
  ``scale_params``, ``expr_params``). Robust to outlier frames. Returns the locked
  shape as a ``ShapeDescriptor`` (opaque representation, the keys are the MHR
  component names). Same shape descriptor will flow downstream as
  ``AnatomicalTrajectory.shape_descriptor``.

- Phase 2, :class:`ShapeLockRegenerator`: runs the vendored MHR rig forward with
  the locked shape + each frame's pose. Produces a sequence of regenerated
  ``BodyEstimate``: the vertices/joints reflect the new (locked, identical) shape,
  the pose of each frame is preserved by passing through ``global_rot``,
  ``pred_cam_t`` (= ``global_trans``), ``body_pose_params``, ``hand_pose_params``.

Important design notes:

- ``expr_params`` is also locked to the median. The face does NOT affect body
  landmarks (no marker lives on the face mesh), so locking it has zero biomech
  consequence; we lock it anyway for full reproducibility.
- ``keypoints_2d`` of the regenerated ``BodyEstimate`` is **dropped** (set to
  ``None``, ``has_2d_keypoints=False``). SAM 3D Body's ``pred_keypoints_2d`` is the
  perspective projection of the predicted 3D keypoints via the camera_head — it
  reflects the **pre-shape-lock** mesh, so after shape-lock it is stale and would
  mislead T7. Downstream stages that need 2D evidence should re-derive it.
- ``pred_keypoints_3d`` is dropped too, for the same reason (the 3D keypoints
  come from a separate ``keypoint_mapping`` in MHRHead that the bare rig forward
  doesn't expose).
- ``pose-fine-refinement under fixed shape is NOT done here`` — that's T7.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from mesh2sim.contracts import (
    BodyEstimate,
    Capabilities,
    MeshData,
    ShapeDescriptor,
    ShapeRepresentation,
    SkeletonState,
)

from .mhr_topology import MHR_N_JOINTS, MHR_TOPOLOGY_ID

SHAPE_LOCK_VERSION = "shape-lock@0.0.1"
"""Tag suffixed to the ``estimator_id`` of every regenerated BodyEstimate, for traceability."""

# Native-params keys whose values are aggregated across frames into the locked shape.
_SHAPE_KEYS: tuple[str, ...] = ("shape_params", "scale_params", "expr_params")

# The 6 shape modes hidden inside body_pose_params at positions 124..129. Per
# CLAUDE.md "Paramètres MHR et chaîne de scaling", these are PCA shape modes
# encoded as 1-DoF translations within body_pose; they belong to the subject's
# morphology, not the pose. We aggregate them with the same median strategy and
# inject the locked values into each per-frame body_pose at regeneration time.
_BODY_POSE_SHAPE_MODE_SLICE: slice = slice(124, 130)
_BODY_POSE_SHAPE_MODE_N: int = 6
_BODY_POSE_SHAPE_MODE_KEY: str = "body_pose_shape_modes"

# Per-frame pose-related keys that survive the regeneration verbatim.
# NB: body_pose_params is handled specially — positions 0..123 + 130..132 are
# pose (pass-through), positions 124..129 are shape modes (locked).
_POSE_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "global_rot",
    "pred_cam_t",
    "hand_pose_params",
    "pred_pose_raw",
    "focal_length",
)


# ---------------------------------------------------------------------------
# Phase 1 — aggregation
# ---------------------------------------------------------------------------


def aggregate_shape(body_estimates: Sequence[BodyEstimate]) -> ShapeDescriptor:
    """Median-aggregate ``shape_params`` / ``scale_params`` / ``expr_params`` over a sequence.

    Per-element median across the time axis. Robust to outlier frames (a few
    frames with aberrant shape estimates won't pull the median).

    Returns a ``ShapeDescriptor`` with ``representation=opaque``, keyed by the
    three MHR component names. ``source_model`` is set to the MHR topology id
    (`mhr_v1` at the vendored commit).

    Raises:
        ValueError: empty input.
        KeyError: a BodyEstimate is missing one of the required shape keys in
            ``native_params``.
    """
    if len(body_estimates) == 0:
        raise ValueError("cannot aggregate from an empty body_estimate sequence")

    stacks: dict[str, list[np.ndarray]] = {key: [] for key in _SHAPE_KEYS}
    body_pose_shape_modes_stack: list[np.ndarray] = []

    for i, be in enumerate(body_estimates):
        np_ = be.native_params or {}
        for key in _SHAPE_KEYS:
            if key not in np_ or np_[key] is None:
                raise KeyError(
                    f"BodyEstimate at index {i} (frame_id={be.frame_id}) is missing "
                    f"native_params[{key!r}] required for shape-lock aggregation"
                )
            stacks[key].append(np.asarray(np_[key]))
        # The 6 shape modes living inside body_pose_params at positions 124..129
        if "body_pose_params" not in np_ or np_["body_pose_params"] is None:
            raise KeyError(
                f"BodyEstimate at index {i} (frame_id={be.frame_id}) is missing "
                "native_params['body_pose_params'] required for shape-lock aggregation "
                "(positions 124..129 hold the 6 shape modes)"
            )
        bp = np.asarray(np_["body_pose_params"])
        if bp.shape[0] < 130:
            raise ValueError(
                f"BodyEstimate at index {i}: body_pose_params has size {bp.shape[0]} < 130; "
                "cannot extract shape modes at indices 124..129"
            )
        body_pose_shape_modes_stack.append(bp[_BODY_POSE_SHAPE_MODE_SLICE])

    locked: dict[str, np.ndarray] = {
        key: np.median(np.stack(stacks[key], axis=0), axis=0).astype(np.float32)
        for key in _SHAPE_KEYS
    }
    locked[_BODY_POSE_SHAPE_MODE_KEY] = np.median(
        np.stack(body_pose_shape_modes_stack, axis=0), axis=0
    ).astype(np.float32)

    return ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data=locked,
        source_model=MHR_TOPOLOGY_ID,
    )


# ---------------------------------------------------------------------------
# Phase 2 — regeneration via the MHR rig forward
# ---------------------------------------------------------------------------


class ShapeLockRegenerator:
    """Hold the vendored MHR rig + the scale-PCA constants, regenerate posed meshes
    with a locked shape.

    Constructor takes pre-loaded torch objects so tests can inject a fake rig
    without touching disk or the GPU. See :meth:`from_files` for the production
    constructor that loads from the vendored rig + the extracted PCA npz.
    """

    def __init__(
        self,
        rig: torch.jit.ScriptModule,
        scale_mean: torch.Tensor,
        scale_comps: torch.Tensor,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.rig = rig
        # Move PCA constants to device. We cast to float32 to match the rig.
        self.scale_mean = scale_mean.to(self.device).float()  # (68,)
        self.scale_comps = scale_comps.to(self.device).float()  # (28, 68)
        if self.scale_mean.shape != (68,):
            raise ValueError(f"scale_mean must be (68,), got {tuple(self.scale_mean.shape)}")
        if self.scale_comps.shape != (28, 68):
            raise ValueError(f"scale_comps must be (28, 68), got {tuple(self.scale_comps.shape)}")

    @classmethod
    def from_files(
        cls,
        rig_path: str | Path,
        scale_pca_path: str | Path,
        device: str = "cuda",
    ) -> ShapeLockRegenerator:
        """Production constructor: load the vendored rig and the extracted PCA npz."""
        rig_path = Path(rig_path)
        scale_pca_path = Path(scale_pca_path)
        if not rig_path.is_file():
            raise FileNotFoundError(f"MHR rig not found at {rig_path}")
        if not scale_pca_path.is_file():
            raise FileNotFoundError(
                f"scale-PCA npz not found at {scale_pca_path}. "
                "Regenerate with stages/frontend_mhr/scripts/extract_scale_pca.py."
            )
        rig = torch.jit.load(str(rig_path), map_location=device).eval()
        pca = np.load(scale_pca_path, allow_pickle=False)
        return cls(
            rig=rig,
            scale_mean=torch.from_numpy(pca["scale_mean"]),
            scale_comps=torch.from_numpy(pca["scale_comps"]),
            device=device,
        )

    @torch.no_grad()
    def regenerate_batch(
        self,
        body_estimates: Sequence[BodyEstimate],
        locked_shape: ShapeDescriptor,
    ) -> list[BodyEstimate]:
        """Regenerate every frame's mesh with the locked shape + that frame's pose.

        Single batched rig forward (efficient for sequences). Returns a list of
        regenerated BodyEstimate in the same order, with the documented field
        adjustments (locked shape in native_params, dropped keypoints_2d, etc.).
        """
        n_frames = len(body_estimates)
        if n_frames == 0:
            return []

        # --- assemble per-frame pose stacks ---
        global_trans = self._stack(body_estimates, "pred_cam_t", expected_size=3)  # (T, 3)
        global_rot = self._stack(body_estimates, "global_rot", expected_size=3)  # (T, 3)
        body_pose_full = self._stack(
            body_estimates, "body_pose_params", expected_size=None
        )  # (T, 133)
        # The rig consumes only the first 130 entries of body_pose_params.
        body_pose = body_pose_full[:, :130].clone()  # (T, 130) — clone before in-place edit
        # Replace positions 124..129 with the LOCKED shape modes (these are the
        # 6 PCA shape modes hidden in body_pose; per CLAUDE.md they belong to
        # the subject's morphology, not the pose, and must be identical across
        # all regenerated frames for a real shape-lock).
        shape_modes_locked = (
            torch.from_numpy(np.asarray(locked_shape.data[_BODY_POSE_SHAPE_MODE_KEY]))
            .to(self.device)
            .float()
        )  # (6,)
        body_pose[:, _BODY_POSE_SHAPE_MODE_SLICE] = shape_modes_locked[None].expand(n_frames, -1)

        # --- replicate locked shape across the time axis ---
        shape_locked = locked_shape.data["shape_params"]
        scale_locked = locked_shape.data["scale_params"]
        expr_locked = locked_shape.data["expr_params"]
        shape = (
            torch.from_numpy(np.asarray(shape_locked))
            .to(self.device)
            .float()[None]
            .expand(n_frames, -1)
        )
        expr = (
            torch.from_numpy(np.asarray(expr_locked))
            .to(self.device)
            .float()[None]
            .expand(n_frames, -1)
        )
        scale_params = (
            torch.from_numpy(np.asarray(scale_locked)).to(self.device).float()[None]
        )  # (1, 28)
        scales_row = self.scale_mean[None] + scale_params @ self.scale_comps  # (1, 68)
        scales = scales_row.expand(n_frames, -1)  # (T, 68), all rows identical

        # --- build the 204-vec batch (mhr_head._mhr_forward_core layout) ---
        model_params = torch.cat(
            [global_trans * 10.0, global_rot, body_pose, scales], dim=1
        )  # (T, 204)

        # --- single batched rig forward ---
        verts, skel = self.rig(shape, model_params, expr, False)
        # verts shape (T, V, 3), skel shape (T, J, 8) — (tx, ty, tz, qx, qy, qz, qw, scale).

        verts_np = verts.detach().cpu().numpy()  # (T, V, 3)
        joint_pos = skel[:, :, :3].detach().cpu().numpy()  # (T, J, 3)
        joint_quats = skel[:, :, 3:7].detach().cpu().numpy()  # (T, J, 4)
        joint_orient = _quats_xyzw_to_rotmats(joint_quats)  # (T, J, 3, 3)

        # --- assemble regenerated BodyEstimate per frame ---
        return [
            self._build_regenerated_be(
                original=be,
                locked_shape=locked_shape,
                vertices=verts_np[i],
                joint_positions=joint_pos[i],
                joint_orientations=joint_orient[i],
            )
            for i, be in enumerate(body_estimates)
        ]

    # -- private helpers -----------------------------------------------------

    def _stack(
        self,
        body_estimates: Sequence[BodyEstimate],
        key: str,
        *,
        expected_size: int | None,
    ) -> torch.Tensor:
        arrs = []
        for i, be in enumerate(body_estimates):
            np_ = be.native_params or {}
            if key not in np_ or np_[key] is None:
                raise KeyError(
                    f"BodyEstimate at index {i} (frame_id={be.frame_id}) is missing "
                    f"native_params[{key!r}] required for shape-lock regeneration"
                )
            arr = np.asarray(np_[key]).reshape(-1)
            if expected_size is not None and arr.shape[0] != expected_size:
                raise ValueError(
                    f"BodyEstimate at index {i}: native_params[{key!r}] expected "
                    f"size {expected_size}, got {arr.shape[0]}"
                )
            arrs.append(arr)
        stacked = torch.from_numpy(np.stack(arrs, axis=0)).to(self.device).float()
        return stacked

    def _build_regenerated_be(
        self,
        *,
        original: BodyEstimate,
        locked_shape: ShapeDescriptor,
        vertices: np.ndarray,
        joint_positions: np.ndarray,
        joint_orientations: np.ndarray,
    ) -> BodyEstimate:
        # locked shape part — written byte-equal for every frame
        new_native: dict = {key: np.asarray(locked_shape.data[key]) for key in _SHAPE_KEYS}
        # per-frame pose passthrough — straight from the original
        orig_np = original.native_params or {}
        for key in _POSE_PASSTHROUGH_KEYS:
            if key in orig_np and orig_np[key] is not None:
                new_native[key] = orig_np[key]
        # body_pose_params is special: pose at positions 0..123 + 130..132, locked
        # shape modes at 124..129. We write the merged version that reflects what
        # actually went into the rig forward.
        if "body_pose_params" in orig_np and orig_np["body_pose_params"] is not None:
            merged = np.asarray(orig_np["body_pose_params"]).copy()
            shape_modes = np.asarray(locked_shape.data[_BODY_POSE_SHAPE_MODE_KEY])
            merged[_BODY_POSE_SHAPE_MODE_SLICE] = shape_modes
            new_native["body_pose_params"] = merged
        # NB: we deliberately DO NOT carry over pred_keypoints_3d nor any
        # 2D-projection cache — those reflect the pre-shape-lock mesh.

        # Joint names: preserve the original (positional MHR names), or fall back to a
        # synthetic positional list if the original was absent.
        if original.skeleton_state is not None:
            joint_names = list(original.skeleton_state.joint_names)
        else:
            joint_names = [f"mhr_joint_{j:03d}" for j in range(MHR_N_JOINTS)]

        return BodyEstimate(
            estimator_id=(original.estimator_id or "unknown") + "+" + SHAPE_LOCK_VERSION,
            frame_id=original.frame_id,
            view_id=original.view_id,
            timestamp=original.timestamp,
            capabilities=Capabilities(
                has_mesh=True,
                has_skeleton=True,
                has_2d_keypoints=False,  # dropped (stale projection of pre-lock mesh)
                has_native_params=True,
            ),
            native_params=new_native,
            mesh=MeshData(
                vertices=vertices,
                topology_id=MHR_TOPOLOGY_ID,
                faces=None,
            ),
            skeleton_state=SkeletonState(
                joint_positions=joint_positions,
                joint_orientations=joint_orientations,
                joint_names=joint_names,
            ),
            keypoints_2d=None,
            camera=original.camera,
            frame_shape=original.frame_shape,
        )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def lock_shape_and_regenerate(
    body_estimates: Sequence[BodyEstimate],
    rig_path: str | Path,
    scale_pca_path: str | Path,
    *,
    device: str = "cuda",
) -> tuple[list[BodyEstimate], ShapeDescriptor]:
    """One-shot: aggregate the locked shape and regenerate every frame's mesh.

    Convenience wrapper around :func:`aggregate_shape` +
    :class:`ShapeLockRegenerator.from_files`. For tests with an injected mock
    rig, use the lower-level building blocks directly.
    """
    locked = aggregate_shape(body_estimates)
    regen = ShapeLockRegenerator.from_files(rig_path, scale_pca_path, device=device)
    return regen.regenerate_batch(body_estimates, locked), locked


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix helper
# ---------------------------------------------------------------------------


def _quats_xyzw_to_rotmats(q: np.ndarray) -> np.ndarray:
    """Convert ``(x, y, z, w)`` quaternions to rotation matrices ``(..., 3, 3)``.

    Matches the convention used by the MHR rig's skel_state output (roma's
    ``(x, y, z, w)`` order — confirmed by reading the vendored
    ``mhr_head._fast_quat_to_rotmat`` and reproducing the same algebra here so
    we don't depend on roma at this layer.
    """
    q = np.asarray(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )
    return r.astype(q.dtype, copy=False)


__all__ = [
    "SHAPE_LOCK_VERSION",
    "ShapeLockRegenerator",
    "aggregate_shape",
    "lock_shape_and_regenerate",
]
