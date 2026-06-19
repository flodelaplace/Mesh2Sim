"""Adapter: ``sam_3d_body.SAM3DBodyEstimator.process_one_image`` per-person dict
→ ``mesh2sim.contracts.BodyEstimate``.

The adapter is **pure**: no IO, no model, no GPU dependency. It receives a
plain dict of numpy arrays (the format used by the vendored estimator after
``recursive_to(out, "numpy")``) and returns a fully-validated ``BodyEstimate``.

What lives where:
- ``mesh``               ← ``pred_vertices``               (V, 3) float32
- ``skeleton_state``     ← ``pred_joint_coords``           (127, 3)
                          ``pred_global_rots``             (127, 3, 3)
                          ``MHR_JOINT_NAMES``              (synthetic — see below)
- ``keypoints_2d``       ← ``pred_keypoints_2d``           (70, 2)
                          names from vendored ``sam_3d_body.metadata.mhr70``
- ``native_params``      ← the MHR-opaque parameters strictly: ``shape_params``,
                          ``expr_params``, ``scale_params``, ``global_rot``,
                          ``body_pose_params``, ``hand_pose_params``,
                          ``pred_pose_raw``, plus camera-frame artefacts
                          (``pred_cam_t``, ``pred_keypoints_3d``,
                          ``focal_length``).
- ``camera``             ← passed in by the caller (if a calibration is known)
- ``frame_shape``        ← passed in by the caller (H, W of the source frame)

Things deliberately NOT placed in ``BodyEstimate``:
- ``bbox`` / ``mask`` / detection scores → pipeline metadata, not estimator
  output. The contract does not expose a slot for this. See README section
  "Contract debt".
- ``lhand_bbox`` / ``rhand_bbox`` → same. Hand-decoder routing decisions are
  pipeline-level, not contract-level.
"""

from __future__ import annotations

import numpy as np
from mesh2sim.contracts import (
    BodyEstimate,
    CameraParams,
    Capabilities,
    Keypoints2D,
    MeshData,
    SkeletonState,
)

# Vendored package — re-imported here because we need the 70 keypoint names.
from sam_3d_body.metadata.mhr70 import mhr_names as _MHR70_NAMES

from .mhr_topology import MHR_JOINT_NAMES, MHR_TOPOLOGY_ID


def _to_float32_2d_xy(arr: np.ndarray) -> np.ndarray:
    """Coerce a (70, 2) keypoint array to ``float32`` without changing values."""
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def sam3db_output_to_body_estimate(
    output: dict,
    *,
    estimator_id: str,
    frame_id: int,
    view_id: str = "mono",
    timestamp: float | None = None,
    camera: CameraParams | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> BodyEstimate:
    """Convert one entry of ``SAM3DBodyEstimator.process_one_image`` output to
    a ``BodyEstimate``.

    The input must be one dict from the list returned by ``process_one_image``,
    after the estimator has moved tensors to CPU/numpy (its normal behaviour).

    All required keys MUST be present — we don't silently skip absent fields,
    because that would propagate downstream as ``has_*`` capability flags that
    aren't trustworthy.
    """
    _require(output, "pred_vertices", "pred_joint_coords", "pred_global_rots", "pred_keypoints_2d")

    mesh = MeshData(
        vertices=output["pred_vertices"],
        topology_id=MHR_TOPOLOGY_ID,
        faces=None,
    )

    skeleton_state = SkeletonState(
        joint_positions=output["pred_joint_coords"],
        joint_orientations=output["pred_global_rots"],
        joint_names=MHR_JOINT_NAMES,
    )

    n_kp = len(_MHR70_NAMES)
    keypoints_2d = Keypoints2D(
        names=list(_MHR70_NAMES),
        xy=_to_float32_2d_xy(output["pred_keypoints_2d"]),
        # SAM 3D Body does not expose per-keypoint confidence in its output
        # dict. We surface a uniform 1.0 vector and document the absence;
        # downstream consumers should treat keypoints_2d.confidence as
        # "presence" not "quality" until we get a real signal.
        confidence=np.ones(n_kp, dtype=np.float32),
    )

    native_params = _collect_mhr_native_params(output)

    capabilities = Capabilities(
        has_mesh=True,
        has_skeleton=True,
        has_2d_keypoints=True,
        has_native_params=True,
    )

    return BodyEstimate(
        estimator_id=estimator_id,
        frame_id=frame_id,
        view_id=view_id,
        timestamp=timestamp,
        capabilities=capabilities,
        native_params=native_params,
        mesh=mesh,
        skeleton_state=skeleton_state,
        keypoints_2d=keypoints_2d,
        camera=camera,
        frame_shape=frame_shape,
    )


# Keys we accept as MHR-native parameters. Anything outside this set is NOT
# pulled into native_params (the SAM 3D Body output dict also carries pipeline
# metadata like ``bbox``/``mask`` that don't belong in BodyEstimate at all).
_MHR_NATIVE_KEYS: tuple[str, ...] = (
    # Core MHR opaque parameters (per the contracts spec: identity 45,
    # model_parameters 204, expression 72 — exposed here as their constituent
    # SAM 3D Body fields). The downstream MHR-to-anatomy adapter knows how to
    # consume these together.
    "shape_params",
    "expr_params",
    "scale_params",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "pred_pose_raw",
    # Camera-frame artefacts produced alongside the MHR fit. Useful to the
    # downstream adapter (e.g. for reconstructing world-frame joints). Kept in
    # native_params because they're estimator-specific, not standard anatomy.
    "pred_cam_t",
    "pred_keypoints_3d",
    "focal_length",
)


def _collect_mhr_native_params(output: dict) -> dict:
    """Pick out MHR-opaque params from the raw output dict, drop the rest.

    Values that are ``None`` are excluded from the dict so downstream consumers
    don't need to defensively check for them.
    """
    out: dict = {}
    for key in _MHR_NATIVE_KEYS:
        if key in output and output[key] is not None:
            out[key] = output[key]
    return out


def _require(output: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in output or output[k] is None]
    if missing:
        raise KeyError(
            f"SAM3DBody output is missing required keys: {missing}. Got keys: {sorted(output)}"
        )


__all__ = [
    "sam3db_output_to_body_estimate",
]
