"""``FastSAM3DBodyEstimator``: ``MeshEstimator`` backed by the vendored
``sam_3d_body.SAM3DBodyEstimator``.

Heavy model loading is **not** done here; see ``from_pretrained`` (raises
``NotImplementedError`` until the integration ticket wires real weights).
For now, the model is injected by the caller — which lets us mock it in unit
tests without ever touching the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from mesh2sim.contracts import BodyEstimate, CameraParams

from .adapter import sam3db_output_to_body_estimate

# Pinned to the vendored commit so two BodyEstimates produced by different
# vendored SHAs are distinguishable by their estimator_id alone.
DEFAULT_ESTIMATOR_ID = "fast-sam-3d-body@936894c"

# ---------------------------------------------------------------------------
# Expected checkpoint layout for ``from_pretrained(checkpoint_dir=...)``
# ---------------------------------------------------------------------------
# Mirrors the upstream HuggingFace snapshot ("facebook/sam-3d-body"-style)
# and the user's existing FastSAM3DToOpenSim production layout:
#
#   <checkpoint_dir>/
#   ├── model.ckpt                    (~2.1 GB: SAM 3D Body state_dict)
#   ├── model_config.yaml             (~1.5 KB: model config consumed by load_sam_3d_body)
#   └── assets/
#       └── mhr_model.pt              (~700 MB: MHR rig TorchScript bundle)
#
# Optional (TensorRT acceleration, not required):
#   └── backbone_trt/
#       └── backbone_dinov3_fp16.engine
#
# Quick install pointers (do this once per machine, outside the repo):
#   - HuggingFace download (preferred):
#       from huggingface_hub import snapshot_download
#       snapshot_download(repo_id="facebook/sam-3d-body", local_dir="/path/to/checkpoints")
#   - Or use FastSAM3DToOpenSim's run-time download in setup_sam_3d_body.
_CKPT_FILE = "model.ckpt"
_CFG_FILE = "model_config.yaml"
_RIG_RELPATH = ("assets", "mhr_model.pt")


class _RawEstimator(Protocol):
    """The slice of ``sam_3d_body.SAM3DBodyEstimator`` we depend on.

    Declared as a Protocol so tests can inject a ``unittest.mock.Mock`` (or
    any duck-typed substitute) without importing the heavy class.
    """

    def process_one_image(self, img, *args, **kwargs) -> list[dict]: ...


class FastSAM3DBodyEstimator:
    """Wrap a ``sam_3d_body`` estimator and emit ``BodyEstimate`` contracts.

    Args:
        model:                 a ready-to-use ``SAM3DBodyEstimator`` (or mock).
        estimator_id:          string identifier embedded in every output;
                               defaults to a vendor-commit-pinned value.
        main_subject_only:     if True, in each frame keep only the person
                               with the largest bbox area (xyxy) and emit one
                               ``BodyEstimate`` per frame. Set False for
                               future multi-person work (see contract debt in
                               the stage README).
        process_one_image_kwargs: extra kwargs forwarded verbatim to
                               ``model.process_one_image`` (e.g.
                               ``hand_box_source="yolo_pose"``,
                               ``inference_type="body"``). Kept here so the
                               estimator is configurable without subclassing.
    """

    def __init__(
        self,
        model: _RawEstimator,
        *,
        estimator_id: str = DEFAULT_ESTIMATOR_ID,
        main_subject_only: bool = True,
        process_one_image_kwargs: dict | None = None,
    ):
        self.model = model
        self.estimator_id = estimator_id
        self.main_subject_only = main_subject_only
        self._raw_kwargs: dict = dict(process_one_image_kwargs or {})

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint_dir: str | Path | None = None,
        hf_repo_id: str | None = None,
        device: str = "cuda",
        estimator_id: str = DEFAULT_ESTIMATOR_ID,
        main_subject_only: bool = True,
        process_one_image_kwargs: dict | None = None,
    ) -> "FastSAM3DBodyEstimator":
        """Build a fully-loaded estimator from checkpoints.

        Provide **either**:
        - ``checkpoint_dir`` — path to a local sam-3d-body-dinov3 checkpoint
          directory (see module-level docstring for the expected layout); OR
        - ``hf_repo_id`` — HuggingFace repository id; ``snapshot_download``
          fetches the artefacts on first call (cached afterwards). Standard
          repo id is ``"facebook/sam-3d-body"``.

        The MHR rig is loaded via ``torch.jit.load`` (the fallback path of
        ``MHRHead`` when the optional ``mhr`` Python library is not
        installed — which is our env by design). A benign
        "Momentum is not enabled" warning is expected.

        No human detector is wired here. Two ways to drive inference:
        - Provide ``process_one_image_kwargs={"bboxes": np.array([[x1, y1, x2, y2]])}``
          to pass bounding boxes manually, OR
        - Construct the estimator yourself with a detector and inject via the
          regular constructor (``FastSAM3DBodyEstimator(model=...)``).
        """
        if hf_repo_id and checkpoint_dir:
            raise ValueError(
                "provide either hf_repo_id OR checkpoint_dir, not both"
            )
        if not (hf_repo_id or checkpoint_dir):
            raise ValueError(
                "provide either hf_repo_id OR checkpoint_dir"
            )

        # Heavy imports deferred until from_pretrained is actually called, so
        # ``import mesh2sim_frontend_mhr`` stays light (matters for mocked
        # tests that don't touch the vendored core).
        from sam_3d_body.build_models import load_sam_3d_body, load_sam_3d_body_hf
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

        if hf_repo_id:
            model, cfg = load_sam_3d_body_hf(hf_repo_id)
        else:
            ckpt_dir = Path(checkpoint_dir)  # type: ignore[arg-type]
            ckpt_path = ckpt_dir / _CKPT_FILE
            rig_path = ckpt_dir.joinpath(*_RIG_RELPATH)
            cfg_path = ckpt_dir / _CFG_FILE
            for p in (ckpt_path, rig_path, cfg_path):
                if not p.is_file():
                    raise FileNotFoundError(
                        f"missing checkpoint artefact {p}; expected layout "
                        f"is documented in mesh2sim_frontend_mhr.estimator"
                    )
            model, cfg = load_sam_3d_body(
                checkpoint_path=str(ckpt_path),
                device=device,
                mhr_path=str(rig_path),
            )

        raw_estimator = SAM3DBodyEstimator(
            model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None
        )
        return cls(
            model=raw_estimator,
            estimator_id=estimator_id,
            main_subject_only=main_subject_only,
            process_one_image_kwargs=process_one_image_kwargs,
        )

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        frame_id: int,
        *,
        view_id: str = "mono",
        timestamp: float | None = None,
        camera: CameraParams | None = None,
    ) -> list[BodyEstimate]:
        """Run inference and convert each detected person to a ``BodyEstimate``."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(
                f"frame_rgb must be (H, W, 3); got shape {frame_rgb.shape}"
            )

        raw_outputs = self.model.process_one_image(frame_rgb, **self._raw_kwargs)
        if not raw_outputs:
            return []

        if self.main_subject_only:
            raw_outputs = [_select_main_subject(raw_outputs)]

        frame_shape = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
        return [
            sam3db_output_to_body_estimate(
                output=out,
                estimator_id=self.estimator_id,
                frame_id=frame_id,
                view_id=view_id,
                timestamp=timestamp,
                camera=camera,
                frame_shape=frame_shape,
            )
            for out in raw_outputs
        ]


def _select_main_subject(outputs: list[dict]) -> dict:
    """Pick the person with the largest bbox area (xyxy format).

    Robust enough as a mono-subject heuristic when the subject is the closest
    person to the camera. Replace with track-based selection when we wire
    multi-person tracking in a later ticket.

    The bbox key is consumed here at the estimator level and DOES NOT survive
    to ``BodyEstimate`` — see ``adapter.py`` and the README contract-debt note.
    """
    if not outputs:
        raise ValueError("cannot select main subject from empty output list")

    def area(o: dict) -> float:
        if "bbox" not in o or o["bbox"] is None:
            return 0.0
        x1, y1, x2, y2 = (float(v) for v in o["bbox"])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(outputs, key=area)


__all__ = ["DEFAULT_ESTIMATOR_ID", "FastSAM3DBodyEstimator"]
