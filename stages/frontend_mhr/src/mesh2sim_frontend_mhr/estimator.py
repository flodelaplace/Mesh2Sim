"""``FastSAM3DBodyEstimator``: ``MeshEstimator`` backed by the vendored
``sam_3d_body.SAM3DBodyEstimator``.

Heavy model loading is **not** done here; see ``from_pretrained`` (raises
``NotImplementedError`` until the integration ticket wires real weights).
For now, the model is injected by the caller — which lets us mock it in unit
tests without ever touching the GPU.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mesh2sim.contracts import BodyEstimate, CameraParams

from .adapter import sam3db_output_to_body_estimate

# Pinned to the vendored commit so two BodyEstimates produced by different
# vendored SHAs are distinguishable by their estimator_id alone.
DEFAULT_ESTIMATOR_ID = "fast-sam-3d-body@936894c"


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
    def from_pretrained(cls, *args, **kwargs) -> "FastSAM3DBodyEstimator":
        """Build the estimator from pre-trained checkpoints. Not implemented yet."""
        raise NotImplementedError(
            "Real model loading lives in the integration ticket. Inject a "
            "ready ``SAM3DBodyEstimator`` (or a mock) into the constructor."
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
