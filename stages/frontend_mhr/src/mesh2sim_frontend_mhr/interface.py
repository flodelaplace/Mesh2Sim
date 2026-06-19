"""Public estimator interface for the frontend stage.

A ``MeshEstimator`` consumes one RGB frame and produces zero or more
``BodyEstimate`` contracts. Implementations wrap a specific mesh estimator
(e.g. SAM 3D Body). They are stateful (typically hold a loaded model on GPU).

Higher-level orchestration (video iteration, multi-person tracking, sequence
serialization) lives outside this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from mesh2sim.contracts import BodyEstimate, CameraParams


@runtime_checkable
class MeshEstimator(Protocol):
    """Producer of ``BodyEstimate`` from a single RGB frame.

    Implementations must populate ``BodyEstimate.estimator_id`` with a stable
    string that uniquely identifies the underlying model AND its version
    (e.g. ``"fast-sam-3d-body@<short-sha>"``), so downstream consumers can
    distinguish two outputs produced by different implementations.
    """

    estimator_id: str
    """Stable identifier of the underlying model + version."""

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        frame_id: int,
        *,
        view_id: str = "mono",
        timestamp: float | None = None,
        camera: CameraParams | None = None,
    ) -> list[BodyEstimate]:
        """Run inference on a single frame.

        Args:
            frame_rgb: ``(H, W, 3)`` ``uint8`` RGB image.
            frame_id: integer frame index in the source sequence.
            view_id: source view identifier; ``"mono"`` in single-camera mode.
            timestamp: optional time (seconds) of this frame, if known.
            camera: optional pre-calibrated intrinsics/extrinsics for this view.

        Returns:
            A list of ``BodyEstimate``, one per person the implementation
            decides to emit (empty if no person detected; one in mono-subject
            mode; N in multi-person mode).
        """
        ...


__all__ = ["MeshEstimator"]
