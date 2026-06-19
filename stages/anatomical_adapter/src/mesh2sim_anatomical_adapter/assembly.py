"""T5 mono — temporal assembly of per-frame ``AnatomicalObservation`` into a
per-trial ``AnatomicalTrajectory``.

In mono mode there is no multi-view fusion; this stage is a pure assembler.

Strict perimeter:
- Sort observations by timestamp (or frame_id when timestamps are absent) and
  refuse non-monotonic / duplicate streams.
- Verify all observations share the same landmark name set and the same view_id.
- Stack positions and confidence into the trajectory tensors.
- Preserve missing markers as NaN (positions) and 0 (confidence). DO NOT
  interpolate or smooth — that's T7's job. DO NOT transform frames — that's
  T6b's job. The trajectory keeps the positions in whatever frame the
  observations were in (typically ``Pos3DFrame.camera``).
- Attach the trial's locked ``ShapeDescriptor`` (from T4) to the trajectory.

The contract ``AnatomicalTrajectory`` doesn't carry an explicit ``pos3d_frame``
field, so we record the observations' source frame in
``provenance.extra["assembled_from_pos3d_frame"]`` for downstream traceability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mesh2sim.contracts import (
    AnatomicalObservation,
    AnatomicalTrajectory,
    Mode,
    Provenance,
    ShapeDescriptor,
    Task,
)

ASSEMBLY_ID = "mesh2sim-anatomical-adapter.assembly@0.0.1"
"""Stable identifier embedded in every assembled trajectory's provenance."""


class TemporalOrderError(ValueError):
    """Raised when timestamps or frame_ids are duplicated or non-monotonic
    (after sorting). T5 refuses to silently drop or merge frames."""


class LandmarkSetMismatchError(ValueError):
    """Raised when observations don't share the same landmark name set.
    Stacking trajectories with heterogeneous landmark sets would produce a
    silently wrong tensor."""


class ViewMismatchError(ValueError):
    """Raised when observations carry mixed ``view_id``. In mono mode all
    frames must share a single view_id (typically ``"mono"``)."""


def assemble_trajectory_mono(
    observations: Sequence[AnatomicalObservation],
    shape_descriptor: ShapeDescriptor,
    *,
    subject_id: str,
    trial_id: str,
    task: Task | str,
    fps: float,
    provenance: Provenance | None = None,
) -> AnatomicalTrajectory:
    """Assemble a per-frame observation sequence into a per-trial trajectory.

    Args:
        observations:        per-frame ``AnatomicalObservation`` (mono, single view_id).
        shape_descriptor:    locked subject shape produced by T4.
        subject_id:          subject identifier (caller-provided, not in observations).
        trial_id:            trial identifier (caller-provided).
        task:                ``Task`` enum or its string form ("gait", "sts", "other").
        fps:                 frame rate of the trajectory (caller-provided, not derived
                             to avoid silent footguns from irregular timestamps).
        provenance:          optional override; default = copy from first observation
                             with assembly_id tagged in ``extra``.

    Returns:
        AnatomicalTrajectory with positions ``(T, L, 3)`` float64, confidence
        ``(T, L)`` float32, NaN/0 at missing markers, ``shape_descriptor`` attached.

    Raises:
        TemporalOrderError, LandmarkSetMismatchError, ViewMismatchError, ValueError.
    """
    if len(observations) == 0:
        raise TemporalOrderError("cannot assemble an empty observation sequence")

    # ----- 1. view consistency (mono = all same view_id) ------------------
    view_ids = {obs.view_id for obs in observations}
    if len(view_ids) > 1:
        raise ViewMismatchError(
            f"observations carry multiple view_ids in mono mode: {sorted(view_ids)!r}"
        )
    (view_id,) = view_ids

    # ----- 2. landmark set consistency -----------------------------------
    canonical_names = list(observations[0].landmarks.keys())
    if not canonical_names:
        raise ValueError(
            "first observation has no landmarks; cannot infer the trajectory landmark set"
        )
    canonical_set = set(canonical_names)
    for i, obs in enumerate(observations):
        obs_set = set(obs.landmarks.keys())
        if obs_set != canonical_set:
            missing = canonical_set - obs_set
            extra = obs_set - canonical_set
            raise LandmarkSetMismatchError(
                f"observation at index {i} (frame_id={obs.frame_id}) has a different "
                f"landmark set than the first observation. "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}. "
                "Trajectories cannot be assembled across heterogeneous landmark sets."
            )

    # ----- 3. temporal ordering ------------------------------------------
    # Use timestamps if all are present, else fall back to frame_id. Reject any
    # duplicate / non-monotonic stream (we don't silently drop or merge).
    have_timestamps = all(obs.timestamp is not None for obs in observations)
    if have_timestamps:
        sorted_obs = sorted(observations, key=lambda o: o.timestamp)
        for i in range(1, len(sorted_obs)):
            if sorted_obs[i].timestamp <= sorted_obs[i - 1].timestamp:
                raise TemporalOrderError(
                    "duplicate or non-monotonic timestamp after sorting at index "
                    f"{i}: {sorted_obs[i - 1].timestamp} → {sorted_obs[i].timestamp} "
                    f"(frame_ids {sorted_obs[i - 1].frame_id} → {sorted_obs[i].frame_id})"
                )
    else:
        sorted_obs = sorted(observations, key=lambda o: o.frame_id)
        for i in range(1, len(sorted_obs)):
            if sorted_obs[i].frame_id <= sorted_obs[i - 1].frame_id:
                raise TemporalOrderError(
                    "duplicate or non-monotonic frame_id after sorting at index "
                    f"{i}: {sorted_obs[i - 1].frame_id} → {sorted_obs[i].frame_id} "
                    "(no timestamps available to disambiguate)"
                )

    # ----- 4. stack positions and confidence -----------------------------
    n_frames = len(sorted_obs)
    n_landmarks = len(canonical_names)
    positions = np.full((n_frames, n_landmarks, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((n_frames, n_landmarks), dtype=np.float32)

    for t, obs in enumerate(sorted_obs):
        for el, name in enumerate(canonical_names):
            lm = obs.landmarks[name]
            # "Missing" rule: pos_3d is None OR visibility <= 0.
            # We DO NOT interpolate or smooth — T5 is an honest assembler.
            if lm.pos_3d is None or lm.visibility <= 0.0:
                continue
            positions[t, el] = lm.pos_3d
            confidence[t, el] = float(lm.confidence)

    # ----- 5. provenance --------------------------------------------------
    if provenance is None:
        base = sorted_obs[0].provenance
        extra = dict(base.extra or {})
        extra["assembly_id"] = ASSEMBLY_ID
        # AnatomicalTrajectory has no explicit pos3d_frame field; we tag the
        # source frame here so T6b (and any consumer) can verify what it's
        # working with.
        source_frames = {obs.pos3d_frame.value for obs in sorted_obs}
        extra["assembled_from_pos3d_frame"] = (
            next(iter(source_frames)) if len(source_frames) == 1 else sorted(source_frames)
        )
        provenance = base.model_copy(update={"extra": extra})

    # ----- 6. task -------------------------------------------------------
    task_enum = Task(task) if isinstance(task, str) else task

    return AnatomicalTrajectory(
        subject_id=subject_id,
        trial_id=trial_id,
        task=task_enum,
        mode=Mode.mono,
        fps=float(fps),
        landmark_names=canonical_names,
        positions=positions,
        confidence=confidence,
        shape_descriptor=shape_descriptor,
        views_used=[view_id],
        uncertainty=None,
        provenance=provenance,
    )


__all__ = [
    "ASSEMBLY_ID",
    "LandmarkSetMismatchError",
    "TemporalOrderError",
    "ViewMismatchError",
    "assemble_trajectory_mono",
]
