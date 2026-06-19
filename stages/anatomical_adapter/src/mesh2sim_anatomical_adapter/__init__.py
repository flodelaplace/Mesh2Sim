"""Anatomical stage — hosts the two light per-trial steps of the bridge in mono:

- T3 (``adapter``): per-frame ``BodyEstimate`` + ``CorrespondenceMap`` →
  ``AnatomicalObservation`` (vertex lookup, no transform).
- T5 mono (``assembly``): per-trial temporal assembly of ``AnatomicalObservation``
  → ``AnatomicalTrajectory`` (no fusion, no smoothing, no transform).

Both are pure numpy + ``mesh2sim.contracts``. No torch, no rig, no GPU.
"""

from __future__ import annotations

from .adapter import (
    ADAPTER_ID,
    TopologyMismatchError,
    VertexOutOfBoundsError,
    body_estimate_to_anatomical_observation,
)
from .assembly import (
    ASSEMBLY_ID,
    LandmarkSetMismatchError,
    TemporalOrderError,
    ViewMismatchError,
    assemble_trajectory_mono,
)

__version__ = "0.0.1"

__all__ = [
    # T3 adapter
    "ADAPTER_ID",
    "TopologyMismatchError",
    "VertexOutOfBoundsError",
    "body_estimate_to_anatomical_observation",
    # T5 mono assembly
    "ASSEMBLY_ID",
    "LandmarkSetMismatchError",
    "TemporalOrderError",
    "ViewMismatchError",
    "assemble_trajectory_mono",
]
