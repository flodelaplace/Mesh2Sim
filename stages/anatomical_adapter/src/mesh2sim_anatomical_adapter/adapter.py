"""T3 adapter — pure function mapping BodyEstimate (mesh MHR) + CorrespondenceMap
to AnatomicalObservation (named anatomical landmarks).

Frame-by-frame, no state, no GPU, no transform of any kind beyond a vertex lookup
and a centroid average. See ``stages/anatomical_adapter/README.md`` for the contract.
"""

from __future__ import annotations

import numpy as np
from mesh2sim.contracts import (
    AnatomicalObservation,
    BodyEstimate,
    Capabilities,
    CorrespondenceMap,
    Landmark,
    Pos3DFrame,
    Provenance,
    Source,
)

ADAPTER_ID = "mesh2sim-anatomical-adapter@0.0.1"
"""Stable identifier embedded in every ``AnatomicalObservation.provenance.adapter_id``."""


class TopologyMismatchError(ValueError):
    """Raised when the CorrespondenceMap's mhr_topology_id does not match the
    BodyEstimate's mesh.topology_id. Vertex indices would silently point to the wrong
    geometry if this were tolerated."""


class VertexOutOfBoundsError(IndexError):
    """Raised when a marker's mhr_vertices contains an index outside the mesh's vertex
    range. Indicates either a corrupt CorrespondenceMap or a mesh resampling change."""


def body_estimate_to_anatomical_observation(
    body_estimate: BodyEstimate,
    correspondence_map: CorrespondenceMap,
    *,
    default_visibility: float = 1.0,
) -> AnatomicalObservation:
    """Extract anatomical landmark positions by reading mhr_vertices on the subject mesh.

    Args:
        body_estimate: per-frame MHR estimate. Must carry mesh data (``capabilities.has_mesh``
            and a non-None ``mesh`` field).
        correspondence_map: the marker → mhr_vertices table, authored once per OpenSim model.
        default_visibility: written into every ``Landmark.visibility``. This adapter has no
            way to know real visibility (the mesh is always "there" by definition); a separate
            T2 / T5 stage will refine it from 2D evidence or multi-view occlusion.

    Returns:
        AnatomicalObservation with one Landmark per marker in the map.
        ``pos3d_frame = Pos3DFrame.camera`` (the BodyEstimate's mesh frame; T6 transforms to world).

    Raises:
        ValueError: if the BodyEstimate has no mesh.
        TopologyMismatchError: if mhr_topology_id strings disagree.
        VertexOutOfBoundsError: if any marker's vertex index falls outside ``[0, V)``.
    """
    if body_estimate.mesh is None:
        raise ValueError(
            "BodyEstimate has no mesh; cannot extract anatomical landmarks from vertices. "
            "Upstream estimator must emit mesh data (capabilities.has_mesh=True)."
        )

    map_topology = correspondence_map.mhr_topology_id
    mesh_topology = body_estimate.mesh.topology_id
    if map_topology != mesh_topology:
        raise TopologyMismatchError(
            f"mhr_topology_id mismatch: CorrespondenceMap declares {map_topology!r} but "
            f"BodyEstimate.mesh.topology_id={mesh_topology!r}. "
            "Vertex indices would point to the wrong geometry; refuse to continue. "
            "Either regenerate the CorrespondenceMap against this topology, or use a "
            "BodyEstimate produced by an estimator matching the map's topology."
        )

    vertices = body_estimate.mesh.vertices  # (V, 3)
    n_vertices = int(vertices.shape[0])

    landmarks: dict[str, Landmark] = {}
    for marker in correspondence_map.markers:
        idxs = list(marker.mhr_vertices)
        max_idx = max(idxs)
        min_idx = min(idxs)
        if min_idx < 0 or max_idx >= n_vertices:
            raise VertexOutOfBoundsError(
                f"marker {marker.name!r}: mhr_vertices out of range [0, {n_vertices}); "
                f"got min={min_idx}, max={max_idx} (full list={idxs})"
            )

        # Vertex lookup. For multi-vertex markers we use the arithmetic mean (centroid),
        # which is the simplest aggregation and matches the documented convention.
        sel = vertices[idxs]  # (N, 3)
        if sel.shape[0] == 1:
            pos = np.asarray(sel[0], dtype=np.float64).copy()
        else:
            pos = sel.mean(axis=0).astype(np.float64)

        landmarks[marker.name] = Landmark(
            pos_3d=pos,
            confidence=1.0,
            visibility=float(default_visibility),
            source=Source.bony if marker.fixed else Source.soft,
        )

    return AnatomicalObservation(
        frame_id=body_estimate.frame_id,
        view_id=body_estimate.view_id,
        timestamp=body_estimate.timestamp,
        landmarks=landmarks,
        # The mesh comes out of the camera-relative inference (before T6 applies the
        # world transform). We label it "camera" — the closest match in the enum.
        pos3d_frame=Pos3DFrame.camera,
        # No segment_frames, no shape_descriptor, no dense_surface in this minimal
        # adapter — all capability flags stay False.
        capabilities=Capabilities(),
        provenance=Provenance(
            estimator_id=body_estimate.estimator_id,
            adapter_id=ADAPTER_ID,
            correspondence_map_id=correspondence_map.marker_set,
        ),
    )


__all__ = [
    "ADAPTER_ID",
    "TopologyMismatchError",
    "VertexOutOfBoundsError",
    "body_estimate_to_anatomical_observation",
]
