"""MHR rig topology — locked constants and integrity fingerprints.

**Why this module exists.** Every correspondence in the pipeline (MHR vertex
index → anatomical landmark, MHR joint index → OpenSim segment, MHR scaling
parameter → bone) is keyed by **position** in the rig's enumeration of joints
and parameters. That enumeration lives in the rig binary (``mhr_model.pt``)
shipped alongside the vendored ``sam_3d_body`` package — NOT in the Python
source. If the rig is ever re-exported with a different joint order or a
different parameter layout, all downstream maps shift by one and the
breakage is invisible (no exception, just wrong angles).

This module locks the rig's identity by sha256 fingerprint of the three
buffers that define the joint enumeration, taken in their **native** dtypes
as stored in the rig file (no silent casting):

- ``joint_parents``              (127,) int32   — sha256 of ``.tobytes()`` on the int32 buffer
- ``joint_translation_offsets``  (127, 3) float32
- ``joint_prerotations``         (127, 4) float32

The integration GPU test loads the rig and verifies these hashes match. A
rig refresh that changes the enumeration MUST update these constants
deliberately — that's the audit trail.

The parameter layout (``parameter_transform`` and the
``pose_parameters`` / ``scaling_parameters`` / ``rigid_parameters`` masks)
also lives in the rig file. The fingerprint above implicitly covers them
because changing the param layout would require regenerating the rig, which
shifts the joint buffer hashes too.
"""

from __future__ import annotations

MHR_TOPOLOGY_ID = "mhr_v1"
"""Stable identifier of the MHR mesh + skeleton topology.

Pinned to the vendored ``sam_3d_body`` commit (see ``third_party/VENDORED.md``).
Downstream consumers (CorrespondenceMap, anatomical adapter) MUST verify
their references to this string before trusting any vertex/joint index.
"""

MHR_N_JOINTS: int = 127
"""Number of joints in the MHR skeleton, as defined by the rig binary."""

MHR_N_VERTICES: int = 18439
"""Number of vertices in the MHR mesh, as defined by the rig binary."""

MHR_N_KEYPOINTS_2D: int = 70
"""Number of named 2D keypoints emitted by SAM 3D Body's camera head.

This is the ``mhr70`` subset; the full MHR keypoint set has 308 entries but
only these 70 have human-readable names in the vendored ``sam_3d_body.metadata.mhr70``.
"""

MHR_N_MODEL_PARAMETERS: int = 204
"""Dimension of the per-frame ``model_parameters`` vector consumed by the rig's
forward pass (``character_torch.model_parameters_to_joint_parameters``).

Decomposition was investigated empirically in
``stages/frontend_mhr/investigate_mhr_params.py`` and
``stages/frontend_mhr/disambiguate_suspects.py`` — see
``stages/frontend_mhr/results/scaling_mask_v2.json`` for the per-index roles.
"""

# Positional names for the 127 joints. The vendored MHR code does not ship a
# human-readable joint name table for the full set (only the 70-keypoint
# subset is named in ``sam_3d_body.metadata.mhr70``). Downstream stages key
# off **joint index** through the frozen correspondence map, not joint names,
# so positional names are operationally sufficient.
MHR_JOINT_NAMES: list[str] = [f"mhr_joint_{i:03d}" for i in range(MHR_N_JOINTS)]


# ---------------------------------------------------------------------------
# Rig integrity fingerprint
# ---------------------------------------------------------------------------
# These sha256 hashes are computed from the rig file at vendor pinning time
# (commit ``936894c37e51de9918012bcbc9ba2d9c20f73252`` of yangtiming/Fast-SAM-3D-Body).
# The integration test ``tests/test_rig_fingerprint.py`` (marker ``mhr_rig``)
# loads the rig and asserts each buffer hashes to the value below. If you
# refresh the vendored core OR the rig file, recompute these and review the
# entire downstream chain before accepting the change.

MHR_RIG_FINGERPRINT: dict[str, str] = {
    "joint_parents_sha256":
        "d0938261185746968d84adb60f012d8de8719170bd65b6a3fe7a6d6389e1ab02",
    "joint_translation_offsets_sha256":
        "3f9a396f5c639393e8bbc86da703282d7e460249cd3233ebcdb918d5c6709e34",
    "joint_prerotations_sha256":
        "2b14e842e03e679cb149a9703a650bcd627d3e15921d55ab43710ad53ac7177a",
}
"""SHA-256 fingerprints of the rig's joint-enumeration buffers, locked at vendor
pinning time. Each value is the hex digest of ``arr.tobytes()`` for the
named buffer in its **native** dtype (int32 for parents, float32 for the
two others — NO cast). The rig binary determines the bytes; we hash exactly
what it ships.
"""


__all__ = [
    "MHR_JOINT_NAMES",
    "MHR_N_JOINTS",
    "MHR_N_KEYPOINTS_2D",
    "MHR_N_MODEL_PARAMETERS",
    "MHR_N_VERTICES",
    "MHR_RIG_FINGERPRINT",
    "MHR_TOPOLOGY_ID",
]
