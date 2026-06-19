"""Topology guards — lock the rig enumeration so a vendor/rig refresh that
shifts joints, vertices, or parameters fails LOUDLY instead of silently
producing wrong angles downstream.

Two tiers:

1. Pure unit tests (no rig, no GPU) — lock the synthetic invariants
   (counts, name pattern, fingerprint string format).
2. ``mhr_rig`` integration test — loads ``mhr_model.pt`` on CPU and verifies
   that the rig's actual buffers hash to the locked sha256 constants. Skipped
   automatically when the rig file is not available (e.g. CI without
   checkpoints).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest
from mesh2sim_frontend_mhr import (
    MHR_JOINT_NAMES,
    MHR_N_JOINTS,
    MHR_N_KEYPOINTS_2D,
    MHR_N_MODEL_PARAMETERS,
    MHR_N_VERTICES,
    MHR_RIG_FINGERPRINT,
    MHR_TOPOLOGY_ID,
)

# ---------------------------------------------------------------------------
# Tier 1: pure unit checks — invariants of the topology declaration itself
# ---------------------------------------------------------------------------


def test_joint_count_locked_at_127():
    """If the rig ever ships with a different joint count, this assertion is
    the first thing that must be reviewed."""
    assert MHR_N_JOINTS == 127


def test_vertex_count_locked_at_18439():
    assert MHR_N_VERTICES == 18439


def test_keypoints_2d_count_locked_at_70():
    assert MHR_N_KEYPOINTS_2D == 70


def test_model_parameters_dim_locked_at_204():
    assert MHR_N_MODEL_PARAMETERS == 204


def test_joint_names_are_positional_and_unique():
    assert len(MHR_JOINT_NAMES) == MHR_N_JOINTS
    assert len(set(MHR_JOINT_NAMES)) == MHR_N_JOINTS, "duplicate joint names"
    for i, name in enumerate(MHR_JOINT_NAMES):
        assert name == f"mhr_joint_{i:03d}", (i, name)


def test_topology_id_pinned():
    assert MHR_TOPOLOGY_ID == "mhr_v1"


def test_fingerprint_constants_are_well_formed_sha256():
    """sha256 hex digests must be 64 lowercase hex chars. Catches typos at PR
    time, before the integration test even runs."""
    expected_keys = {
        "joint_parents_sha256",
        "joint_translation_offsets_sha256",
        "joint_prerotations_sha256",
    }
    assert set(MHR_RIG_FINGERPRINT) == expected_keys
    sha_pat = re.compile(r"^[0-9a-f]{64}$")
    for key, value in MHR_RIG_FINGERPRINT.items():
        assert sha_pat.match(value), f"{key}={value!r} is not a valid sha256"


# ---------------------------------------------------------------------------
# Tier 2: rig-load integration — verify the actual binary matches the locked
# hashes. Skipped if the rig file isn't available locally.
# ---------------------------------------------------------------------------

_DEFAULT_RIG_PATHS = (
    # Standard FastSAM3DToOpenSim layout used during development on WSL2.
    "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    # Production container default (when present).
    "/workspace/stages/frontend_mhr/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
)


def _find_rig_path() -> str | None:
    explicit = os.environ.get("MHR_RIG_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    for p in _DEFAULT_RIG_PATHS:
        if Path(p).is_file():
            return p
    return None


@pytest.mark.mhr_rig
def test_rig_fingerprint_matches_locked_hashes():
    """If this fails, the rig has been refreshed/re-exported. Re-derive the
    sha256s manually and update ``mhr_topology.MHR_RIG_FINGERPRINT`` in the
    same commit that bumps the vendored core, AND review every downstream
    correspondence map for joint-index drift."""
    rig_path = _find_rig_path()
    if rig_path is None:
        pytest.skip(
            f"mhr_model.pt not found; set MHR_RIG_PATH or place the rig at {_DEFAULT_RIG_PATHS[0]}"
        )

    import torch  # heavy import, only for this test

    module = torch.jit.load(rig_path, map_location="cpu").eval()
    skeleton = module.character_torch.skeleton

    parents = skeleton.joint_parents.cpu().numpy()
    trans = skeleton.joint_translation_offsets.cpu().numpy()
    prerot = skeleton.joint_prerotations.cpu().numpy()

    # Sanity: the shapes themselves had better match the locked counts before
    # we even hash anything.
    assert parents.shape == (MHR_N_JOINTS,), (
        f"joint_parents shape {parents.shape}, expected ({MHR_N_JOINTS},)"
    )
    assert trans.shape == (MHR_N_JOINTS, 3)
    assert prerot.shape == (MHR_N_JOINTS, 4)

    # Hash buffers in the same way ``mhr_topology`` documents.
    got = {
        "joint_parents_sha256": hashlib.sha256(parents.tobytes()).hexdigest(),
        "joint_translation_offsets_sha256": hashlib.sha256(trans.tobytes()).hexdigest(),
        "joint_prerotations_sha256": hashlib.sha256(prerot.tobytes()).hexdigest(),
    }

    # Compare key-by-key so the failure message names the buffer that drifted.
    mismatches = {
        k: (MHR_RIG_FINGERPRINT[k], got[k])
        for k in MHR_RIG_FINGERPRINT
        if MHR_RIG_FINGERPRINT[k] != got[k]
    }
    assert not mismatches, (
        "MHR rig fingerprint drift detected — the joint enumeration in the "
        "loaded rig differs from the locked baseline. DO NOT just bump the "
        "constant: every downstream correspondence map must be reviewed "
        "before accepting the new enumeration.\n"
        f"Locked vs got: {mismatches}\nRig path: {rig_path}"
    )


@pytest.mark.mhr_rig
def test_probe_results_use_same_rig_enumeration():
    """The MHR investigation script (``investigate_mhr_params.py``) saves the
    rig's joint_parents alongside its results. We assert the saved enumeration
    is value-equal to the loaded rig — if the probe was run against a
    different rig version, the saved scaling mask is stale and must be
    regenerated.

    Compare by value (``np.array_equal``), not by byte hash: the probe stored
    parents as ``int64`` (its own internal choice) while the rig ships
    natively as ``int32``. Same values, different bytes; only values matter
    for the enumeration check.
    """
    rig_path = _find_rig_path()
    if rig_path is None:
        pytest.skip("rig not available")

    import numpy as np
    import torch  # heavy import, only for this test

    probe_npz = Path(__file__).resolve().parent.parent / "results" / "raw_effects.npz"
    if not probe_npz.is_file():
        pytest.skip(f"no probe results at {probe_npz}")

    probe_parents = np.load(probe_npz)["parents"]
    module = torch.jit.load(rig_path, map_location="cpu").eval()
    rig_parents = module.character_torch.skeleton.joint_parents.cpu().numpy()

    assert np.array_equal(probe_parents, rig_parents), (
        "probe results were computed against a different rig — the saved "
        "joint_parents values differ from the currently-loaded rig. Re-run "
        "investigate_mhr_params.py + disambiguate_suspects.py against the "
        "current rig before trusting scaling_mask_v2.json."
    )
