"""Version compatibility rules at load time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mesh2sim.contracts import (
    SCHEMA_VERSION,
    AnatomicalObservation,
    Capabilities,
    Landmark,
    Pos3DFrame,
    Provenance,
    check_compatible,
    is_compatible,
    load,
    save,
)


def test_current_version_is_self_compatible():
    assert is_compatible(SCHEMA_VERSION)
    check_compatible(SCHEMA_VERSION)


def test_minor_difference_is_compatible():
    major = SCHEMA_VERSION.split(".")[0]
    assert is_compatible(f"{major}.99.99")
    assert is_compatible(f"{major}.0.0")


def test_major_difference_is_incompatible():
    major = int(SCHEMA_VERSION.split(".")[0])
    bumped = f"{major + 1}.0.0"
    assert not is_compatible(bumped)
    with pytest.raises(ValueError, match="incompatible schema_version"):
        check_compatible(bumped)


def test_load_refuses_incompatible_major(tmp_path: Path):
    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={"RASI": Landmark(pos_3d=np.array([0.0, 0.0, 0.0]))},
        pos3d_frame=Pos3DFrame.world,
        capabilities=Capabilities(),
        provenance=Provenance(),
    )
    out = save(obs, tmp_path / "obs")

    # Tamper with the envelope to declare a major bump.
    manifest_path = out / "manifest.json"
    envelope = json.loads(manifest_path.read_text())
    bumped = int(SCHEMA_VERSION.split(".")[0]) + 1
    envelope["__schema_version__"] = f"{bumped}.0.0"
    manifest_path.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="incompatible schema_version"):
        load(AnatomicalObservation, out)


def test_load_refuses_wrong_class(tmp_path: Path):
    from mesh2sim.contracts import BodyEstimate

    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={},
        pos3d_frame=Pos3DFrame.none,
        capabilities=Capabilities(),
        provenance=Provenance(),
    )
    out = save(obs, tmp_path / "obs")

    with pytest.raises(ValueError, match="not a 'BodyEstimate'"):
        load(BodyEstimate, out)
