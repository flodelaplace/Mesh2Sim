"""Critical guarantee: the JSON manifest never holds bulk array data.

A regression here (e.g. someone changes ``save`` to dump arrays as JSON lists "for
debugging") would silently bloat manifests, lose dtype, and break the layout contract.
We check this two ways:

1. Manifest size stays small even when the mesh is huge.
2. Walking the parsed JSON, no list-of-numbers is longer than a small threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mesh2sim.contracts import (
    AnatomicalObservation,
    Capabilities,
    Landmark,
    MeshData,
    Pos3DFrame,
    Provenance,
    save,
)


def _walk_lists(obj, found: list[int]) -> None:
    if isinstance(obj, list):
        if all(isinstance(x, (int, float)) for x in obj):
            found.append(len(obj))
        else:
            for v in obj:
                _walk_lists(v, found)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_lists(v, found)


def test_manifest_holds_no_large_numeric_lists(tmp_path: Path):
    v = 2000
    f = 4000
    big_mesh = MeshData(
        vertices=np.random.default_rng(0).standard_normal((v, 3)).astype(np.float32),
        topology_id="mhr_v1",
        faces=np.random.default_rng(0).integers(0, v, size=(f, 3), dtype=np.int32),
    )
    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={"RASI": Landmark(pos_3d=np.array([0.1, 1.0, 0.0]))},
        pos3d_frame=Pos3DFrame.world,
        dense_surface=big_mesh,
        capabilities=Capabilities(has_mesh=True),
        provenance=Provenance(),
    )

    out = save(obs, tmp_path / "obs")

    manifest_size = (out / "manifest.json").stat().st_size
    arrays_size = (out / "arrays.npz").stat().st_size

    # Manifest is metadata only: should stay well under a few KB regardless of mesh size.
    assert manifest_size < 5_000, (
        f"manifest grew to {manifest_size} bytes — arrays may have leaked into JSON"
    )
    # Arrays file holds the bulk.
    assert arrays_size > 10_000, f"arrays.npz too small: {arrays_size} bytes"

    # Parse the JSON and assert no embedded numeric list exceeds a small length.
    with (out / "manifest.json").open() as f_:
        envelope = json.load(f_)

    found_list_lengths: list[int] = []
    _walk_lists(envelope, found_list_lengths)
    # A few short numeric lists are fine (e.g. (3,) pos_3d landmarks, resolution tuples).
    # Anything longer would be smuggled array data.
    too_long = [n for n in found_list_lengths if n > 8]
    assert not too_long, f"manifest contains suspiciously long numeric lists: {too_long}"


def test_arrays_are_npz_keys_we_can_introspect(tmp_path: Path):
    """The arrays.npz holds keys that match the dotted paths we expect."""
    obs = AnatomicalObservation(
        frame_id=0,
        view_id="cam0",
        landmarks={
            "RASI": Landmark(pos_3d=np.array([0.1, 1.0, 0.0])),
        },
        pos3d_frame=Pos3DFrame.world,
        capabilities=Capabilities(),
        provenance=Provenance(),
    )
    out = save(obs, tmp_path / "obs")
    with np.load(out / "arrays.npz") as npz:
        keys = set(npz.files)
    # Landmark pos_3d should be addressable by its dotted path (paths start at the body
    # root, no envelope prefix).
    assert "landmarks.RASI.pos_3d" in keys
