"""Serialization for Mesh2Sim contracts.

Storage layout (one contract → one directory):

    <path>/
      manifest.json   # structure, scalars, enums, list of array references
      arrays.npz      # all numpy arrays keyed by their dotted path

The JSON file holds ONLY metadata. Every ndarray is replaced in the JSON by a placeholder
``{"__npz_key__": "<dotted.key>"}`` and the actual buffer lives in ``arrays.npz``. This is
load-bearing: it keeps the JSON small, preserves dtype exactly through the round-trip, and
guarantees that vertices/positions never accidentally land as JSON lists.

Round-trip guarantee: ``load(cls, save(c)) == c`` for every contract, including ``dtype``
and bit-for-bit array equality. Only documented dtype coercions (e.g. mesh vertices to
float32) happen, and they happen at construction time, not at I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

from .version import SCHEMA_VERSION, check_compatible

NPZ_KEY = "__npz_key__"
"""Marker key used in JSON to point at an array stored in ``arrays.npz``."""

_MANIFEST_FILE = "manifest.json"
_ARRAYS_FILE = "arrays.npz"

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Internal walkers
# ---------------------------------------------------------------------------


def _extract_arrays(obj: Any, prefix: str = "") -> tuple[Any, dict[str, np.ndarray]]:
    """Recursively walk ``obj``, replacing ndarrays with ``{NPZ_KEY: <path>}`` placeholders.

    Returns ``(manifest_subtree, arrays_dict)`` where ``arrays_dict`` maps dotted keys to
    the original ndarrays. ``obj`` must be the output of ``BaseModel.model_dump(mode="python")``
    or equivalent (dicts, lists, primitives, ndarrays, enums, tuples).

    Note: dict keys must not contain ``.`` since we use it as a path separator. Landmark
    names and segment names in the vocab never contain dots; arbitrary user data in
    ``Provenance.extra`` is the caller's responsibility.
    """
    if isinstance(obj, np.ndarray):
        return {NPZ_KEY: prefix}, {prefix: obj}

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            child, child_arrays = _extract_arrays(v, child_prefix)
            out[k] = child
            arrays.update(child_arrays)
        return out, arrays

    if isinstance(obj, (list, tuple)):
        out_list: list[Any] = []
        arrays = {}
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}.{i}" if prefix else str(i)
            child, child_arrays = _extract_arrays(v, child_prefix)
            out_list.append(child)
            arrays.update(child_arrays)
        return out_list, arrays

    return obj, {}


def _restore_arrays(obj: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Reverse of ``_extract_arrays``: substitute placeholders with the actual ndarrays."""
    if isinstance(obj, dict):
        if NPZ_KEY in obj and len(obj) == 1:
            key = obj[NPZ_KEY]
            if key not in arrays:
                raise ValueError(f"missing array {key!r} in arrays.npz")
            return arrays[key]
        return {k: _restore_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(v, arrays) for v in obj]
    return obj


def _json_default(o: object) -> object:
    """JSON encoder for non-standard scalars that survive ``model_dump``."""
    if isinstance(o, np.generic):
        # numpy scalars (e.g. np.float32(0.5)): cast to nearest Python type
        return o.item()
    raise TypeError(f"object of type {type(o).__name__} is not JSON-serializable")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save(contract: BaseModel, path: str | Path) -> Path:
    """Serialize ``contract`` into a directory at ``path``.

    Creates ``path/manifest.json`` and (if the contract holds any ndarrays) ``path/arrays.npz``.
    Returns the directory path.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    body = contract.model_dump(mode="python")
    manifest_body, arrays = _extract_arrays(body)

    envelope = {
        "__class__": type(contract).__name__,
        "__schema_version__": SCHEMA_VERSION,
        "body": manifest_body,
    }

    with (out / _MANIFEST_FILE).open("w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, default=_json_default)

    if arrays:
        # ``np.savez_compressed`` preserves dtype exactly.
        np.savez_compressed(out / _ARRAYS_FILE, **arrays)

    return out


def load(cls: type[T], path: str | Path) -> T:
    """Deserialize a contract of type ``cls`` from a directory at ``path``.

    Validates:
    - file present and class name matches ``cls``
    - manifest schema_version is compatible with the running package
    - all model invariants (re-runs every pydantic validator)
    """
    src = Path(path)
    manifest_file = src / _MANIFEST_FILE
    if not manifest_file.exists():
        raise FileNotFoundError(f"no manifest at {manifest_file}")

    with manifest_file.open("r", encoding="utf-8") as f:
        envelope = json.load(f)

    expected_name = cls.__name__
    actual_name = envelope.get("__class__")
    if actual_name != expected_name:
        raise ValueError(f"manifest is a {actual_name!r}, not a {expected_name!r}")

    check_compatible(envelope["__schema_version__"])

    arrays_file = src / _ARRAYS_FILE
    arrays: dict[str, np.ndarray] = {}
    if arrays_file.exists():
        with np.load(arrays_file, allow_pickle=False) as npz:
            # Copy out of the lazy npz handle so the file can close cleanly.
            arrays = {k: npz[k].copy() for k in npz.files}

    body = _restore_arrays(envelope["body"], arrays)
    return cls.model_validate(body)


__all__ = ["NPZ_KEY", "load", "save"]
