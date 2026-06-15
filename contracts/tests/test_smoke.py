"""Smoke test: package imports and exposes its version."""

import mesh2sim.contracts as contracts


def test_import_exposes_version():
    assert isinstance(contracts.__version__, str)
    assert isinstance(contracts.SCHEMA_VERSION, str)


def test_light_dependencies_only():
    import sys

    forbidden = {"torch", "mmcv", "nimblephysics", "jax", "mujoco"}
    leaked = forbidden & set(sys.modules)
    assert not leaked, f"contracts must stay light, but found: {leaked}"
