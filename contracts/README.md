# mesh2sim-contracts

Shared data contracts (schemas) that travel between Mesh2Sim stages.

## Rules

- Light dependencies only: `pydantic` (v2) + `numpy`. Never add heavy deps (no torch, mmcv,
  nimblephysics, jax, mujoco). This package must install cleanly into every stage environment.
- Anatomical semantics, not model-specific. See `docs/contracts_spec.md` at repo root.
- No imports from `stages/*`, `orchestrator/`, or anywhere else in the monorepo.

## Install (editable)

```bash
pip install -e ./contracts[dev]
```

## Test

```bash
pytest contracts/tests
```
