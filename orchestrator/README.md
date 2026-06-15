# orchestrator

Cross-stage runner. The orchestrator is the only component that crosses stage boundaries — it
does so by **subprocess / container invocation**, never by Python import. Each stage runs in
its own environment; the orchestrator wires stages together via serialized contract files
(see `contracts/` and `docs/contracts_spec.md`).

## Responsibilities

- Pipeline definition (video in → `BiomechFit` out).
- I/O between stages: writes/reads contract files, never in-memory handoff.
- Container/env selection per stage (Docker built from WSL2, GPU via NVIDIA Container Toolkit).
- Run manifests and provenance tracking.

## Environment

Light: only depends on `mesh2sim-contracts` (for reading/writing contract files) plus a
process/container launcher (subprocess, docker SDK).
