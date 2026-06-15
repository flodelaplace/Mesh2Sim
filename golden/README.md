# golden

Golden fixtures and end-to-end regression artifacts.

Holds small, versioned reference inputs/outputs used to detect regressions across the
pipeline (per `CLAUDE.md`: "Tests d'abord aux frontières de contrat, golden tests pour la
régression de bout en bout").

## Layout (to be filled)

- `inputs/`   — small reference clips, calibration files, expected metadata.
- `outputs/`  — frozen contract files produced by a known-good pipeline run.
- `manifests/` — provenance and run config used to produce the outputs.

Large binary data lives outside the repo (DVC / object storage); only small fixtures live
here.
