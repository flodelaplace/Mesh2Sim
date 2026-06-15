# stages/validation

Validation harness: comparison of markerless joint angles against the optoelectronic
reference.

## Scope

- 200 subjects target (young + elderly, gait + sit-to-stand).
- Method-agreement framing (Bland-Altman), **not** "ground truth" — opto has soft-tissue
  artefact too.
- Per-DoF, per-task accuracy targets (no uniform 5° threshold).
- Includes a learned-bridge baseline (BioPose-style) for comparison.

## Environment

Light: numpy, scipy, pandas, matplotlib. Standalone venv.
