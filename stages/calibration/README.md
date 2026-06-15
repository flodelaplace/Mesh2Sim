# stages/calibration

Camera calibration and world-frame setup.

## Scope

- Intrinsics (K, distortion) and extrinsics (R, t) for each view.
- World frame: Y-up, right-handed, consistent with OpenSim.
- Time offsets per view.
- Reuses `CoordinateTransformer` (MoGe / lean variants) from the FastSAM3DToOpenSim repo as
  starting point — do **not** copy that whole repo, just lift the relevant components.

## Environment

Light: numpy, opencv. Standalone venv.
