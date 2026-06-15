# stages/biomech_bridge

Biomechanics bridge: from `AnatomicalTrajectory` to joint angles via an OpenSim model.

## Scope

- Marker registration on the OpenSim model: scaling + marker offsets + IK.
- Per-subject residual on top of the frozen generic MHR-to-anatomy map.
- Emits `BiomechFit` contracts (scaled `.osim`, `.mot`, residuals, optional uncertainty).
- Single-model validation rule: markerless and reference fits use the *same* OpenSim model.

## Environment

Heavy & fragile: **nimblephysics** (Python 3.9, frozen late-2024). **Isolated env**, never
mixed with PyTorch-CUDA stages. MuJoCo/MJX kept as a GPU fallback in its own env.
