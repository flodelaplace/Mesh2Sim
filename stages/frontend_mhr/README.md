# stages/frontend_mhr

Front-end stage: video frames to MHR mesh estimates.

## Scope

- Wraps Fast-SAM-3D-Body (or a swappable mesh estimator).
- Emits `BodyEstimate` contracts (one per frame, per view).
- Pairs with an adapter that converts `BodyEstimate` to `AnatomicalObservation` via the
  generic MHR-to-anatomy correspondence map (the central project lock — see CLAUDE.md).

## Environment

Heavy: PyTorch + CUDA. **Isolated env**, never mixed with `biomech_bridge` (nimblephysics),
`targets_2d` (mmcv), or any MJX/JAX stage.
