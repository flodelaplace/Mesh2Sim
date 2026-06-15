# stages/targets_2d

2D anatomical targets used to supervise mesh fit and/or seed adapters.

## Scope

- SynthPose-style 2D keypoint detection on input frames.
- Emits `Keypoints2D` payloads consumed via `BodyEstimate`/`AnatomicalObservation`.

## Environment

Heavy: mmcv / mmpose ecosystem. **Isolated env** (mmcv pins clash with most other stages).
