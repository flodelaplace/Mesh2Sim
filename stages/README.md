# Stages

Each subdirectory is an independent processing stage with its **own** Python environment
(see `CLAUDE.md`: dependencies between stages are mutually incompatible). Stages communicate
only through serialized contracts (see `contracts/` and `docs/contracts_spec.md`); they
**never** import from each other.

| Stage              | Role                                                              |
| ------------------ | ----------------------------------------------------------------- |
| `frontend_mhr`     | Video to MHR mesh estimation (SAM 3D Body adapter)                |
| `targets_2d`       | 2D anatomical targets (e.g. SynthPose) for fit supervision        |
| `fusion_multiview` | Multi-view fusion of `AnatomicalObservation` into a trajectory    |
| `calibration`      | Camera intrinsics/extrinsics; world frame setup                   |
| `biomech_bridge`   | Marker registration on OpenSim model (scaling + offsets + IK)     |
| `validation`       | Comparison against optoelectronic reference (Bland-Altman, etc.)  |
