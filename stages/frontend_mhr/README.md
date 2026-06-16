# stages/frontend_mhr

Front-end stage: video frames → MHR mesh estimates, emitted as serialised
``mesh2sim.contracts.BodyEstimate``.

## Scope

- Wraps the **vendored** Fast-SAM-3D-Body inference core (see
  ``third_party/VENDORED.md``) behind the swappable ``MeshEstimator`` interface.
- Emits ``BodyEstimate`` (one per frame, per view) — pure pass-through of the
  estimator's native output, no coordinate transform, no scaling, no anatomy.
- The MHR-to-anatomy bridge (correspondence map → ``AnatomicalObservation``)
  is downstream.

## Discipline rules

- **Never import ``sam_3d_body.visualization``** from this stage's code. The
  frontend never draws — it serialises. Importing the visualization submodule
  would pull in ``pyrender`` (purposefully not in our dependency list) and
  fail at import time. Verified via ``grep``: the inference core
  (``__init__``, ``sam_3d_body_estimator``, ``build_models``, ``models/``)
  does not import ``visualization`` transitively, so the rule is just
  "don't add it".

## Environment

CUDA 12.4.1 + Python 3.11. Conda env name: ``mesh2sim-frontend-mhr``.
Strict isolation:
- **Never** install OpenSim, nimblephysics, MoGe, mmcv, JAX/MJX in this env.
- Production deps lifted from ``FastSAM3DToOpenSim/docker/requirements_docker.txt``
  at the pinned vendor commit, minus ``vtk``/``pygltflib``/``pyrender``
  (downstream / visualization, not needed for serialised output).

## Contract debt — flagged for later

The ``BodyEstimate`` schema in ``mesh2sim.contracts`` has **no slot** for
pipeline-side per-detection metadata (e.g. ``track_id``, source bounding box,
detector confidence). For now:

- **Mono-subject mode (default)**: the estimator selects the largest-bbox
  person per frame and emits a single ``BodyEstimate``. ``track_id`` is moot.
- **Multi-person mode (opt-in via ``main_subject_only=False``)**: emits one
  ``BodyEstimate`` per detection. They are NOT linked across frames — no
  ``track_id`` exists in the contract. Use with care; multi-person tracking
  requires a contract extension first.

When we genuinely need multi-person tracking, the right move is to add a
clean field to ``BodyEstimate`` (likely ``detection_id: int | None`` or a
nested ``DetectionMeta``), bump the schema_version minor, and update the
adapter — **NOT** to stash track ids inside ``native_params`` (which is
reserved for opaque model parameters).

## Code layout

```
stages/frontend_mhr/
├── src/mesh2sim_frontend_mhr/
│   ├── interface.py           # MeshEstimator Protocol
│   ├── adapter.py             # sam_3d_body output dict → BodyEstimate
│   └── estimator.py           # FastSAM3DBodyEstimator wraps the vendored core
├── third_party/               # vendored vanilla sam_3d_body
├── tests/                     # mocked unit tests (no GPU)
├── environment.yml            # conda dev env
├── Dockerfile                 # prod container
└── pyproject.toml
```
