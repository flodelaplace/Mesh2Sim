# Vendored: sam_3d_body (Fast SAM 3D Body inference core)

Vanilla, unmodified copy of the ``sam_3d_body`` Python package from
``yangtiming/Fast-SAM-3D-Body``, used by the ``frontend_mhr`` stage to run
mesh estimation. Mesh2Sim does not depend on any live upstream repository;
this directory is the authoritative copy.

## Source

- Upstream repo       : https://github.com/yangtiming/Fast-SAM-3D-Body
- Upstream of upstream: https://github.com/facebookresearch/sam-3d-body
  (Meta SAM 3D Body — original research release)
- Commit pinned       : 936894c37e51de9918012bcbc9ba2d9c20f73252
- Commit date         : 2026-05-28
- Commit subject      : "Clarify IMG_SIZE comment in run_demo.sh"
- Vendored on         : 2026-06-16

## Scope

Only ``sam_3d_body/`` was copied (48 ``.py`` files, no binaries or data assets).
The rest of the upstream repo (demos, notebooks, ``mhr2smpl/``, ``mocap/``,
TensorRT conversion scripts, RealSense capture tools) was deliberately
excluded because it is not part of the inference core.

The ``sam_3d_body/export/`` subdirectory present in some downstream forks
(``AitorIriondo/FastSAM3DToOpenSim``, ``flodelaplace/FastSAM3DToOpenSim``)
contains the OpenSim bridge. It is also excluded — that lives in a separate
Mesh2Sim stage (``biomech_bridge``), not here.

## Patches NOT applied

We took vanilla ``yangtiming`` at the pinned commit and did NOT apply any
of the known downstream patches:

| Patch                                                              | Origin       | Why ignored                                |
| ------------------------------------------------------------------ | ------------ | ------------------------------------------ |
| ``visualization/renderer.py`` lazy ``pyrender`` import             | AitorIriondo | WSL2 + Linux Docker, EGL works as-is       |
| ``visualization/utils.py`` ``LazyConfig`` import promoted top-level | flodelaplace | Cosmetic; detectron2 is not a frontend dep |

## License & attribution

The vendored code is under the MIT License of ``Fast-SAM-3D-Body``
(see ``LICENSE`` next to this file). Each file retains its original
``# Copyright (c) Meta Platforms, Inc. and affiliates.`` header from
Meta's SAM 3D Body research release.

Credits:
- Meta SAM 3D Body authors (research code)
- yangtiming / Fast SAM 3D Body Authors (packaged inference release)

## How to refresh

When a newer ``yangtiming/Fast-SAM-3D-Body`` is desired:

1. Clone the upstream repo at the chosen commit.
2. Replace this ``sam_3d_body/`` directory with the new ``sam_3d_body/`` from
   that commit, verbatim.
3. Update the "Commit pinned" / "Commit date" / "Vendored on" lines above.
4. Re-run the frontend stage tests to catch any breaking API changes.
5. Commit as a single ``vendor: bump sam_3d_body to <SHA>`` commit.

Never edit files inside ``sam_3d_body/`` directly. If a patch is genuinely
needed, document it here AND apply it in a separate, clearly-named commit
so the next refresh can re-apply or reconsider it.

## Verification

The vendoring was checksum-verified at copy time. Every ``.py`` file under
``sam_3d_body/`` plus ``LICENSE`` matches the upstream ``sha256`` byte-for-byte.
To re-verify against the pinned commit:

```bash
# Clone upstream at pinned SHA
git clone https://github.com/yangtiming/Fast-SAM-3D-Body /tmp/yang
git -C /tmp/yang checkout 936894c37e51de9918012bcbc9ba2d9c20f73252

# Diff against the vendored copy — must be empty
diff -r /tmp/yang/sam_3d_body $(dirname "$0")/sam_3d_body
```
