"""Extract the MHR scale-PCA constants from the SAM 3D Body model checkpoint.

Output: a small npz (~8 KB) containing the buffers required by T4 (shape-lock) to
reconstruct the 68-dim ``scales`` vector from the 28-dim ``scale_params`` predicted by
inference. The buffers live in the head_pose submodule of the SAM 3D Body model:

    scales = scale_mean + scale_params @ scale_comps

where ``scale_mean.shape == (68,)`` and ``scale_comps.shape == (28, 68)``.

These are constants for the model (do not change between subjects), and our T4 stage
needs them but cannot afford to load the 2 GB ``model.ckpt`` every run. We pre-extract
them once and ship the resulting npz next to the other frontend reference data.

Re-run this script when the vendored sam_3d_body or the model.ckpt changes:

    python stages/frontend_mhr/scripts/extract_scale_pca.py \
        --checkpoint /path/to/sam-3d-body-dinov3/model.ckpt \
        --out stages/frontend_mhr/reference/mhr_scale_pca.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_KEY_MEAN = "head_pose.scale_mean"
_KEY_COMPS = "head_pose.scale_comps"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/model.ckpt"),
        help="path to the SAM 3D Body model.ckpt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reference" / "mhr_scale_pca.npz",
        help="destination npz",
    )
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    for key in (_KEY_MEAN, _KEY_COMPS):
        if key not in sd:
            raise SystemExit(
                f"missing key {key!r} in checkpoint state_dict. The checkpoint "
                "may have a different layout than expected."
            )

    scale_mean = sd[_KEY_MEAN].detach().cpu().numpy().astype(np.float32)
    scale_comps = sd[_KEY_COMPS].detach().cpu().numpy().astype(np.float32)

    expected_mean = (68,)
    expected_comps = (28, 68)
    if scale_mean.shape != expected_mean:
        raise SystemExit(f"scale_mean shape {scale_mean.shape} != expected {expected_mean}")
    if scale_comps.shape != expected_comps:
        raise SystemExit(f"scale_comps shape {scale_comps.shape} != expected {expected_comps}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        scale_mean=scale_mean,
        scale_comps=scale_comps,
        source_checkpoint=str(args.checkpoint),
        source_key_mean=_KEY_MEAN,
        source_key_comps=_KEY_COMPS,
        note=np.asarray(
            "scales = scale_mean + scale_params @ scale_comps; "
            "see CLAUDE.md, section 'Paramètres MHR et chaîne de scaling'."
        ),
    )

    print(f"[OK] wrote {args.out}")
    print(f"     scale_mean  shape={scale_mean.shape}  dtype={scale_mean.dtype}")
    print(f"     scale_comps shape={scale_comps.shape}  dtype={scale_comps.dtype}")
    print(f"     source: {args.checkpoint}")


if __name__ == "__main__":
    main()
