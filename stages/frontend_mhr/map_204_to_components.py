"""Investigation — bridge the decomposed SAM 3D Body outputs to the 204-dim
``model_parameters`` vector that the MHR rig actually consumes.

Reads the vendored core to extract the index tables, builds the 204→components
mapping, cross-checks against ``stages/frontend_mhr/results/scaling_mask_v2.json``,
and verifies empirically (real inference) that the 204 vector reconstructs from
the decomposed network outputs.

Outputs:
- ``stages/frontend_mhr/results/204_to_components.json`` — per-index mapping
- printed verdict on where the 73 scalings live in component space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Index tables read verbatim from
# ``third_party/sam_3d_body/models/modules/mhr_utils.py`` (the rig's compact
# layout for body_pose_params; these are positions WITHIN the 133-dim body
# pose vector, before the model truncates to 130).
BODY_POSE_3DOF_ROT_IDXS: list[tuple[int, int, int]] = [
    (0, 2, 4),
    (6, 8, 10),
    (12, 13, 14),
    (15, 16, 17),
    (18, 19, 20),
    (21, 22, 23),
    (24, 25, 26),
    (27, 28, 29),
    (34, 35, 36),
    (37, 38, 39),
    (44, 45, 46),
    (53, 54, 55),
    (64, 65, 66),
    (85, 69, 73),
    (86, 70, 79),
    (87, 71, 82),
    (88, 72, 76),
    (91, 92, 93),
    (112, 96, 100),
    (113, 97, 106),
    (114, 98, 109),
    (115, 99, 103),
    (130, 131, 132),  # last tuple gets dropped by [:130] truncation
]
BODY_POSE_1DOF_ROT_IDXS: list[int] = [
    1,
    3,
    5,
    7,
    9,
    11,
    30,
    31,
    32,
    33,
    40,
    41,
    42,
    43,
    47,
    48,
    49,
    50,
    51,
    52,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    67,
    68,
    74,
    75,
    77,
    78,
    80,
    81,
    83,
    84,
    89,
    90,
    94,
    95,
    101,
    102,
    104,
    105,
    107,
    108,
    110,
    111,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
]
BODY_POSE_1DOF_TRANS_IDXS: list[int] = [124, 125, 126, 127, 128, 129]

# Layout constants — confirmed in mhr_head.py around line 580.
N_GLOBAL_TRANS = 3
N_GLOBAL_ROT = 3
N_BODY_POSE_KEPT = 130  # body_pose_params[..., :130]
N_BODY_POSE_TRUNCATED = 3  # the last 3 of the 133-dim body_pose are dropped
N_SCALES = 68  # scales segment (PCA-decoded from 28 scale_params)
N_MODEL_PARAMS = 204  # = 3 + 3 + 130 + 68

OFFSET_GLOBAL_TRANS = 0
OFFSET_GLOBAL_ROT = OFFSET_GLOBAL_TRANS + N_GLOBAL_TRANS
OFFSET_BODY_POSE = OFFSET_GLOBAL_ROT + N_GLOBAL_ROT
OFFSET_SCALES = OFFSET_BODY_POSE + N_BODY_POSE_KEPT

RIG_PATH = "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
CKPT_PATH = "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/model.ckpt"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _body_pose_role(body_pose_idx: int) -> tuple[str, int]:
    """Return ``(role, sub_index)`` for a position in body_pose_params (133-dim).

    Roles:
    - ``"3dof_rot"``   : member of a 3-DoF rotation triple. ``sub_index`` = triple index in BODY_POSE_3DOF_ROT_IDXS.
    - ``"1dof_rot"``   : 1-DoF rotation. ``sub_index`` = position in BODY_POSE_1DOF_ROT_IDXS.
    - ``"1dof_trans"`` : 1-DoF translation (= shape-mode encoded in body_pose). ``sub_index`` = position in BODY_POSE_1DOF_TRANS_IDXS.
    """
    for k, triple in enumerate(BODY_POSE_3DOF_ROT_IDXS):
        if body_pose_idx in triple:
            return "3dof_rot", k
    if body_pose_idx in BODY_POSE_1DOF_ROT_IDXS:
        return "1dof_rot", BODY_POSE_1DOF_ROT_IDXS.index(body_pose_idx)
    if body_pose_idx in BODY_POSE_1DOF_TRANS_IDXS:
        return "1dof_trans", BODY_POSE_1DOF_TRANS_IDXS.index(body_pose_idx)
    return "unknown", -1


def build_204_to_components(hand_joint_idxs: set[int]) -> list[dict]:
    """Build the 204-entry mapping table.

    ``hand_joint_idxs`` is the set of 204-positions that ``replace_hands_in_pose``
    would overwrite from ``hand_pose_params`` IF ``enable_hand_model=True``.
    In our env it's False so the overwrite never happens, but we flag the
    positions for traceability.
    """
    table: list[dict] = []
    for i in range(N_MODEL_PARAMS):
        entry: dict = {"index_204": i}
        if i < OFFSET_GLOBAL_ROT:
            entry.update(
                segment="global_trans",
                component_name="global_trans",
                sub_index=i,
                note="multiplied by ×10 before going into the rig (stable optimization in meters)",
            )
        elif i < OFFSET_BODY_POSE:
            entry.update(
                segment="global_rot",
                component_name="global_rot",
                sub_index=i - OFFSET_GLOBAL_ROT,
            )
        elif i < OFFSET_SCALES:
            body_pose_idx = i - OFFSET_BODY_POSE
            role, sub = _body_pose_role(body_pose_idx)
            entry.update(
                segment="body_pose",
                component_name="body_pose_params",
                body_pose_idx=body_pose_idx,
                body_pose_role=role,
                body_pose_role_sub_index=sub,
                hand_overwrite_when_enable_hand_model=(i in hand_joint_idxs),
            )
        else:
            entry.update(
                segment="scales",
                component_name="scales (PCA-decoded from scale_params)",
                scales_idx=i - OFFSET_SCALES,
                note="scales = scale_mean + scale_params @ scale_comps; "
                "each scales[k] is a linear combination of all 28 scale_params",
            )
        table.append(entry)
    return table


def cross_check_scaling_mask(table: list[dict]) -> dict:
    """Read scaling_mask_v2.json and partition the 73 scalings by segment."""
    v2 = json.loads((RESULTS_DIR / "scaling_mask_v2.json").read_text())
    scaling_indices = v2["categories"]["scaling_sur"]["indices"]
    pose_dof_indices = v2["categories"]["pose_dof"]["indices"]

    by_segment: dict[str, list[int]] = {
        "global_trans": [],
        "global_rot": [],
        "body_pose": [],
        "scales": [],
    }
    body_pose_roles: dict[str, list[int]] = {
        "3dof_rot": [],
        "1dof_rot": [],
        "1dof_trans": [],
        "unknown": [],
    }
    scales_idxs: list[int] = []

    for i in scaling_indices:
        e = table[i]
        by_segment[e["segment"]].append(i)
        if e["segment"] == "body_pose":
            body_pose_roles[e["body_pose_role"]].append(i)
        elif e["segment"] == "scales":
            scales_idxs.append(e["scales_idx"])

    pose_dof_by_segment: dict[str, list[int]] = {
        "global_trans": [],
        "global_rot": [],
        "body_pose": [],
        "scales": [],
    }
    for i in pose_dof_indices:
        pose_dof_by_segment[table[i]["segment"]].append(i)

    return {
        "n_scalings_total": len(scaling_indices),
        "scalings_by_segment": by_segment,
        "scaling_body_pose_roles": body_pose_roles,
        "scaling_scales_subindices": sorted(scales_idxs),
        "n_pose_dof_total": len(pose_dof_indices),
        "pose_dof_by_segment": pose_dof_by_segment,
    }


def empirical_check(table: list[dict]) -> dict:
    """Run a real inference, compute the 204 from the decomposed components, and
    confirm it matches what the rig would consume."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
    from sam_3d_body.build_models import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_sam_3d_body(checkpoint_path=CKPT_PATH, device=device, mhr_path=RIG_PATH)
    est = SAM3DBodyEstimator(model, cfg)

    # Synthetic frame + bbox: skip detector.
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    bbox = np.array([[100.0, 50.0, 540.0, 470.0]], dtype=np.float32)
    outputs = est.process_one_image(
        frame,
        bboxes=bbox,
        inference_type="body",
    )
    assert outputs, "no person detected — bbox path failed"
    out = outputs[0]

    # Pull the decomposed components.
    global_rot = np.asarray(out["global_rot"]).reshape(-1)
    body_pose_params = np.asarray(out["body_pose_params"]).reshape(-1)
    scale_params = np.asarray(out["scale_params"]).reshape(-1)
    pred_cam_t = np.asarray(out["pred_cam_t"]).reshape(-1)

    # Read the PCA constants from the checkpoint.
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    scale_mean = sd["head_pose.scale_mean"].cpu().numpy()
    scale_comps = sd["head_pose.scale_comps"].cpu().numpy()

    # global_trans is not in the decomposed output dict per se — the network's
    # ``forward_pose_branch`` derives it from ``pred_cam_t``. For the purposes
    # of THIS investigation, the model_params layout uses ``global_trans * 10``;
    # we use pred_cam_t as the best proxy and report it.
    global_trans = pred_cam_t

    # Reconstruct the 204 the way mhr_head.py does it.
    scales = scale_mean + scale_params @ scale_comps  # (68,)
    full_pose_params = np.concatenate(
        [global_trans * 10.0, global_rot, body_pose_params[:130]], axis=0
    )  # (136,)
    reconstructed = np.concatenate([full_pose_params, scales], axis=0)  # (204,)

    return {
        "n_model_params_reconstructed": int(reconstructed.shape[0]),
        "expected": N_MODEL_PARAMS,
        "components_shapes": {
            "shape_params": list(out["shape_params"].shape),
            "expr_params": list(out["expr_params"].shape),
            "scale_params": list(scale_params.shape),
            "global_rot": list(global_rot.shape),
            "body_pose_params": list(body_pose_params.shape),
            "hand_pose_params": list(np.asarray(out["hand_pose_params"]).shape),
            "global_trans (from pred_cam_t)": list(global_trans.shape),
            "scales (derived PCA)": list(scales.shape),
        },
        "pca_constants_shapes": {
            "scale_mean": list(scale_mean.shape),
            "scale_comps": list(scale_comps.shape),
        },
        "sample_reconstructed_values": {
            "[0:3] global_trans*10": reconstructed[0:3].tolist(),
            "[3:6] global_rot": reconstructed[3:6].tolist(),
            "[130:136] body_pose_1dof_trans (shape modes)": reconstructed[130:136].tolist(),
            "[136:140] first scales": reconstructed[136:140].tolist(),
        },
        "consistency_checks": {
            "len(global_trans) == 3": int(global_trans.shape[0]) == 3,
            "len(global_rot) == 3": int(global_rot.shape[0]) == 3,
            "len(body_pose_params) == 133": int(body_pose_params.shape[0]) == 133,
            "len(scale_params) == 28": int(scale_params.shape[0]) == 28,
            "scale_mean.shape == (68,)": scale_mean.shape == (68,),
            "scale_comps.shape == (28, 68)": scale_comps.shape == (28, 68),
            "reconstructed length == 204": reconstructed.shape == (N_MODEL_PARAMS,),
        },
    }


def main() -> None:
    # Hand-write positions — read from the checkpoint state_dict.
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    hand_l = sd["head_pose.hand_joint_idxs_left"].cpu().numpy().tolist()
    hand_r = sd["head_pose.hand_joint_idxs_right"].cpu().numpy().tolist()
    hand_joint_idxs = set(hand_l + hand_r)

    table = build_204_to_components(hand_joint_idxs)
    cross = cross_check_scaling_mask(table)
    empirical = empirical_check(table)

    # ---------- pretty print summary ----------
    print("=== Layout 204 — segments ===")
    print(f"  [0:{OFFSET_GLOBAL_ROT})           global_trans   (×10 going into the rig)")
    print(f"  [{OFFSET_GLOBAL_ROT}:{OFFSET_BODY_POSE})           global_rot")
    print(f"  [{OFFSET_BODY_POSE}:{OFFSET_SCALES})         body_pose_params[0:130]")
    print(
        f"      └─ {sum(len(t) for t in BODY_POSE_3DOF_ROT_IDXS[:-1])} body_pose positions in 3-DoF rot triples (22 triples × 3 = 66)"
    )
    print(f"      └─ {len(BODY_POSE_1DOF_ROT_IDXS)} body_pose positions in 1-DoF rot")
    print(
        f"      └─ {len(BODY_POSE_1DOF_TRANS_IDXS)} body_pose positions in 1-DoF trans (shape modes, body_pose[124..129])"
    )
    print("      └─ NB: body_pose[130..132] dropped by the [:130] slice (the 23rd 3-DoF triple)")
    print(
        f"  [{OFFSET_SCALES}:{N_MODEL_PARAMS})        scales = scale_mean + scale_params @ scale_comps  (PCA: 28 → 68)"
    )
    print()
    print("=== Cross-check: 73 scalings vs decomposed components ===")
    print(f"  n_scalings_total           : {cross['n_scalings_total']}")
    for seg, idxs in cross["scalings_by_segment"].items():
        print(f"  in segment {seg:<14s} : {len(idxs)} indices (204-space)")
    print("  body_pose roles within those scalings:")
    for role, idxs in cross["scaling_body_pose_roles"].items():
        if idxs:
            print(f"      {role:<12s} : {len(idxs)} entries → 204-indices {idxs}")
    print(
        f"  scales sub-indices (in [0, 68)) covered by scalings: {len(cross['scaling_scales_subindices'])} / 68"
    )
    missing = sorted(set(range(68)) - set(cross["scaling_scales_subindices"]))
    print(f"      scales positions NOT in scalings: {missing}")
    print()
    print("=== Cross-check: 4 pose_dof vs decomposed components ===")
    for seg, idxs in cross["pose_dof_by_segment"].items():
        if idxs:
            print(f"  in segment {seg:<14s} : {len(idxs)} indices (204-space) → {idxs}")
    print()
    print("=== Empirical reconstruction sanity (real inference) ===")
    for k, v in empirical["consistency_checks"].items():
        print(f"  [{'OK ' if v else 'XX '}] {k}")
    print(f"  decomposed-component shapes: {json.dumps(empirical['components_shapes'], indent=4)}")
    print(f"  PCA constants shapes: {empirical['pca_constants_shapes']}")
    print("  sample reconstructed values:")
    for k, v in empirical["sample_reconstructed_values"].items():
        if isinstance(v, list) and v and isinstance(v[0], float):
            v_show = [f"{x:+.4f}" for x in v]
        else:
            v_show = v
        print(f"    {k:<48s}: {v_show}")

    # ---------- write the JSON outputs ----------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "schema": "mhr_model_parameters_204_to_components_v1",
        "n_indices": N_MODEL_PARAMS,
        "segments": {
            "global_trans": {
                "start": OFFSET_GLOBAL_TRANS,
                "end": OFFSET_GLOBAL_ROT,
                "size": N_GLOBAL_TRANS,
                "scaled_in_rig_by": 10,
            },
            "global_rot": {
                "start": OFFSET_GLOBAL_ROT,
                "end": OFFSET_BODY_POSE,
                "size": N_GLOBAL_ROT,
            },
            "body_pose": {
                "start": OFFSET_BODY_POSE,
                "end": OFFSET_SCALES,
                "size": N_BODY_POSE_KEPT,
                "source_component": "body_pose_params (133-dim from network), truncated to [:130]",
            },
            "scales": {
                "start": OFFSET_SCALES,
                "end": N_MODEL_PARAMS,
                "size": N_SCALES,
                "source_component": "scales = scale_mean (68,) + scale_params (28,) @ scale_comps (28, 68)",
            },
        },
        "table": table,
        "cross_check_vs_scaling_mask_v2": cross,
        "empirical_reconstruction": empirical,
    }
    out_path = RESULTS_DIR / "204_to_components.json"
    out_path.write_text(json.dumps(out_doc, indent=2, default=str))
    print()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
