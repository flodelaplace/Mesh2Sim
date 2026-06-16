"""Disambiguate the 10 empirical-only indices: morphology (shape) vs pose DoF (movement)?

Recap of upstream finding
-------------------------
``investigate_mhr_params.py`` flagged 10 indices that change parent-child distances when
perturbed but are NOT labelled ``scaling`` by Momentum (they're labelled ``pose`` and/or
``rigid``): 0, 1, 2, 62, 130, 131, 132, 133, 134, 135. We need a ruling: is each one a
true morphological scaling we should propagate to OpenSim, or a per-joint translation
DoF that moves during the motion (and must NOT be baked into segment lengths)?

Method
------
**Primary discriminator — parameter_transform structure** (decisive).

``character_torch.parameter_transform.parameter_transform`` is a constant linear map of
shape ``(889, 249)`` with the layout ``output[joint_k * 7 + dof] = sum_i M[..., i] * input[i]``
where ``dof ∈ {tx, ty, tz, rx, ry, rz, scale}``. By inspecting which output rows each
input column drives, we recover what the parameter "is":

- ``scale`` row of joint(s)        → bone-scale factor (rare in practice)
- coordinated ``tx/ty/tz`` rows on **many** joints with PCA-like weights
                                   → shape mode (rest-pose morphology variation)
- single ``tx/ty/tz`` row on **one** joint with unit/round weight
                                   → joint translation DoF (pose, moves during motion)
- single ``rx/ry/rz`` row          → joint rotation DoF (pose)

This labelling is independent of any internal Momentum mask — it's a direct reading of
the linear-algebra structure.

**Secondary discriminator — pose stability** (sanity check).

For both shape modes and translation DoFs, the bone-length change is invariant under
upstream rotations (lengths are local), so the discriminator above already captures the
distinction. Pose stability is run anyway to catch any non-linearity from
``apply_correctives`` couplings or hidden non-linear pieces in the rig that would make a
parameter's effect drift with pose. A parameter whose Σ|Δ‖edge‖| varies > 5% across
poses gets flagged ``ambiguous``.

Outputs
-------
- ``results/disambiguation.json``     — per-suspect verdict + structural signature
- ``results/scaling_mask_v2.json``    — updated mask with three groups
                                        (``scaling_sur``, ``pose_dof``, ``a_decider``)
- ``results/pose_stability.png``      — per-suspect effect at each pose, side-by-side
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_MHR_PATH = Path(
    "/home/fdela/FastSAM3DToOpenSim/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
N_MODEL_PARAMS = 204
N_IDENTITY = 45
N_FACE_EXPR = 72
N_JOINTS = 127
N_DOFS_PER_JOINT = 7
DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz", "scale"]

# The 10 indices the upstream investigation flagged as "empirical-only scaling".
SUSPECTS = [0, 1, 2, 62, 130, 131, 132, 133, 134, 135]
# Stability threshold: fraction of mean effect that the effect is allowed to drift
# across poses before we call the index ambiguous.
POSE_DRIFT_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# Parameter-transform structural inspection
# ---------------------------------------------------------------------------


@dataclass
class StructuralSignature:
    n_nonzero_rows: int
    affected_joints: list[int]
    dof_types: list[str]  # unique DOF types affected, in order
    rows: list[tuple[int, str, float]]  # (joint, dof, weight), sorted by |weight| desc

    def is_single_joint_single_dof(self) -> bool:
        return self.n_nonzero_rows == 1

    def is_pure_scale_dof(self) -> bool:
        """All affected rows are scale outputs (rare in MHR but possible)."""
        return self.n_nonzero_rows > 0 and set(self.dof_types) == {"scale"}

    def is_coordinated_translation_pattern(self) -> bool:
        """Multiple joints, all translation DOFs — looks like a shape PCA mode."""
        return (
            self.n_nonzero_rows >= 2
            and len(self.affected_joints) >= 2
            and set(self.dof_types).issubset({"tx", "ty", "tz"})
        )


def signature(transform: np.ndarray, col: int) -> StructuralSignature:
    """Decode column ``col`` of the (889, 249) parameter_transform."""
    column = transform[:, col]
    nz = np.where(np.abs(column) > 1e-12)[0]
    rows: list[tuple[int, str, float]] = []
    joints: set[int] = set()
    dofs: list[str] = []
    for r in nz:
        joint = int(r) // N_DOFS_PER_JOINT
        dof = DOF_NAMES[int(r) % N_DOFS_PER_JOINT]
        weight = float(column[r])
        rows.append((joint, dof, weight))
        joints.add(joint)
        if dof not in dofs:
            dofs.append(dof)
    rows.sort(key=lambda t: -abs(t[2]))
    return StructuralSignature(
        n_nonzero_rows=int(nz.size),
        affected_joints=sorted(joints),
        dof_types=dofs,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Pose-stability test
# ---------------------------------------------------------------------------


def make_poses(rotation_indices: np.ndarray, n_poses: int) -> list[torch.Tensor]:
    """Build deterministic non-neutral pose vectors.

    Pose 0 is always neutral. The other poses are constructed by setting a fixed subset
    of *rotation* parameter slots to specific angles, so the rig is bent but the
    perturbation we're studying still runs on top of a well-defined configuration.
    """
    rng = np.random.default_rng(seed=0)
    poses = [torch.zeros(1, N_MODEL_PARAMS)]  # neutral
    for k in range(1, n_poses):
        v = np.zeros(N_MODEL_PARAMS, dtype=np.float32)
        # Pick a stable per-pose subset of the rotation indices, set to magnitudes near 0.3 rad
        chosen = rng.choice(rotation_indices, size=min(20, rotation_indices.size), replace=False)
        v[chosen] = (rng.random(chosen.size).astype(np.float32) * 2 - 1) * 0.3 + 0.1 * k
        poses.append(torch.from_numpy(v).unsqueeze(0))
    return poses


@torch.no_grad()
def forward_joints(module: torch.jit.ScriptModule, mp: torch.Tensor, device: torch.device) -> np.ndarray:
    identity = torch.zeros(1, N_IDENTITY, device=device)
    face = torch.zeros(1, N_FACE_EXPR, device=device)
    _v, skel = module(identity, mp.to(device), face, False)
    return skel[0, :, :3].cpu().numpy()


def parent_child_lengths(joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    child = np.where(parents >= 0)[0]
    par = parents[child]
    diff = joints[child] - joints[par]
    return np.linalg.norm(diff, axis=-1)


def measure_effect_at_pose(
    module, base_pose: torch.Tensor, parents: np.ndarray, indices: list[int], eps: float, device: torch.device
) -> dict[int, float]:
    ref_joints = forward_joints(module, base_pose, device)
    ref_lengths = parent_child_lengths(ref_joints, parents)
    out = {}
    for i in indices:
        mp = base_pose.clone()
        mp[0, i] += eps
        new_joints = forward_joints(module, mp, device)
        new_lengths = parent_child_lengths(new_joints, parents)
        out[i] = float(np.abs(new_lengths - ref_lengths).sum())
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    idx: int
    structural_label: str  # "pose_translation_dof", "pose_rotation_dof", "shape_mode",
                           # "pure_scale", "unknown_structure"
    structural_reason: str
    pose_drift_rel: float  # max |effect_pose - effect_neutral| / mean(effect)
    final: str  # "scaling_sur", "pose_dof", "a_decider"
    signature: StructuralSignature = field(repr=False)


def classify_structural(sig: StructuralSignature) -> tuple[str, str]:
    if sig.n_nonzero_rows == 0:
        return "unknown_structure", "no non-zero entries in parameter_transform column"
    if sig.is_single_joint_single_dof():
        joint, dof, w = sig.rows[0]
        if dof in {"tx", "ty", "tz"}:
            return "pose_translation_dof", f"single output: joint {joint} {dof} (weight {w:+.3f})"
        if dof in {"rx", "ry", "rz"}:
            return "pose_rotation_dof", f"single output: joint {joint} {dof} (weight {w:+.3f})"
        if dof == "scale":
            return "pure_scale", f"single output: joint {joint} scale (weight {w:+.3f})"
    if sig.is_pure_scale_dof():
        return "pure_scale", f"all {sig.n_nonzero_rows} outputs are scale rows"
    if sig.is_coordinated_translation_pattern():
        return (
            "shape_mode",
            f"coordinated {sig.n_nonzero_rows} translation rows across "
            f"{len(sig.affected_joints)} joints",
        )
    return (
        "mixed",
        f"{sig.n_nonzero_rows} non-zero rows across {len(sig.affected_joints)} joints, "
        f"DOFs: {sig.dof_types}",
    )


def final_decision(structural: str, drift_rel: float) -> str:
    # For single-joint single-DoF parameters, the structural reading is decisive: a single
    # output row on a translation or rotation DoF IS a pose parameter by construction.
    # The pose-drift check is meaningless here because the absolute effect on bone lengths
    # is float32 noise (rotation DoFs preserve lengths analytically; the tiny non-zero
    # numerical readings just reflect float chain round-off).
    if structural in {"pose_translation_dof", "pose_rotation_dof"}:
        return "pose_dof"
    # For shape modes (coordinated multi-joint translations) and pure-scale rows, the
    # absolute effect IS real geometry and pose-stability is the relevant sanity check.
    if structural in {"shape_mode", "pure_scale"}:
        return "scaling_sur" if drift_rel <= POSE_DRIFT_TOLERANCE else "a_decider"
    # Mixed structures (e.g. translation + rotation, or many DoF types) are inherently
    # ambiguous — escalate to manual review.
    return "a_decider"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_pose_stability(
    pose_effects: dict[int, list[float]], pose_labels: list[str], out_path: Path
) -> None:
    indices = sorted(pose_effects)
    n = len(indices)
    width = 0.8 / len(pose_labels)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, label in enumerate(pose_labels):
        vals = [pose_effects[i][j] for i in indices]
        ax.bar(x + j * width, vals, width=width, label=label)
    ax.set_xticks(x + width * (len(pose_labels) - 1) / 2)
    ax.set_xticklabels([str(i) for i in indices])
    ax.set_xlabel("suspect parameter index")
    ax.set_ylabel("Σ |Δ‖parent-child‖|  (ε fixed)")
    ax.set_title("Pose-stability test on the 10 suspect indices")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR_PATH)
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n-poses", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[setup] device={device} mhr={args.mhr_path}")

    module = torch.jit.load(str(args.mhr_path), map_location=device).eval()
    ct = module.character_torch
    parents = ct.skeleton.joint_parents.cpu().numpy().astype(np.int64)
    transform = ct.parameter_transform.parameter_transform.cpu().numpy()
    internal_scaling = ct.parameter_transform.scaling_parameters.cpu().numpy().astype(bool)
    print(
        f"[setup] parameter_transform={transform.shape}, "
        f"internal scalings in [0..203] = {int(internal_scaling[:N_MODEL_PARAMS].sum())}"
    )

    # Load upstream mask
    upstream = json.loads(
        (args.results_dir / "scaling_mask.json").read_text(encoding="utf-8")
    )
    print(
        f"[upstream] scaling={len(upstream['scaling_indices'])}, "
        f"rotation={len(upstream['rotation_indices'])}, "
        f"unused={len(upstream['unused_indices'])}"
    )

    rotation_indices = np.asarray(upstream["rotation_indices"], dtype=np.int64)

    # 1. Structural signatures for all 204 indices (for the mask update). Verdicts only
    #    for the 10 suspects but we report signature totals for diagnostics.
    print("\n=== structural signatures: 10 suspect indices ===")
    sigs = {i: signature(transform, i) for i in range(N_MODEL_PARAMS)}
    suspects_verdicts: dict[int, Verdict] = {}
    for i in SUSPECTS:
        s = sigs[i]
        slabel, sreason = classify_structural(s)
        top = ", ".join(f"j{j}.{d}={w:+.2f}" for j, d, w in s.rows[:4])
        print(f"  idx={i:>3d}  [{slabel:<23s}]  rows={s.n_nonzero_rows}  → {top}")
        suspects_verdicts[i] = Verdict(
            idx=i,
            structural_label=slabel,
            structural_reason=sreason,
            pose_drift_rel=float("nan"),
            final="",  # filled below
            signature=s,
        )

    # 2. Pose-stability test for those 10 suspects
    print("\n=== pose-stability test ===")
    poses = make_poses(rotation_indices, args.n_poses)
    pose_labels = ["neutral"] + [f"pose{k}" for k in range(1, args.n_poses)]
    per_pose_effects: dict[int, list[float]] = {i: [] for i in SUSPECTS}
    for j, pose in enumerate(poses):
        effects_at_pose = measure_effect_at_pose(module, pose, parents, SUSPECTS, args.eps, device)
        for i, e in effects_at_pose.items():
            per_pose_effects[i].append(e)
        print(f"  {pose_labels[j]:>10s} :", {i: f"{e:.4f}" for i, e in effects_at_pose.items()})

    # Compute drift per index
    for i in SUSPECTS:
        effects = np.asarray(per_pose_effects[i])
        mean_e = max(effects.mean(), 1e-12)
        drift = float(np.abs(effects - effects[0]).max() / mean_e)
        suspects_verdicts[i].pose_drift_rel = drift
        suspects_verdicts[i].final = final_decision(suspects_verdicts[i].structural_label, drift)

    # 3. Print verdict table
    print("\n=== VERDICT ===")
    print(f"{'idx':>4s} {'structural':<25s} {'drift':>8s}  {'final':<13s} reason")
    for i, v in suspects_verdicts.items():
        print(
            f"{i:>4d} {v.structural_label:<25s} {v.pose_drift_rel:>8.4f}  {v.final:<13s} {v.structural_reason}"
        )

    # 4. Save disambiguation JSON
    disamb = {
        "suspects": [
            {
                "idx": v.idx,
                "structural_label": v.structural_label,
                "structural_reason": v.structural_reason,
                "rows": [{"joint": j, "dof": d, "weight": w} for j, d, w in v.signature.rows],
                "pose_drift_rel": v.pose_drift_rel,
                "per_pose_effects": dict(zip(pose_labels, per_pose_effects[v.idx], strict=True)),
                "final": v.final,
            }
            for v in suspects_verdicts.values()
        ],
        "pose_drift_tolerance": POSE_DRIFT_TOLERANCE,
        "eps": args.eps,
    }
    (args.results_dir / "disambiguation.json").write_text(json.dumps(disamb, indent=2))

    # 5. Update scaling_mask with 3 categories
    scaling_sur: list[int] = []   # 67 sure scalings (everything previously empirical *minus* suspects flagged pose) + suspects re-classified as scaling
    pose_dof: list[int] = []
    a_decider: list[int] = []
    suspects_set = set(SUSPECTS)
    for i in upstream["scaling_indices"]:
        if i in suspects_set:
            v = suspects_verdicts[i]
            if v.final == "scaling_sur":
                scaling_sur.append(i)
            elif v.final == "pose_dof":
                pose_dof.append(i)
            else:
                a_decider.append(i)
        else:
            scaling_sur.append(i)

    mask_v2 = {
        "schema": "mhr_model_parameters_204_classification_v2",
        "n_params": N_MODEL_PARAMS,
        "derived_from": "scaling_mask.json (upstream empirical) + parameter_transform structure"
                       " + pose-stability sanity check",
        "categories": {
            "scaling_sur": {
                "count": len(scaling_sur),
                "definition": "morphological shape parameter (multi-joint coordinated translation or scale row) "
                             "that changes parent-child distances and is pose-stable. "
                             "Propagate to OpenSim segment scaling.",
                "indices": sorted(scaling_sur),
            },
            "pose_dof": {
                "count": len(pose_dof),
                "definition": "joint translation/rotation DoF — moves DURING motion. "
                             "Must NOT be baked into segment lengths.",
                "indices": sorted(pose_dof),
            },
            "a_decider": {
                "count": len(a_decider),
                "definition": "structurally ambiguous or pose-drift exceeds tolerance — needs manual review.",
                "indices": sorted(a_decider),
            },
        },
        "pose_dof_tolerance_rel": POSE_DRIFT_TOLERANCE,
        "disambiguation_details": "see disambiguation.json",
    }
    (args.results_dir / "scaling_mask_v2.json").write_text(json.dumps(mask_v2, indent=2))

    plot_pose_stability(per_pose_effects, pose_labels, args.results_dir / "pose_stability.png")
    print(
        f"\n[mask v2] scaling_sur={len(scaling_sur)}  pose_dof={len(pose_dof)}  a_decider={len(a_decider)}"
    )
    print(f"[saved] {args.results_dir}/{{disambiguation.json, scaling_mask_v2.json, pose_stability.png}}")


if __name__ == "__main__":
    main()
