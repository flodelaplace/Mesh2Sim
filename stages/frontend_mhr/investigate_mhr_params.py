"""Empirically classify MHR ``model_parameters`` (204 indices) into scalings vs rotations.

Method
------
1. Load the TorchScript MHR rig (``mhr_model.pt``) which sits at the heart of SAM 3D Body
   — same module used by ``FastSAM3DToOpenSim/sam_3d_body``.
2. Compute a reference forward pass at neutral pose: identity = 0₄₅, model_params = 0₂₀₄,
   face_expr = 0₇₂. Read the 127-joint skeleton state.
3. For each of the 204 indices, perturb that single parameter by ε (testing several ε)
   and re-run the forward pass. Measure the change in **parent-child bone lengths** (the
   126 edges of the skeleton hierarchy).

Why parent-child distances? A pure rotation parameter rotates downstream joints rigidly
around their parent — every bone length is preserved. A scaling parameter changes the
length of one or more bones. So the metric
``effect(i) = Σ_edges |‖p-c‖_perturbed - ‖p-c‖_reference|`` is, in theory, **zero for any
non-scaling parameter** and non-zero only for scalings. The boundary is what we want to
see clearly in the barplot.

Outputs
-------
- ``results/scaling_mask.json`` — empirical scaling/rotation/unused mask for all 204 indices
- ``results/classification.png`` — barplot of |Δlength|, log-scale, colored by empirical
  class, with the internal ``scaling_parameters`` mask overlaid for cross-check
- ``results/stability.png`` — log-log of effect vs ε to verify linear regime
- ``results/raw_effects.npz`` — per-index, per-ε effect magnitudes (for later re-analysis)

The model itself ships a ``parameter_transform.scaling_parameters`` boolean buffer that
labels these 204 indices — we compute the empirical mask FIRST without looking at that,
then cross-check at the end.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class Rig:
    """Wrapper around the TorchScript MHR module exposing what we need."""

    module: torch.jit.ScriptModule
    device: torch.device
    n_joints: int
    parents: np.ndarray  # (n_joints,) int — joint_parents
    internal_scaling_mask: np.ndarray  # (249,) bool
    internal_pose_mask: np.ndarray  # (249,) bool
    internal_rigid_mask: np.ndarray  # (249,) bool

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "Rig":
        module = torch.jit.load(str(path), map_location=device).eval()
        ct = module.character_torch
        parents = ct.skeleton.joint_parents.cpu().numpy().astype(np.int64)
        pt = ct.parameter_transform
        return cls(
            module=module,
            device=device,
            n_joints=int(parents.shape[0]),
            parents=parents,
            internal_scaling_mask=pt.scaling_parameters.cpu().numpy().astype(bool),
            internal_pose_mask=pt.pose_parameters.cpu().numpy().astype(bool),
            internal_rigid_mask=pt.rigid_parameters.cpu().numpy().astype(bool),
        )

    @torch.no_grad()
    def forward_joints(self, model_params: torch.Tensor) -> torch.Tensor:
        """Return joint world positions (n_joints, 3) at the requested model_params.

        Identity and face expressions stay at zero. ``apply_correctives=False`` because
        we have no pose-corrective signal to apply (and it makes each call ~10x faster).
        """
        identity = torch.zeros(1, N_IDENTITY, device=self.device)
        face_expr = torch.zeros(1, N_FACE_EXPR, device=self.device)
        _verts, skel = self.module(identity, model_params, face_expr, False)
        # skel layout per joint: (tx, ty, tz, qx, qy, qz, qw, scale)
        return skel[0, :, :3]


def parent_child_lengths(joints_xyz: torch.Tensor, parents: np.ndarray) -> torch.Tensor:
    """Return (n_edges,) tensor of ‖joint - parent_joint‖ for each non-root joint."""
    child_idx = np.where(parents >= 0)[0]
    p_idx = parents[child_idx]
    diff = joints_xyz[child_idx] - joints_xyz[p_idx]
    return diff.norm(dim=-1)


def measure_effects(
    rig: Rig, epsilons: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Run the perturbation sweep.

    Returns:
        effects: (n_params, n_eps) — Σ |Δ‖edge‖| for each (param, ε)
        joint_motion: (n_params, n_eps) — Σ ‖Δjoint_position‖ (sanity check that the
            param does *something*; helps spot truly-dead indices)
    """
    zero = torch.zeros(1, N_MODEL_PARAMS, device=rig.device)
    ref_joints = rig.forward_joints(zero)
    ref_lengths = parent_child_lengths(ref_joints, rig.parents)

    n_eps = len(epsilons)
    effects = np.zeros((N_MODEL_PARAMS, n_eps), dtype=np.float64)
    joint_motion = np.zeros((N_MODEL_PARAMS, n_eps), dtype=np.float64)

    for j, eps in enumerate(epsilons):
        for i in range(N_MODEL_PARAMS):
            mp = zero.clone()
            mp[0, i] = eps
            new_joints = rig.forward_joints(mp)
            new_lengths = parent_child_lengths(new_joints, rig.parents)
            effects[i, j] = float((new_lengths - ref_lengths).abs().sum().item())
            joint_motion[i, j] = float(
                (new_joints - ref_joints).norm(dim=-1).sum().item()
            )
    return effects, joint_motion


def classify(
    effects: np.ndarray, joint_motion: np.ndarray, epsilons: list[float]
) -> dict[str, np.ndarray]:
    """Split the 204 indices into scaling / rotation / unused via log-log slope.

    Discriminator: how does ``Σ|Δ‖edge‖|`` scale with ε?

    - True scaling parameter: ``Δlen ∝ ε`` (locally linear) → log-log slope ≈ 1
    - True rotation/rigid parameter: bone lengths are mathematically invariant → the only
      change seen is float32 round-off noise, which does NOT scale with ε → slope ≈ 0
    - Unused / dead slot: joint motion stays near zero for any ε

    This is what the user's stability concern points at directly: use the ε sweep to
    discriminate signal-scaling-with-ε from constant numerical noise.
    """
    # Log-log fit per index.
    e = np.clip(effects, 1e-20, None)
    log_e = np.log10(e)
    log_x = np.log10(np.asarray(epsilons))
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    slopes = np.empty(N_MODEL_PARAMS)
    intercepts = np.empty(N_MODEL_PARAMS)
    for i in range(N_MODEL_PARAMS):
        sol, *_ = np.linalg.lstsq(A, log_e[i], rcond=None)
        slopes[i] = sol[0]
        intercepts[i] = sol[1]

    # Truly unused: joint motion stays at noise floor even at the largest ε.
    motion_last = joint_motion[:, -1]
    active_motion = motion_last[motion_last > 1e-9]
    motion_floor = (
        float(np.median(active_motion)) * 1e-3 if active_motion.size else 1e-9
    )
    unused = motion_last < motion_floor

    # Scaling: slope ≳ 1. Rotation: slope ≲ ~0 (noise-dominated).
    # 0.5 is the natural midpoint and the histogram is strongly bimodal, so the exact
    # threshold barely matters.
    SLOPE_THRESHOLD = 0.5
    scaling = (slopes > SLOPE_THRESHOLD) & (~unused)
    rotation = (~scaling) & (~unused)

    return {
        "unused": unused,
        "scaling": scaling,
        "rotation": rotation,
        "slopes": slopes,
        "intercepts": intercepts,
        "slope_threshold": np.asarray(SLOPE_THRESHOLD),
    }


def plot_classification(
    effects: np.ndarray,
    cls: dict[str, np.ndarray],
    internal_scaling_mask_204: np.ndarray,
    out_path: Path,
) -> None:
    """Three panels:
    1. Per-index bone-length-change (largest ε) — colored by empirical class
    2. Per-index log-log slope — the discriminator (~1 for scaling, ~0 for rotation)
    3. Strip showing the model's internal ``scaling_parameters`` mask for the same indices
    """
    e_last = effects[:, -1]
    slopes = cls["slopes"]
    idx = np.arange(N_MODEL_PARAMS)

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 2, 1]}, sharex=True
    )
    ax_top, ax_mid, ax_bot = axes

    colors = np.full(N_MODEL_PARAMS, "C7", dtype=object)  # gray = unused
    colors[cls["scaling"]] = "C3"  # red = scaling
    colors[cls["rotation"]] = "C0"  # blue = rotation

    # Panel 1: |Δlength| at the largest ε
    ax_top.bar(idx, np.clip(e_last, 1e-20, None), color=list(colors), width=1.0)
    ax_top.set_yscale("log")
    ax_top.set_ylabel(f"Σ |Δ‖parent–child‖|  at ε={effects.shape[1] and 'largest'}")
    ax_top.set_title(
        f"MHR model_parameters[0..203] — empirical classification by log-log slope "
        f"(scaling={int(cls['scaling'].sum())}, "
        f"rotation={int(cls['rotation'].sum())}, "
        f"unused={int(cls['unused'].sum())})"
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="C3", label=f"empirical scaling ({int(cls['scaling'].sum())})"),
        plt.Rectangle((0, 0), 1, 1, color="C0", label=f"empirical rotation ({int(cls['rotation'].sum())})"),
        plt.Rectangle((0, 0), 1, 1, color="C7", label=f"unused ({int(cls['unused'].sum())})"),
    ]
    ax_top.legend(handles=legend_handles, loc="upper right")

    # Panel 2: slope per index — the real discriminator
    ax_mid.bar(idx, slopes, color=list(colors), width=1.0)
    ax_mid.axhline(float(cls["slope_threshold"]), color="black", linestyle="--",
                   linewidth=0.8, label=f"slope threshold = {float(cls['slope_threshold']):.2f}")
    ax_mid.axhline(1.0, color="gray", linestyle=":", linewidth=0.6,
                   label="expected slope = 1 for true scaling")
    ax_mid.axhline(0.0, color="gray", linestyle=":", linewidth=0.6,
                   label="expected slope = 0 for pure rotation (noise)")
    ax_mid.set_ylabel("log-log slope d(log Σ|Δlen|) / d(log ε)")
    ax_mid.legend(loc="center right", fontsize=8)
    ax_mid.set_ylim(-0.5, 1.5)

    # Panel 3: internal model labels for the same 204 indices
    internal_strip = internal_scaling_mask_204.astype(int)
    ax_bot.imshow(
        internal_strip.reshape(1, -1),
        aspect="auto",
        cmap="Reds",
        extent=(-0.5, N_MODEL_PARAMS - 0.5, 0, 1),
    )
    ax_bot.set_yticks([0.5])
    ax_bot.set_yticklabels(["internal\nscaling_parameters"])
    ax_bot.set_xlabel("parameter index (0..203)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_stability(effects: np.ndarray, epsilons: list[float], out_path: Path) -> None:
    """Plot effect vs ε for each parameter, log-log, to verify linear regime."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.asarray(epsilons)
    e = np.clip(effects, 1e-20, None)
    for i in range(N_MODEL_PARAMS):
        ax.plot(x, e[i], color="C7", alpha=0.15, linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε (single-param perturbation magnitude)")
    ax.set_ylabel("Σ |Δ‖parent–child‖|")
    ax.set_title(
        f"Effect vs ε for each of the {N_MODEL_PARAMS} indices (log-log).\n"
        "A clean linear bundle = stable classification."
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def cross_check(
    cls: dict[str, np.ndarray],
    internal_scaling_mask_204: np.ndarray,
    internal_pose_mask_204: np.ndarray,
    internal_rigid_mask_204: np.ndarray,
) -> dict:
    """Compare empirical scaling mask vs the model's internal ``scaling_parameters``.

    Also annotates each disagreement with the internal labels (pose/rigid/scaling). The
    Momentum framework distinguishes ``scaling_parameters`` (blend-shape-style PCA scales)
    from ``pose_parameters`` (anything affecting joint transforms). Several pose/rigid
    parameters are per-joint translations that, geometrically, change the offset to the
    parent and therefore change *bone length* even though Momentum calls them "pose".
    For our biomech use case (mapping to OpenSim segment scaling) we care about the
    operational definition: "does this index change parent-child distances?".
    """
    emp_scaling = cls["scaling"]

    def _label(i: int) -> list[str]:
        lab = []
        if internal_pose_mask_204[i]:
            lab.append("pose")
        if internal_rigid_mask_204[i]:
            lab.append("rigid")
        if internal_scaling_mask_204[i]:
            lab.append("scaling")
        return lab or ["unlabelled"]

    agree_scaling = int((emp_scaling & internal_scaling_mask_204).sum())
    only_empirical_idx = sorted(np.where(emp_scaling & ~internal_scaling_mask_204)[0].tolist())
    only_internal_idx = sorted(np.where(~emp_scaling & internal_scaling_mask_204)[0].tolist())
    return {
        "empirical_scaling_count": int(emp_scaling.sum()),
        "internal_scaling_count_204": int(internal_scaling_mask_204.sum()),
        "agree_count": agree_scaling,
        "only_empirical": [
            {"idx": int(i), "internal_labels": _label(int(i)), "slope": float(cls["slopes"][i])}
            for i in only_empirical_idx
        ],
        "only_internal": [
            {"idx": int(i), "internal_labels": _label(int(i)), "slope": float(cls["slopes"][i])}
            for i in only_internal_idx
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR_PATH)
    ap.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1],
        help="ε values to perturb each index by (multiple values check stability).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[setup] device={device} mhr={args.mhr_path}")

    rig = Rig.load(args.mhr_path, device)
    print(f"[setup] n_joints={rig.n_joints}, edges={(rig.parents >= 0).sum()}")
    print(
        f"[setup] internal masks over the full 249 vector — "
        f"scaling={int(rig.internal_scaling_mask.sum())}, "
        f"pose={int(rig.internal_pose_mask.sum())}, "
        f"rigid={int(rig.internal_rigid_mask.sum())}"
    )

    effects, joint_motion = measure_effects(rig, args.epsilons)
    cls = classify(effects, joint_motion, args.epsilons)
    print(
        f"[empirical] scaling={int(cls['scaling'].sum())}, "
        f"rotation={int(cls['rotation'].sum())}, "
        f"unused={int(cls['unused'].sum())}, "
        f"slope threshold={float(cls['slope_threshold']):.2f}"
    )

    # Stability: report slope distribution of each class to make the bimodality obvious.
    s_slopes = cls["slopes"][cls["scaling"]]
    r_slopes = cls["slopes"][cls["rotation"]]
    print(
        f"[stability] scaling slopes  median={np.median(s_slopes):.3f}, "
        f"min={s_slopes.min():.3f}, max={s_slopes.max():.3f}  (expect ≈ 1)"
    )
    if r_slopes.size:
        print(
            f"[stability] rotation slopes median={np.median(r_slopes):.3f}, "
            f"min={r_slopes.min():.3f}, max={r_slopes.max():.3f}  (expect ≈ 0)"
        )

    internal_scaling_204 = rig.internal_scaling_mask[:N_MODEL_PARAMS]
    internal_pose_204 = rig.internal_pose_mask[:N_MODEL_PARAMS]
    internal_rigid_204 = rig.internal_rigid_mask[:N_MODEL_PARAMS]
    xc = cross_check(cls, internal_scaling_204, internal_pose_204, internal_rigid_204)
    print(
        f"[cross-check] empirical_scaling={xc['empirical_scaling_count']}, "
        f"internal_scaling_204={xc['internal_scaling_count_204']}, "
        f"agreement={xc['agree_count']}, "
        f"empirical-only={len(xc['only_empirical'])}, "
        f"internal-only={len(xc['only_internal'])}"
    )
    if xc["only_empirical"]:
        labels_seen = sorted({tuple(d["internal_labels"]) for d in xc["only_empirical"]})
        print(
            f"[cross-check] empirical-only indices' internal labels: {labels_seen} — "
            f"interpretation: per-joint translations that change parent-child offsets, "
            f"i.e. operationally bone-length-changing but not labelled 'scaling' by Momentum"
        )

    # Save outputs
    np.savez_compressed(
        args.out_dir / "raw_effects.npz",
        epsilons=np.asarray(args.epsilons),
        effects=effects,
        joint_motion=joint_motion,
        loglog_slopes=cls["slopes"],
        loglog_intercepts=cls["intercepts"],
        internal_scaling_249=rig.internal_scaling_mask,
        internal_pose_249=rig.internal_pose_mask,
        internal_rigid_249=rig.internal_rigid_mask,
        parents=rig.parents,
    )
    mask_doc = {
        "schema": "mhr_model_parameters_204_classification_v0",
        "n_params": N_MODEL_PARAMS,
        "epsilons_used": args.epsilons,
        "discriminator": "log-log slope of Σ|Δ‖parent-child‖| vs ε; "
                         "true scaling has slope ≈ 1, pure rotation/rigid has slope ≈ 0 (noise)",
        "slope_threshold": float(cls["slope_threshold"]),
        "scaling_indices": sorted(int(i) for i in np.where(cls["scaling"])[0]),
        "rotation_indices": sorted(int(i) for i in np.where(cls["rotation"])[0]),
        "unused_indices": sorted(int(i) for i in np.where(cls["unused"])[0]),
        "cross_check_vs_internal": xc,
    }
    (args.out_dir / "scaling_mask.json").write_text(json.dumps(mask_doc, indent=2))
    plot_classification(
        effects, cls, internal_scaling_204, args.out_dir / "classification.png"
    )
    plot_stability(effects, args.epsilons, args.out_dir / "stability.png")
    print(f"[saved] {args.out_dir}/{{scaling_mask.json, raw_effects.npz, classification.png, stability.png}}")


if __name__ == "__main__":
    main()
