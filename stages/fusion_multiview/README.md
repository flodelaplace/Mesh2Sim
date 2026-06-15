# stages/fusion_multiview

Multi-view fusion: combine per-view `AnatomicalObservation`s into a per-trial
`AnatomicalTrajectory` in the world frame.

## Scope

- MAMMA-style fusion (no re-trained dense detector; MHR topology gives correspondence for
  free).
- Asymmetric scaling: free, weakly regularized in multi-view (vs. frozen in mono).
- Mono = multi with N=1 (single code path).

## Environment

Moderate: PyTorch (CPU or GPU). Shares a venv with `frontend_mhr` is *possible* but not
required — keep separate if dependency pins drift.
