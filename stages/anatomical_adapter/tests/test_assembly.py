"""T5 mono assembly tests — pure, no GPU.

Coverage matrix:
- Nominal assembly: positions stacked at the right temporal index, dimensions correct.
- Temporal reordering: out-of-order observations get sorted by timestamp.
- Frame_id fallback: works when timestamps are absent.
- Missing landmark: visibility=0 or pos_3d=None → NaN in positions, 0 in confidence.
- Verrou ensemble de landmarks: heterogeneous landmark sets → LandmarkSetMismatchError.
- Verrou temporel: duplicate or non-monotonic timestamps/frame_ids → TemporalOrderError.
- Verrou view: multiple view_ids in mono → ViewMismatchError.
- ShapeDescriptor attached and preserved.
- Round-trip via contracts IO.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mesh2sim.contracts import (
    AnatomicalObservation,
    AnatomicalTrajectory,
    Capabilities,
    Landmark,
    Mode,
    Pos3DFrame,
    Provenance,
    ShapeDescriptor,
    ShapeRepresentation,
    Source,
    load,
    save,
)
from mesh2sim_anatomical_adapter import (
    ASSEMBLY_ID,
    LandmarkSetMismatchError,
    TemporalOrderError,
    ViewMismatchError,
    assemble_trajectory_mono,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _lm(
    pos, *, confidence: float = 0.9, visibility: float = 1.0, source: Source = Source.bony
) -> Landmark:
    """Build a Landmark, default visibility=1.0 (the T3 adapter convention)."""
    return Landmark(
        pos_3d=np.asarray(pos, dtype=np.float64),
        confidence=confidence,
        visibility=visibility,
        source=source,
    )


def _make_obs(
    frame_id: int,
    *,
    landmarks: dict,
    timestamp: float | None = None,
    view_id: str = "mono",
    pos3d_frame: Pos3DFrame = Pos3DFrame.camera,
) -> AnatomicalObservation:
    return AnatomicalObservation(
        frame_id=frame_id,
        view_id=view_id,
        timestamp=timestamp,
        landmarks=landmarks,
        pos3d_frame=pos3d_frame,
        capabilities=Capabilities(),
        provenance=Provenance(
            estimator_id="test-frontend",
            adapter_id="test-adapter",
            correspondence_map_id="test-map",
        ),
    )


def _shape_desc() -> ShapeDescriptor:
    return ShapeDescriptor(
        representation=ShapeRepresentation.opaque,
        data={
            "shape_params": np.zeros(45, dtype=np.float32),
            "scale_params": np.zeros(28, dtype=np.float32),
            "expr_params": np.zeros(72, dtype=np.float32),
            "body_pose_shape_modes": np.zeros(6, dtype=np.float32),
        },
        source_model="mhr_v1",
    )


# ---------------------------------------------------------------------------
# Nominal assembly
# ---------------------------------------------------------------------------


def test_nominal_assembly_T_L_3_dimensions_and_byte_exact():
    """3 frames × 2 landmarks → trajectory (3, 2, 3), positions byte-exact."""
    obs = [
        _make_obs(
            0,
            landmarks={"RASI": _lm([1.0, 2.0, 3.0]), "LASI": _lm([4.0, 5.0, 6.0])},
            timestamp=0.000,
        ),
        _make_obs(
            1,
            landmarks={"RASI": _lm([1.1, 2.1, 3.1]), "LASI": _lm([4.1, 5.1, 6.1])},
            timestamp=0.033,
        ),
        _make_obs(
            2,
            landmarks={"RASI": _lm([1.2, 2.2, 3.2]), "LASI": _lm([4.2, 5.2, 6.2])},
            timestamp=0.066,
        ),
    ]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S001", trial_id="t1", task="gait", fps=30.0
    )
    assert traj.positions.shape == (3, 2, 3)
    assert traj.confidence.shape == (3, 2)
    assert traj.landmark_names == ["RASI", "LASI"]
    assert traj.mode == Mode.mono
    assert traj.views_used == ["mono"]
    assert traj.fps == 30.0
    np.testing.assert_array_equal(traj.positions[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(traj.positions[0, 1], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(traj.positions[2, 1], [4.2, 5.2, 6.2])
    assert (traj.confidence == 0.9).all()


def test_assembly_records_assembly_id_in_provenance():
    obs = [_make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=0)]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    assert traj.provenance.extra.get("assembly_id") == ASSEMBLY_ID


def test_assembly_records_pos3d_frame_in_extra():
    """The source pos3d_frame is stashed in provenance.extra (the contract has no
    explicit field). T6b downstream reads it to verify what it's transforming."""
    obs = [_make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=0)]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    assert traj.provenance.extra.get("assembled_from_pos3d_frame") == "camera"


# ---------------------------------------------------------------------------
# Temporal ordering
# ---------------------------------------------------------------------------


def test_reorders_observations_by_timestamp():
    """Observations passed out of order → reordered by timestamp."""
    a = _make_obs(0, landmarks={"RASI": _lm([1.0, 2.0, 3.0])}, timestamp=0.066)
    b = _make_obs(1, landmarks={"RASI": _lm([2.0, 3.0, 4.0])}, timestamp=0.000)
    c = _make_obs(2, landmarks={"RASI": _lm([3.0, 4.0, 5.0])}, timestamp=0.033)
    traj = assemble_trajectory_mono(
        [a, b, c], _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    # Expected order by timestamp: b (0.000) → c (0.033) → a (0.066)
    np.testing.assert_array_equal(traj.positions[0, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(traj.positions[1, 0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(traj.positions[2, 0], [1.0, 2.0, 3.0])


def test_falls_back_to_frame_id_when_no_timestamps():
    """When all timestamps are None, sort by frame_id."""
    a = _make_obs(2, landmarks={"RASI": _lm([3.0, 4.0, 5.0])}, timestamp=None)
    b = _make_obs(0, landmarks={"RASI": _lm([1.0, 2.0, 3.0])}, timestamp=None)
    c = _make_obs(1, landmarks={"RASI": _lm([2.0, 3.0, 4.0])}, timestamp=None)
    traj = assemble_trajectory_mono(
        [a, b, c], _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    np.testing.assert_array_equal(traj.positions[0, 0], [1.0, 2.0, 3.0])  # frame 0
    np.testing.assert_array_equal(traj.positions[1, 0], [2.0, 3.0, 4.0])  # frame 1
    np.testing.assert_array_equal(traj.positions[2, 0], [3.0, 4.0, 5.0])  # frame 2


# ---------------------------------------------------------------------------
# Missing marker preservation
# ---------------------------------------------------------------------------


def test_missing_marker_pos_3d_none_becomes_nan():
    """pos_3d=None on a frame → NaN in positions, 0 in confidence, no interpolation."""
    obs = [
        _make_obs(
            0,
            landmarks={"RASI": _lm([1, 2, 3]), "LASI": _lm([4, 5, 6])},
            timestamp=0.000,
        ),
        _make_obs(
            1,
            landmarks={
                "RASI": _lm([1.1, 2.1, 3.1]),
                "LASI": Landmark(
                    pos_3d=None, confidence=0.0, visibility=0.0, source=Source.unknown
                ),
            },
            timestamp=0.033,
        ),
        _make_obs(
            2,
            landmarks={"RASI": _lm([1.2, 2.2, 3.2]), "LASI": _lm([4.2, 5.2, 6.2])},
            timestamp=0.066,
        ),
    ]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    # LASI at frame 1 is missing → NaN, conf=0; surrounding frames untouched.
    assert np.isnan(traj.positions[1, 1]).all()
    assert traj.confidence[1, 1] == 0.0
    np.testing.assert_array_equal(traj.positions[0, 1], [4, 5, 6])
    np.testing.assert_array_equal(traj.positions[2, 1], [4.2, 5.2, 6.2])
    # No interpolation: the value at t=1 must NOT be the mean of t=0 and t=2.


def test_missing_marker_visibility_zero_becomes_nan():
    """visibility=0 marks the landmark missing even if pos_3d is non-None."""
    obs = [
        _make_obs(
            0,
            landmarks={"RASI": _lm([1, 2, 3]), "LASI": _lm([99, 99, 99], visibility=0.0)},
            timestamp=0.0,
        ),
    ]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    assert np.isnan(traj.positions[0, 1]).all()
    assert traj.confidence[0, 1] == 0.0


# ---------------------------------------------------------------------------
# Verrous (loud rejections)
# ---------------------------------------------------------------------------


def test_rejects_landmark_set_mismatch():
    obs = [
        _make_obs(
            0,
            landmarks={"RASI": _lm([1, 2, 3]), "LASI": _lm([4, 5, 6])},
            timestamp=0.0,
        ),
        _make_obs(
            1,
            landmarks={"RASI": _lm([1.1, 2.1, 3.1]), "C7": _lm([7, 8, 9])},
            timestamp=0.033,
        ),
    ]
    with pytest.raises(LandmarkSetMismatchError, match="different landmark set"):
        assemble_trajectory_mono(
            obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
        )


def test_rejects_duplicate_frame_id_without_timestamps():
    obs = [
        _make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=None),
        _make_obs(0, landmarks={"RASI": _lm([2, 3, 4])}, timestamp=None),
    ]
    with pytest.raises(TemporalOrderError, match="non-monotonic frame_id"):
        assemble_trajectory_mono(
            obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
        )


def test_rejects_duplicate_timestamp():
    obs = [
        _make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=0.5),
        _make_obs(1, landmarks={"RASI": _lm([2, 3, 4])}, timestamp=0.5),
    ]
    with pytest.raises(TemporalOrderError, match="non-monotonic timestamp"):
        assemble_trajectory_mono(
            obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
        )


def test_rejects_view_id_mismatch():
    obs = [
        _make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=0, view_id="mono"),
        _make_obs(1, landmarks={"RASI": _lm([2, 3, 4])}, timestamp=0.033, view_id="cam0"),
    ]
    with pytest.raises(ViewMismatchError, match="multiple view_ids"):
        assemble_trajectory_mono(
            obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
        )


def test_rejects_empty_observations():
    with pytest.raises(TemporalOrderError, match="empty"):
        assemble_trajectory_mono(
            [], _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
        )


# ---------------------------------------------------------------------------
# ShapeDescriptor attachment
# ---------------------------------------------------------------------------


def test_shape_descriptor_attached_and_unique():
    """The trajectory must carry the ShapeDescriptor passed in, unchanged."""
    obs = [_make_obs(0, landmarks={"RASI": _lm([1, 2, 3])}, timestamp=0)]
    sd = _shape_desc()
    traj = assemble_trajectory_mono(obs, sd, subject_id="S", trial_id="t", task="gait", fps=30)
    assert traj.shape_descriptor is not None
    assert traj.shape_descriptor.source_model == "mhr_v1"
    np.testing.assert_array_equal(
        traj.shape_descriptor.data["shape_params"], sd.data["shape_params"]
    )
    np.testing.assert_array_equal(
        traj.shape_descriptor.data["body_pose_shape_modes"], sd.data["body_pose_shape_modes"]
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_through_contracts_io(tmp_path: Path):
    obs = [
        _make_obs(
            i,
            landmarks={"RASI": _lm([i, 2 * i, 3 * i]), "LASI": _lm([-i, -2 * i, -3 * i])},
            timestamp=i / 30.0,
        )
        for i in range(5)
    ]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S001", trial_id="t1", task="gait", fps=30.0
    )
    out = save(traj, tmp_path / "traj")
    back = load(AnatomicalTrajectory, out)

    np.testing.assert_array_equal(back.positions, traj.positions)
    np.testing.assert_array_equal(back.confidence, traj.confidence)
    assert back.landmark_names == traj.landmark_names
    assert back.mode == traj.mode
    assert back.fps == traj.fps
    assert back.subject_id == "S001"
    assert back.trial_id == "t1"
    assert back.views_used == ["mono"]
    assert back.uncertainty is None
    # Shape descriptor round-trips too
    np.testing.assert_array_equal(
        back.shape_descriptor.data["shape_params"], traj.shape_descriptor.data["shape_params"]
    )
    # Provenance tags survive
    assert back.provenance.extra.get("assembly_id") == ASSEMBLY_ID
    assert back.provenance.extra.get("assembled_from_pos3d_frame") == "camera"


def test_roundtrip_preserves_nan_at_missing_markers(tmp_path: Path):
    """NaN positions and 0 confidence must survive save/load byte-identical."""
    obs = [
        _make_obs(
            0,
            landmarks={"RASI": _lm([1, 2, 3]), "LASI": _lm([4, 5, 6])},
            timestamp=0.0,
        ),
        _make_obs(
            1,
            landmarks={
                "RASI": Landmark(pos_3d=None, confidence=0, visibility=0, source=Source.unknown),
                "LASI": _lm([4.1, 5.1, 6.1]),
            },
            timestamp=0.033,
        ),
    ]
    traj = assemble_trajectory_mono(
        obs, _shape_desc(), subject_id="S", trial_id="t", task="gait", fps=30
    )
    out = save(traj, tmp_path / "traj")
    back = load(AnatomicalTrajectory, out)
    assert np.isnan(back.positions[1, 0]).all()
    assert back.confidence[1, 0] == 0.0
    np.testing.assert_array_equal(back.positions[0, 0], [1, 2, 3])
