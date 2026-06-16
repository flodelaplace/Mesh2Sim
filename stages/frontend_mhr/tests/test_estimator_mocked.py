"""FastSAM3DBodyEstimator unit tests — model is mocked, no GPU.

These tests verify that the estimator-class behaviour around the adapter is
correct: empty detection, single-person mono, multi-person main-subject
selection, kwargs forwarding to the underlying ``process_one_image``.

The vendored ``SAM3DBodyEstimator`` is never imported or instantiated — only
its ``process_one_image`` is faked through a ``unittest.mock.MagicMock``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mesh2sim_frontend_mhr import (
    DEFAULT_ESTIMATOR_ID,
    FastSAM3DBodyEstimator,
    MeshEstimator,
)


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


def test_default_estimator_id_is_pinned_to_vendor_commit():
    # If the vendor SHA in DEFAULT_ESTIMATOR_ID drifts from the VENDORED.md,
    # downstream consumers can't distinguish outputs from two different
    # vendored cores. Pin tightly.
    assert DEFAULT_ESTIMATOR_ID.startswith("fast-sam-3d-body@")
    assert "936894c" in DEFAULT_ESTIMATOR_ID, (
        f"DEFAULT_ESTIMATOR_ID {DEFAULT_ESTIMATOR_ID!r} no longer references "
        "the vendored commit; sync it with third_party/VENDORED.md"
    )


def test_implements_mesh_estimator_protocol():
    est = FastSAM3DBodyEstimator(model=MagicMock())
    assert isinstance(est, MeshEstimator)


# ---------------------------------------------------------------------------
# Behaviour: empty detection
# ---------------------------------------------------------------------------


def test_no_person_detected_returns_empty_list(fake_frame_rgb):
    model = MagicMock()
    model.process_one_image.return_value = []
    est = FastSAM3DBodyEstimator(model=model)
    assert est.estimate_frame(fake_frame_rgb, frame_id=0) == []


# ---------------------------------------------------------------------------
# Behaviour: mono-subject mode (default)
# ---------------------------------------------------------------------------


def test_single_person_mono_emits_one_body_estimate(fake_frame_rgb, sam3db_one_person):
    model = MagicMock()
    model.process_one_image.return_value = [sam3db_one_person]
    est = FastSAM3DBodyEstimator(model=model)

    results = est.estimate_frame(fake_frame_rgb, frame_id=42, timestamp=1.5)
    assert len(results) == 1

    be = results[0]
    assert be.frame_id == 42
    assert be.view_id == "mono"
    assert be.timestamp == 1.5
    assert be.estimator_id == DEFAULT_ESTIMATOR_ID
    # frame_shape inferred from the input frame's (H, W).
    assert be.frame_shape == (fake_frame_rgb.shape[0], fake_frame_rgb.shape[1])


def test_multi_person_mono_subject_picks_largest_bbox(fake_frame_rgb, sam3db_two_people):
    """With main_subject_only=True (default), N detections collapse to 1."""
    model = MagicMock()
    model.process_one_image.return_value = sam3db_two_people  # [small, large]
    est = FastSAM3DBodyEstimator(model=model)

    results = est.estimate_frame(fake_frame_rgb, frame_id=0)
    assert len(results) == 1
    # The "large" person was seed=2; vertices were drawn from that RNG. The
    # cheapest robust check: pick the same person again via the helper and
    # compare a tiny invariant (the deterministic first vertex).
    from mesh2sim_frontend_mhr.estimator import _select_main_subject
    expected = _select_main_subject(sam3db_two_people)
    assert np.array_equal(results[0].mesh.vertices[0], expected["pred_vertices"][0])


# ---------------------------------------------------------------------------
# Behaviour: multi-person mode (opt-in)
# ---------------------------------------------------------------------------


def test_multi_person_mode_emits_one_per_detection(fake_frame_rgb, sam3db_two_people):
    model = MagicMock()
    model.process_one_image.return_value = sam3db_two_people
    est = FastSAM3DBodyEstimator(model=model, main_subject_only=False)

    results = est.estimate_frame(fake_frame_rgb, frame_id=0)
    assert len(results) == 2
    # Each output is contract-valid (constructor passed), and they share the
    # same frame_id (the contract has no track_id slot — see README).
    assert all(be.frame_id == 0 for be in results)
    assert all(be.view_id == "mono" for be in results)


# ---------------------------------------------------------------------------
# Behaviour: kwargs forwarding to process_one_image
# ---------------------------------------------------------------------------


def test_process_one_image_kwargs_forwarded(fake_frame_rgb, sam3db_one_person):
    model = MagicMock()
    model.process_one_image.return_value = [sam3db_one_person]
    est = FastSAM3DBodyEstimator(
        model=model,
        process_one_image_kwargs={"inference_type": "body", "hand_box_source": "yolo_pose"},
    )
    est.estimate_frame(fake_frame_rgb, frame_id=0)

    # Inspect the call: positional arg = the frame, kwargs as configured.
    args, kwargs = model.process_one_image.call_args
    assert args[0] is fake_frame_rgb
    assert kwargs == {"inference_type": "body", "hand_box_source": "yolo_pose"}


# ---------------------------------------------------------------------------
# Frame shape validation
# ---------------------------------------------------------------------------


def test_rejects_non_rgb_frame():
    est = FastSAM3DBodyEstimator(model=MagicMock())
    with pytest.raises(ValueError, match="frame_rgb must be"):
        est.estimate_frame(np.zeros((480, 640), dtype=np.uint8), frame_id=0)
    with pytest.raises(ValueError, match="frame_rgb must be"):
        est.estimate_frame(np.zeros((480, 640, 4), dtype=np.uint8), frame_id=0)


# ---------------------------------------------------------------------------
# Lazy loading is correctly stubbed
# ---------------------------------------------------------------------------


def test_from_pretrained_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="integration ticket"):
        FastSAM3DBodyEstimator.from_pretrained("/nonexistent/path")
