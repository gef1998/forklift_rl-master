#!/usr/bin/env python3
"""Pure NumPy tests for Emma policy trace validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "validate_emma_policy_trace.py"
)
SPEC = importlib.util.spec_from_file_location("validate_emma_policy_trace", MODULE_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


def _write_trace(path: Path, *, stage: str = validator.EXPECTED_OBSERVATION_STAGE) -> None:
    frame_count = 1001
    np.savez_compressed(
        path,
        observations=np.zeros((frame_count, 130), dtype=np.float32),
        pt_actions=np.zeros((frame_count, 2), dtype=np.float32),
        episode_ids=np.concatenate(
            [np.zeros(500, dtype=np.int64), np.ones(501, dtype=np.int64)]
        ),
        episode_steps=np.concatenate(
            [np.arange(500, dtype=np.int64), np.arange(501, dtype=np.int64)]
        ),
        checkpoint_sha256=np.asarray("abc123"),
        observation_stage=np.asarray(stage),
        checkpoint_path=np.asarray("/tmp/best_agent.pt"),
        task=np.asarray("Emma-RL-ObsNav-Direct-v0"),
    )


def test_load_trace_accepts_raw_observation_schema(tmp_path: Path):
    path = tmp_path / "trace.npz"
    _write_trace(path)
    trace = validator.load_trace(path)

    assert trace["observations"].shape == (1001, 130)
    assert trace["pt_actions"].shape == (1001, 2)
    assert len(np.unique(trace["episode_ids"])) == 2
    assert trace["checkpoint_sha256"] == "abc123"


def test_load_trace_rejects_preprocessed_observations(tmp_path: Path):
    path = tmp_path / "normalized_trace.npz"
    _write_trace(path, stage="after_running_standard_scaler")

    with pytest.raises(ValueError, match="scaled twice"):
        validator.load_trace(path)


def test_error_metrics_pass_for_small_export_noise():
    reference = np.array([[0.5, -0.25], [0.0, 0.75]], dtype=np.float32)
    candidate = reference + np.array(
        [[1.0e-6, -1.0e-6], [5.0e-7, 0.0]], dtype=np.float32
    )

    metrics = validator.calculate_error_metrics(
        reference,
        candidate,
        max_abs_limit=2.0e-5,
        mean_abs_limit=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    assert metrics["passed"]
    assert metrics["failed_frame_count"] == 0
    assert metrics["max_abs_error"] < 2.0e-5


def test_error_metrics_report_worst_frame_and_dimension():
    reference = np.zeros((4, 2), dtype=np.float32)
    candidate = reference.copy()
    candidate[2, 1] = 3.0e-4

    metrics = validator.calculate_error_metrics(
        reference,
        candidate,
        max_abs_limit=2.0e-5,
        mean_abs_limit=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    assert not metrics["passed"]
    assert metrics["worst_frame"] == 2
    assert metrics["worst_action_dim"] == 1
    assert metrics["failed_frame_indices_first_100"] == [2]

