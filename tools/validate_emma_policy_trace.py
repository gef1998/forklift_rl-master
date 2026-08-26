#!/usr/bin/env python3
"""Compare an existing Emma ONNX policy with actions recorded by skrl play.

This validator never exports or modifies the ONNX model. The trace must contain
raw observations captured before skrl's ``RunningStandardScaler`` and the
matching deterministic Beta ``mean_actions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_TRACE_FIELDS = (
    "observations",
    "pt_actions",
    "episode_ids",
    "episode_steps",
    "checkpoint_sha256",
    "observation_stage",
)
EXPECTED_OBSERVATION_STAGE = "raw_before_running_standard_scaler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an existing Emma ONNX model against an Isaac/skrl policy trace."
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Existing ONNX policy")
    parser.add_argument("--trace", type=Path, required=True, help="Policy trace .npz")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON (default: <trace>.onnx_validation.json)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-frames", type=int, default=1000)
    parser.add_argument("--min-episodes", type=int, default=2)
    parser.add_argument("--max-abs", type=float, default=2.0e-5)
    parser.add_argument("--mean-abs", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        type=str,
        default=None,
        help="Fail if the trace was not recorded from this checkpoint hash",
    )
    parser.add_argument(
        "--expected-onnx-sha256",
        type=str,
        default=None,
        help="Fail if the ONNX file hash differs",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_string(value: np.ndarray, field: str) -> str:
    if value.shape != ():
        raise ValueError(f"Trace field {field!r} must be a scalar string, got {value.shape}")
    return str(value.item())


def load_trace(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Trace not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_TRACE_FIELDS) - set(archive.files))
        if missing:
            raise KeyError(f"Trace is missing fields: {missing}")
        trace: dict[str, Any] = {
            "observations": np.asarray(archive["observations"], dtype=np.float32),
            "pt_actions": np.asarray(archive["pt_actions"], dtype=np.float32),
            "episode_ids": np.asarray(archive["episode_ids"], dtype=np.int64),
            "episode_steps": np.asarray(archive["episode_steps"], dtype=np.int64),
            "checkpoint_sha256": _scalar_string(
                archive["checkpoint_sha256"], "checkpoint_sha256"
            ),
            "observation_stage": _scalar_string(
                archive["observation_stage"], "observation_stage"
            ),
        }
        for optional in ("checkpoint_path", "task"):
            if optional in archive.files:
                trace[optional] = _scalar_string(archive[optional], optional)
        for optional in ("terminated", "truncated"):
            if optional in archive.files:
                trace[optional] = np.asarray(archive[optional], dtype=np.bool_)
    validate_trace_shapes(trace)
    return trace


def validate_trace_shapes(trace: dict[str, Any]) -> None:
    observations = trace["observations"]
    actions = trace["pt_actions"]
    if observations.ndim != 2 or observations.shape[1] != 130:
        raise ValueError(f"observations must have shape (N, 130), got {observations.shape}")
    if observations.shape[0] == 0:
        raise ValueError("Trace contains no frames")
    if actions.ndim != 2 or actions.shape != (observations.shape[0], 2):
        raise ValueError(
            f"pt_actions must have shape ({observations.shape[0]}, 2), got {actions.shape}"
        )
    for field in ("episode_ids", "episode_steps"):
        if trace[field].shape != (observations.shape[0],):
            raise ValueError(
                f"{field} must have shape ({observations.shape[0]},), got {trace[field].shape}"
            )
    for field in ("terminated", "truncated"):
        if field in trace and trace[field].shape != (observations.shape[0],):
            raise ValueError(
                f"{field} must have shape ({observations.shape[0]},), got {trace[field].shape}"
            )
    if trace["observation_stage"] != EXPECTED_OBSERVATION_STAGE:
        raise ValueError(
            "Trace observation_stage must be "
            f"{EXPECTED_OBSERVATION_STAGE!r}, got {trace['observation_stage']!r}. "
            "A normalized trace would be scaled twice by the ONNX model."
        )
    if not np.isfinite(observations).all():
        raise ValueError("Trace observations contain NaN or inf")
    if not np.isfinite(actions).all():
        raise ValueError("Trace PT actions contain NaN or inf")


def run_onnx_batches(session: Any, observations: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise ValueError(
            f"Expected one ONNX input and at least one output, got {len(inputs)} and {len(outputs)}"
        )
    input_name = inputs[0].name
    output_name = outputs[0].name
    chunks = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        chunks.append(session.run([output_name], {input_name: batch})[0])
    actions = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if actions.shape != (observations.shape[0], 2):
        raise ValueError(f"ONNX output must have shape ({observations.shape[0]}, 2), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("ONNX actions contain NaN or inf")
    return actions


def calculate_error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_abs_limit: float,
    mean_abs_limit: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shape mismatch: {reference.shape} vs {candidate.shape}")
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError(f"Actions must have shape (N, 2), got {reference.shape}")
    if max_abs_limit <= 0 or mean_abs_limit <= 0 or atol < 0 or rtol < 0:
        raise ValueError("Error tolerances must be non-negative and aggregate limits positive")

    absolute = np.abs(reference - candidate)
    relative = absolute / np.maximum(np.abs(reference), 1.0e-12)
    tolerance = atol + rtol * np.abs(reference)
    tolerance_failures = absolute > tolerance
    hard_failures = absolute >= max_abs_limit
    failed_frames = np.flatnonzero(np.any(tolerance_failures | hard_failures, axis=1))
    worst_flat = int(np.argmax(absolute))
    worst_frame, worst_action_dim = np.unravel_index(worst_flat, absolute.shape)
    max_abs_error = float(absolute[worst_frame, worst_action_dim])
    mean_abs_error = float(absolute.mean())

    return {
        "passed": bool(
            max_abs_error < max_abs_limit
            and mean_abs_error < mean_abs_limit
            and len(failed_frames) == 0
        ),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_relative_error": float(relative.max()),
        "per_action_max_abs_error": absolute.max(axis=0).astype(float).tolist(),
        "per_action_mean_abs_error": absolute.mean(axis=0).astype(float).tolist(),
        "worst_frame": int(worst_frame),
        "worst_action_dim": int(worst_action_dim),
        "reference_action_at_worst_frame": reference[worst_frame].astype(float).tolist(),
        "onnx_action_at_worst_frame": candidate[worst_frame].astype(float).tolist(),
        "failed_frame_count": int(len(failed_frames)),
        "failed_frame_indices_first_100": failed_frames[:100].astype(int).tolist(),
        "max_abs_limit": max_abs_limit,
        "mean_abs_limit": mean_abs_limit,
        "atol": atol,
        "rtol": rtol,
    }


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    report_path = (
        args.report
        or trace_path.with_suffix(trace_path.suffix + ".onnx_validation.json")
    ).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    trace = load_trace(trace_path)
    frame_count = int(trace["observations"].shape[0])
    episode_count = int(len(np.unique(trace["episode_ids"])))
    checkpoint_hash_ok = (
        args.expected_checkpoint_sha256 is None
        or trace["checkpoint_sha256"].lower() == args.expected_checkpoint_sha256.lower()
    )
    onnx_hash = sha256(onnx_path)
    onnx_hash_ok = (
        args.expected_onnx_sha256 is None
        or onnx_hash.lower() == args.expected_onnx_sha256.lower()
    )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to validate a policy trace") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    onnx_actions = run_onnx_batches(session, trace["observations"], args.batch_size)
    metrics = calculate_error_metrics(
        trace["pt_actions"],
        onnx_actions,
        max_abs_limit=args.max_abs,
        mean_abs_limit=args.mean_abs,
        atol=args.atol,
        rtol=args.rtol,
    )
    frame_count_ok = frame_count >= args.min_frames
    episode_count_ok = episode_count >= args.min_episodes
    passed = bool(
        metrics["passed"]
        and frame_count_ok
        and episode_count_ok
        and checkpoint_hash_ok
        and onnx_hash_ok
    )

    report = {
        "passed": passed,
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_hash,
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "checkpoint_path": trace.get("checkpoint_path", ""),
        "checkpoint_sha256": trace["checkpoint_sha256"],
        "task": trace.get("task", ""),
        "observation_stage": trace["observation_stage"],
        "frames": frame_count,
        "episodes": episode_count,
        "requirements": {
            "min_frames": args.min_frames,
            "frame_count_ok": frame_count_ok,
            "min_episodes": args.min_episodes,
            "episode_count_ok": episode_count_ok,
            "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
            "checkpoint_hash_ok": checkpoint_hash_ok,
            "expected_onnx_sha256": args.expected_onnx_sha256,
            "onnx_hash_ok": onnx_hash_ok,
        },
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"onnx       : {onnx_path}")
    print(f"trace      : {trace_path}")
    print(f"frames     : {frame_count} ({'PASS' if frame_count_ok else 'FAIL'})")
    print(f"episodes   : {episode_count} ({'PASS' if episode_count_ok else 'FAIL'})")
    print(f"checkpoint : {trace['checkpoint_sha256']} ({'PASS' if checkpoint_hash_ok else 'FAIL'})")
    print(f"onnx sha256: {onnx_hash} ({'PASS' if onnx_hash_ok else 'FAIL'})")
    print("\nskrl mean_actions vs ONNX Runtime (CPU)")
    print(f"max abs    : {metrics['max_abs_error']:.8e} (limit {args.max_abs:.8e})")
    print(f"mean abs   : {metrics['mean_abs_error']:.8e} (limit {args.mean_abs:.8e})")
    print(f"max rel    : {metrics['max_relative_error']:.8e}")
    print(f"per dim max: {metrics['per_action_max_abs_error']}")
    print(f"failed rows: {metrics['failed_frame_count']}")
    print(f"worst frame: {metrics['worst_frame']}, action dim: {metrics['worst_action_dim']}")
    print(f"report     : {report_path}")
    print(f"RESULT     : {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
