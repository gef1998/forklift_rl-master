#!/usr/bin/env python3
"""Export the Emma ObsNav Beta actor to ONNX and validate it against PyTorch.

The exported graph consumes the *raw* 130-D policy observation. It embeds the
``RunningStandardScaler`` state stored in the skrl checkpoint and returns the
deterministic signed Beta mean used by ``scripts/skrl/play.py``::

    action = 2 * alpha / (alpha + beta) - 1

This script intentionally does not create an Isaac Lab environment. Run it
with the same Isaac Sim Python that was used for training so the installed
``torch`` and ``skrl`` versions match the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


POLICY_KEYS = (
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "net.4.weight",
    "net.4.bias",
    "policy_layer.weight",
    "policy_layer.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Emma ObsNav checkpoint and compare ONNX/PyTorch outputs."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/best_agent.pt"),
        help="skrl checkpoint (default: logs/best_agent.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path (default: checkpoint path with .onnx suffix)",
    )
    parser.add_argument(
        "--input-npy",
        type=Path,
        action="append",
        default=[],
        help="Optional raw observation .npy file; may be repeated",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-batch", type=int, default=256)
    parser.add_argument("--min-concentration", type=float, default=1.0)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <output>.validation.json)",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}")
    for key in ("policy", "observation_preprocessor"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint does not contain {key!r}; available keys: {list(checkpoint)}"
            )
    return checkpoint


def normalize_policy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept ordinary and DistributedDataParallel-style policy state keys."""

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key.removeprefix("module.")
        if clean_key in POLICY_KEYS:
            normalized[clean_key] = value.detach().to(device="cpu", dtype=torch.float32)
    missing = sorted(set(POLICY_KEYS) - set(normalized))
    if missing:
        raise KeyError(f"Policy checkpoint is missing actor parameters: {missing}")
    return normalized


class EmmaBetaActor(nn.Module):
    """Standalone actor matching ``EmmaNavShared`` without its value head."""

    def __init__(self, state: dict[str, torch.Tensor], min_concentration: float) -> None:
        super().__init__()
        if min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive")

        input_dim = state["net.0.weight"].shape[1]
        hidden_0 = state["net.0.weight"].shape[0]
        hidden_1 = state["net.2.weight"].shape[0]
        hidden_2 = state["net.4.weight"].shape[0]
        policy_outputs = state["policy_layer.weight"].shape[0]
        if policy_outputs % 2:
            raise ValueError(f"Policy output dimension must be even, got {policy_outputs}")

        self.observation_dim = int(input_dim)
        self.action_dim = int(policy_outputs // 2)
        self.min_concentration = float(min_concentration)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_0),
            nn.ELU(),
            nn.Linear(hidden_0, hidden_1),
            nn.ELU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(hidden_2, policy_outputs)
        self.load_state_dict(state, strict=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raw_alpha, raw_beta = self.policy_layer(self.net(observations)).chunk(2, dim=-1)
        alpha = F.softplus(raw_alpha) + self.min_concentration
        beta = F.softplus(raw_beta) + self.min_concentration
        return 2.0 * alpha / (alpha + beta) - 1.0


def make_skrl_preprocessor(
    state: dict[str, torch.Tensor], observation_dim: int
) -> nn.Module:
    try:
        from skrl.resources.preprocessors.torch import RunningStandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "skrl is required so the exporter uses the exact training-time "
            "RunningStandardScaler. Run this with Isaac Sim python.sh."
        ) from exc

    errors: list[str] = []
    scaler = None
    # skrl releases have accepted either an integer or a tuple as ``size``.
    for size in ((observation_dim,), observation_dim):
        try:
            scaler = RunningStandardScaler(size=size, device="cpu")
            try:
                scaler.load_state_dict(state, strict=True)
            except TypeError:
                # Compatibility with older skrl preprocessors whose
                # load_state_dict did not expose torch.nn.Module's ``strict``.
                scaler.load_state_dict(state)
            break
        except Exception as exc:  # retain both constructor/load diagnostics
            errors.append(f"size={size!r}: {exc}")
            scaler = None
    if scaler is None:
        raise RuntimeError(
            "Unable to restore skrl RunningStandardScaler:\n  " + "\n  ".join(errors)
        )
    scaler.eval()
    return scaler


class EmmaDeploymentPolicy(nn.Module):
    """Raw observation -> frozen skrl scaler -> deterministic Beta action."""

    def __init__(self, preprocessor: nn.Module, actor: EmmaBetaActor) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.actor = actor

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        normalized = self.preprocessor(raw_observation, train=False)
        return self.actor(normalized)


def deployment_like_observations(
    rng: np.random.Generator, batch: int, observation_dim: int
) -> np.ndarray:
    if observation_dim != 130:
        return rng.normal(size=(batch, observation_dim)).astype(np.float32)

    observation = np.zeros((batch, 130), dtype=np.float32)
    observation[:, :120] = rng.uniform(0.0, 1.0, size=(batch, 120))
    goal = rng.normal(size=(batch, 2))
    goal /= np.maximum(np.linalg.norm(goal, axis=1, keepdims=True), 1.0e-8)
    observation[:, 120:122] = goal.astype(np.float32)
    observation[:, 122] = rng.uniform(0.0, 2.0, size=batch)
    observation[:, 123:126] = rng.uniform(-1.0, 1.0, size=(batch, 3))
    observation[:, 126:130] = rng.uniform(-1.0, 1.0, size=(batch, 4))
    return observation


def make_test_cases(
    observation_dim: int,
    random_batch: int,
    seed: int,
    input_paths: list[Path],
) -> dict[str, np.ndarray]:
    if random_batch <= 0:
        raise ValueError("random_batch must be positive")
    rng = np.random.default_rng(seed)
    cases = {
        "zeros": np.zeros((1, observation_dim), dtype=np.float32),
        "linear_ramp": np.linspace(
            -1.0, 1.0, num=observation_dim, dtype=np.float32
        ).reshape(1, -1),
        "random_normal": rng.normal(size=(random_batch, observation_dim)).astype(np.float32),
        "deployment_like": deployment_like_observations(
            rng, random_batch, observation_dim
        ),
        "scaler_clip_stress": np.stack(
            [
                np.full(observation_dim, -100.0, dtype=np.float32),
                np.full(observation_dim, 100.0, dtype=np.float32),
            ]
        ),
    }
    for path in input_paths:
        array = np.asarray(np.load(path), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != observation_dim:
            raise ValueError(
                f"{path} must have shape ({observation_dim},) or (N, {observation_dim}); "
                f"got {array.shape}"
            )
        cases[f"file:{path}"] = array
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_path = (args.output or args.checkpoint.with_suffix(".onnx")).resolve()
    report_path = (
        args.report or output_path.with_suffix(output_path.suffix + ".validation.json")
    ).resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    actor_state = normalize_policy_state(checkpoint["policy"])
    actor = EmmaBetaActor(actor_state, args.min_concentration).eval()
    preprocessor = make_skrl_preprocessor(
        checkpoint["observation_preprocessor"], actor.observation_dim
    )
    policy = EmmaDeploymentPolicy(preprocessor, actor).eval()

    if actor.observation_dim != 130 or actor.action_dim != 2:
        raise ValueError(
            "This is not the expected Emma ObsNav policy: "
            f"observation_dim={actor.observation_dim}, action_dim={actor.action_dim}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(4, actor.observation_dim, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            policy,
            dummy,
            str(output_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Validation requires onnx and onnxruntime. Install them into the "
            "Isaac Sim Python environment before running this script."
        ) from exc

    onnx.checker.check_model(onnx.load(str(output_path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_inputs()[0].name != "observation":
        raise RuntimeError(f"Unexpected ONNX input: {session.get_inputs()[0].name}")

    cases = make_test_cases(
        actor.observation_dim, args.random_batch, args.seed, args.input_npy
    )
    results: dict[str, Any] = {}
    failed = False
    print(f"checkpoint : {checkpoint_path}")
    print(f"onnx       : {output_path}")
    print(
        f"policy     : obs={actor.observation_dim}, actions={actor.action_dim}, "
        f"min_concentration={args.min_concentration:g}"
    )
    print(f"checkpoint keys: {list(checkpoint)}")
    print("\nPyTorch vs ONNX Runtime (CPU)")
    print("case                         shape          max_abs       mean_abs      max_rel  status")

    for name, raw_observation in cases.items():
        with torch.inference_mode():
            torch_output = policy(torch.from_numpy(raw_observation)).cpu().numpy()
        onnx_output = session.run(["action"], {"observation": raw_observation})[0]
        difference = np.abs(torch_output - onnx_output)
        denominator = np.maximum(np.abs(torch_output), 1.0e-12)
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean())
        max_rel = float((difference / denominator).max(initial=0.0))
        passed = bool(
            np.allclose(torch_output, onnx_output, atol=args.atol, rtol=args.rtol)
        )
        failed |= not passed
        results[name] = {
            "shape": list(raw_observation.shape),
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "max_relative_error": max_rel,
            "passed": passed,
            "torch_action_min": float(torch_output.min()),
            "torch_action_max": float(torch_output.max()),
            "torch_action_first": torch_output[0].astype(float).tolist(),
            "onnx_action_first": onnx_output[0].astype(float).tolist(),
        }
        print(
            f"{name[:28]:28s} {str(tuple(raw_observation.shape)):14s} "
            f"{max_abs:12.5e} {mean_abs:12.5e} {max_rel:12.5e}  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    report = {
        "passed": not failed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx": str(output_path),
        "onnx_sha256": sha256(output_path),
        "opset": args.opset,
        "observation_dim": actor.observation_dim,
        "action_dim": actor.action_dim,
        "min_concentration": args.min_concentration,
        "atol": args.atol,
        "rtol": args.rtol,
        "seed": args.seed,
        "cases": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport     : {report_path}")
    if failed:
        print("RESULT     : FAIL - ONNX output differs from PyTorch", file=sys.stderr)
        return 1
    print("RESULT     : PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
