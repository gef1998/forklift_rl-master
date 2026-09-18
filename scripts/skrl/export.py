#!/usr/bin/env python3
"""Export an Emma ObsNav checkpoint to a self-contained ONNX policy.

Supported tasks
    Emma-RL-ObsNav-Direct-v0        models.name == "EmmaNavShared"  (MLP)
    Emma-RL-ObsNav-GRU-Direct-v0    models.name == "EmmaNavGRU"     (GRU)

``--task`` is required: it selects the network, and the checkpoint's own saved
``params/agent.yaml`` is then held against it.  A checkpoint trained under the
other task is reported as a mismatch instead of being exported with a network
built for the wrong task.

The exported graph maps a raw environment observation to the deterministic
signed action ``2 * mean(Beta(alpha, beta)) - 1``.  That is exactly the value
``test.py`` and ``play.py`` execute, so a simulation rollout and an ONNX
deployment run the same control law.  The agent's ``RunningStandardScaler``
observation preprocessor is baked into the graph, so the caller feeds the same
raw observation the simulator produces.

The GRU variant additionally takes and returns the recurrent hidden state, which
makes the caller responsible for resetting it at every episode start.

Only the network modules are imported, never the ``emma_rl`` package itself:
the package ``__init__`` pulls in Isaac Sim, which would force this export to
run inside a launched simulator.  Exporting therefore takes a couple of seconds
on CPU and needs neither a GPU nor an Isaac Sim installation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml


# Directory holding the Isaac-free network modules, relative to the repository.
NETWORK_PACKAGE_PATH = (
    Path("source") / "emma_rl" / "emma_rl" / "tasks" / "direct" / "emma" / "emma_obs_nav"
)
# Synthetic package name used to import those modules without their parent
# package ``__init__`` (which imports isaaclab and therefore Isaac Sim).
SHIM_PACKAGE = "_emma_obs_nav_export"
SHIM_MODULES = {
    "beta_policy": "beta_policy",
    "plain": "emma_obs_nav_network",
    "gru": "emma_obs_nav_gru_network",
}
# Task id -> the network class that task is trained with. The task is required
# on the command line rather than derived from the checkpoint: the checkpoint is
# then held against this table, so a mismatch is reported instead of silently
# exporting a network built for the wrong task.
TASK_MODELS = {
    "Emma-RL-ObsNav-Direct-v0": "EmmaNavShared",
    "Emma-RL-ObsNav-GRU-Direct-v0": "EmmaNavGRU",
}
MODEL_MODULES = {"EmmaNavShared": "plain", "EmmaNavGRU": "gru"}
DEFAULT_OPSET = 17


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export an Emma ObsNav PPO checkpoint to a self-contained ONNX policy. "
            "Supports the MLP (Emma-RL-ObsNav-Direct-v0) and GRU "
            "(Emma-RL-ObsNav-GRU-Direct-v0) baselines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to agent_<step>.pt, or to a run directory holding checkpoints/.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_MODELS),
        help=(
            "Task id the checkpoint was trained with. It selects the network and is "
            "verified against the checkpoint's saved config."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="ONNX output path (default: alongside the checkpoint, same stem).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for tracing, e.g. cpu or cuda:0.",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Export a fixed batch dimension instead of a dynamic one. "
            "Useful for runtimes that require static shapes."
        ),
    )
    parser.add_argument(
        "--no-preprocessor",
        action="store_true",
        help=(
            "Do not bake the RunningStandardScaler into the graph. The exported "
            "model then expects observations that are already normalized."
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the onnxruntime parity check against the torch policy.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-4,
        help=(
            "Maximum tolerated absolute difference during the parity check. The "
            "GRU accumulates float32 error across its hidden state, so a tight "
            "tolerance is not meaningful."
        ),
    )
    parser.add_argument(
        "--nav-dir",
        default=None,
        help="Override the directory holding the Emma ObsNav network modules.",
    )
    return parser


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_network_modules(nav_dir: Path) -> dict[str, Any]:
    """Import the network modules without executing the isaaclab-importing package init."""

    required = [nav_dir / f"{module}.py" for module in SHIM_MODULES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot find the Emma ObsNav network modules; looked for "
            f"{missing[0]}. Pass --nav-dir to point at the package directory."
        )
    if SHIM_PACKAGE not in sys.modules:
        package = types.ModuleType(SHIM_PACKAGE)
        package.__path__ = [str(nav_dir)]
        sys.modules[SHIM_PACKAGE] = package
    return {
        key: importlib.import_module(f"{SHIM_PACKAGE}.{module}")
        for key, module in SHIM_MODULES.items()
    }


def _resolve_checkpoint(path: str) -> Path:
    """Accept either a checkpoint file or a run directory containing checkpoints/."""

    candidate = Path(path).expanduser()
    if candidate.is_dir():
        checkpoints = sorted(
            (candidate / "checkpoints").glob("agent_*.pt"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"no agent_*.pt checkpoint found under {candidate}")
        return checkpoints[-1].resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    return candidate.resolve()


def _load_saved_model_config(checkpoint: Path) -> dict[str, Any]:
    """Read the models section the run was trained with."""

    config_path = checkpoint.parent.parent / "params" / "agent.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing saved training config: {config_path}. Export needs the "
            "params/ directory that sits next to the checkpoints/ directory of a run."
        )
    with config_path.open(encoding="utf-8") as stream:
        saved = yaml.safe_load(stream)
    models = saved.get("models")
    if not isinstance(models, dict) or not models.get("name"):
        raise ValueError(f"{config_path} does not declare models.name")
    return models


def _infer_dimensions(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Recover the observation and action widths from the saved policy weights."""

    try:
        observations = int(state_dict["net.0.weight"].shape[1])
        actions = int(state_dict["policy_layer.weight"].shape[0]) // 2
    except KeyError as exc:
        raise ValueError(f"checkpoint policy is not an Emma ObsNav network: {exc}") from exc
    if observations < 1 or actions < 1 or state_dict["policy_layer.weight"].shape[0] % 2:
        raise ValueError("checkpoint policy has an unexpected output shape")
    return observations, actions


def _build_parameters(models: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Collect the scalar hyper-parameters the network constructors read.

    ``generate_custom_model`` routes the same fields through skrl's generic
    config parser, but the Emma networks only consume these scalars: the layer
    sizes are hard-coded in the class definitions.
    """

    policy = models.get("policy") or {}
    value = models.get("value") or {}
    parameters: dict[str, Any] = {
        "policy": {
            "min_concentration": policy.get("min_concentration", 1.0),
            "initial_concentration": policy.get("initial_concentration", 2.0),
            "action_eps": policy.get("action_eps", 1.0e-6),
        },
        "value": {"clip_actions": value.get("clip_actions", False)},
    }
    if model_name == "EmmaNavGRU":
        recurrent = models.get("recurrent")
        if not isinstance(recurrent, dict):
            raise ValueError("EmmaNavGRU requires models.recurrent in the saved config")
        # Deployment drives a single robot, so the recurrent state is one env wide.
        parameters["recurrent"] = {**recurrent, "num_envs": 1}
    return parameters


def _load_observation_preprocessor(
    checkpoint_data: dict[str, Any], num_observations: int, device: str
) -> nn.Module:
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    state = checkpoint_data.get("observation_preprocessor")
    if not isinstance(state, dict):
        raise ValueError(
            "the checkpoint has no observation_preprocessor state; re-run with "
            "--no-preprocessor if the policy was trained without normalization"
        )
    preprocessor = RunningStandardScaler(size=num_observations, device=device)
    preprocessor.load_state_dict(state)
    return preprocessor.to(device)


class _FeedforwardPolicyExport(nn.Module):
    """Raw observation -> deterministic signed action, preprocessor baked in."""

    def __init__(self, model: nn.Module, mean_action, preprocessor: nn.Module | None) -> None:
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self._mean_action = mean_action

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            observations = self.preprocessor(observations)
        alpha, outputs = self.model.compute({"observations": observations}, role="policy")
        return self._mean_action(alpha, outputs["beta"])


class _RecurrentPolicyExport(nn.Module):
    """Raw observation + hidden state -> action + next hidden state."""

    def __init__(self, model: nn.Module, mean_action, preprocessor: nn.Module | None) -> None:
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self._mean_action = mean_action

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor):
        if self.preprocessor is not None:
            observations = self.preprocessor(observations)
        alpha, outputs = self.model.compute(
            {"observations": observations, "rnn": [hidden]}, role="policy"
        )
        return self._mean_action(alpha, outputs["beta"]), outputs["rnn"][0]


def _export_onnx(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    path: Path,
    opset: int,
) -> None:
    """Export through a temporary file so a failure never leaves a partial model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    options: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset,
        # The legacy TorchScript exporter is used because the newer dynamo path
        # requires onnxscript, which Isaac Sim's Python does not ship.
        "dynamo": False,
    }
    if dynamic_axes is not None:
        options["dynamic_axes"] = dynamic_axes
    try:
        torch.onnx.export(module, example_inputs, str(temporary), **options)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _name_action_dim(path: Path, num_actions: int) -> None:
    """Replace the exporter's auto-generated dim name on the actions output.

    The legacy exporter names any dimension it cannot prove after the producing
    node, so the action width comes out as e.g. ``Subactions_dim_1``. The width
    is known exactly from the policy layer, so declare it.
    """

    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(path))
    changed = False
    for output in model.graph.output:
        if output.name != "actions":
            continue
        dimensions = output.type.tensor_type.shape.dim
        if len(dimensions) != 2 or not dimensions[1].dim_param:
            continue
        if dimensions[1].dim_param == "batch":
            continue
        dimensions[1].ClearField("dim_param")
        dimensions[1].dim_value = num_actions
        changed = True
    if changed:
        onnx.checker.check_model(model)
        onnx.save(model, str(path))


def _sample_observations(
    checkpoint_data: dict[str, Any],
    num_observations: int,
    batch_size: int,
    preprocessor_baked: bool,
    seed: int = 1234,
) -> torch.Tensor:
    """Draw realistic observations so the parity check exercises the linear range.

    Sampling from the training distribution keeps most standardized values
    inside the preprocessor's clip threshold instead of saturating every input,
    which is a much more sensitive comparison than uniform noise.
    """

    if not preprocessor_baked:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(batch_size, num_observations, generator=generator)
    state = checkpoint_data["observation_preprocessor"]
    mean = state["running_mean"].float()
    deviation = torch.sqrt(state["running_variance"].float()).clamp_min(1.0e-6)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch_size, num_observations, generator=generator)
    return (mean + 3.0 * deviation * noise).float()


def _verify(
    path: Path,
    module: nn.Module,
    checkpoint_data: dict[str, Any],
    num_observations: int,
    recurrent: dict[str, Any] | None,
    preprocessor_baked: bool,
    atol: float,
    batch_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Compare the exported graph against the torch policy at several batch sizes.

    A statically shaped export only accepts the batch size it was traced with,
    so the caller passes exactly the sizes the artifact can serve.
    """

    try:
        import onnxruntime as ort
    except ImportError:
        return {"performed": False, "reason": "onnxruntime is not installed"}

    import onnx

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report: dict[str, Any] = {
        "performed": True,
        "onnxruntime_version": ort.__version__,
        "ir_version": int(onnx.load(str(path)).ir_version),
        "batch_sizes": [],
        "atol": atol,
    }
    worst_action = 0.0
    worst_hidden = 0.0
    for batch_size in batch_sizes:
        observations = _sample_observations(
            checkpoint_data, num_observations, batch_size, preprocessor_baked
        )
        feeds = {"observations": observations.numpy()}
        inputs: tuple[torch.Tensor, ...] = (observations,)
        if recurrent is not None:
            hidden = torch.randn(
                recurrent["num_layers"],
                batch_size,
                recurrent["hidden_size"],
                generator=torch.Generator().manual_seed(7),
            ) * 0.5
            feeds["hidden"] = hidden.numpy()
            inputs = (observations, hidden)
        with torch.no_grad():
            expected = module(*inputs)
        actual = session.run(None, feeds)
        expected = expected if isinstance(expected, tuple) else (expected,)
        action_error = float(
            np.abs(actual[0] - expected[0].numpy()).max()
        )
        worst_action = max(worst_action, action_error)
        entry = {"batch_size": batch_size, "max_abs_action_error": action_error}
        if recurrent is not None:
            hidden_error = float(np.abs(actual[1] - expected[1].numpy()).max())
            worst_hidden = max(worst_hidden, hidden_error)
            entry["max_abs_hidden_error"] = hidden_error
        report["batch_sizes"].append(entry)

    report["max_abs_action_error"] = worst_action
    if recurrent is not None:
        report["max_abs_hidden_error"] = worst_hidden
    report["passed"] = (
        bool(report["batch_sizes"]) and worst_action <= atol and worst_hidden <= atol
    )
    return report


def _describe(
    checkpoint: Path,
    output: Path,
    task: str,
    model_name: str,
    num_observations: int,
    num_actions: int,
    recurrent: dict[str, Any] | None,
    preprocessor_baked: bool,
    opset: int,
    batch_size: int | None,
    verification: dict[str, Any],
) -> dict[str, Any]:
    batch_dim: Any = batch_size if batch_size is not None else "batch"
    observations: dict[str, Any] = {
        "name": "observations",
        "shape": [batch_dim, num_observations],
        "dtype": "float32",
        "description": (
            "raw environment observation, exactly as returned by the Isaac Lab env"
            if preprocessor_baked
            else "already normalized observation; the RunningStandardScaler is NOT baked in"
        ),
    }
    actions: dict[str, Any] = {
        "name": "actions",
        "shape": [batch_dim, num_actions],
        "dtype": "float32",
        "description": (
            "deterministic signed action 2 * mean(Beta(alpha, beta)) - 1 in (-1, 1); "
            "same value as extras['mean_actions'] used by test.py and play.py"
        ),
    }
    detail: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "onnx": {
            "path": str(output),
            "sha256": _sha256(output),
            "opset": opset,
            "size_bytes": output.stat().st_size,
        },
        "model": {
            "name": model_name,
            "task": task,
            "parameter_count": None,  # filled by the caller
            "observations": observations,
            "actions": actions,
            "recurrent": None,
        },
        "preprocessor": {
            "class": "RunningStandardScaler",
            "baked_in": preprocessor_baked,
        },
        "verification": verification,
    }
    if recurrent is not None:
        detail["model"]["recurrent"] = {
            "hidden_size": recurrent["hidden_size"],
            "num_layers": recurrent["num_layers"],
            "sequence_length": recurrent["sequence_length"],
            "input": "hidden",
            "output": "next_hidden",
            "shape": ["num_layers", batch_dim, "hidden_size"],
            "description": (
                "zero the hidden state at every episode start; carry next_hidden "
                "into the following control step"
            ),
        }
    return detail


def _export(args: argparse.Namespace) -> int:
    if args.atol <= 0.0 or not np.isfinite(args.atol):
        raise ValueError("--atol must be finite and positive")
    if args.opset < 11:
        raise ValueError("--opset must be at least 11")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    checkpoint = _resolve_checkpoint(args.checkpoint)
    model_name = TASK_MODELS[args.task]
    models = _load_saved_model_config(checkpoint)
    declared_name = str(models["name"])
    if declared_name != model_name:
        raise ValueError(
            f"task/checkpoint mismatch: --task {args.task} is trained with "
            f"{model_name}, but this checkpoint records models.name={declared_name} "
            f"in {checkpoint.parent.parent / 'params' / 'agent.yaml'}. Pass the task "
            "this checkpoint was actually trained with (--task, or TASK=<task id> "
            "when going through launch.sh)."
        )

    nav_dir = (
        Path(args.nav_dir).expanduser().resolve()
        if args.nav_dir
        else (Path(__file__).resolve().parents[2] / NETWORK_PACKAGE_PATH)
    )
    modules = _load_network_modules(nav_dir)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy_state = checkpoint_data.get("policy")
    if not isinstance(policy_state, dict):
        raise ValueError(f"{checkpoint} does not contain a 'policy' state dict")

    num_observations, num_actions = _infer_dimensions(policy_state)
    parameters = _build_parameters(models, model_name)

    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32
    )
    model_class = getattr(modules[MODEL_MODULES[model_name]], model_name)
    model = model_class(observation_space, None, action_space, parameters, args.device)
    try:
        model.load_state_dict(policy_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "the checkpoint weights do not match the network built from "
            f"{checkpoint.parent.parent / 'params' / 'agent.yaml'}: {exc}"
        ) from exc
    model.eval()
    model.to(args.device)

    preprocessor = None
    if not args.no_preprocessor:
        preprocessor = _load_observation_preprocessor(
            checkpoint_data, num_observations, args.device
        )
        preprocessor.eval()

    recurrent = parameters.get("recurrent")
    mean_action = modules["beta_policy"].mapped_beta_mean
    if recurrent is None:
        module: nn.Module = _FeedforwardPolicyExport(model, mean_action, preprocessor)
        input_names = ["observations"]
        output_names = ["actions"]
    else:
        module = _RecurrentPolicyExport(model, mean_action, preprocessor)
        input_names = ["observations", "hidden"]
        output_names = ["actions", "next_hidden"]
    module.eval().to(args.device)

    batch_size = args.batch_size if args.batch_size is not None else 1
    generator = torch.Generator().manual_seed(0)
    observations = torch.randn(batch_size, num_observations, generator=generator).to(args.device)
    example_inputs: tuple[torch.Tensor, ...] = (observations,)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if recurrent is not None:
        hidden = torch.zeros(
            recurrent["num_layers"], batch_size, recurrent["hidden_size"], device=args.device
        )
        example_inputs = (observations, hidden)
    if args.batch_size is None:
        dynamic_axes = {"observations": {0: "batch"}}
        if recurrent is not None:
            dynamic_axes["hidden"] = {1: "batch"}
            dynamic_axes["next_hidden"] = {1: "batch"}
        dynamic_axes["actions"] = {0: "batch"}

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.with_suffix(".onnx")
    )

    with torch.no_grad():
        _export_onnx(
            module,
            example_inputs,
            input_names,
            output_names,
            dynamic_axes,
            output,
            args.opset,
        )
    _name_action_dim(output, num_actions)

    if args.skip_verify:
        verification: dict[str, Any] = {"performed": False, "reason": "--skip-verify"}
    else:
        verification = _verify(
            output,
            module,
            checkpoint_data,
            num_observations,
            recurrent,
            not args.no_preprocessor,
            args.atol,
            # A statically shaped graph only serves the batch size it was traced with.
            (batch_size,) if args.batch_size is not None else (1, 3),
        )

    detail = _describe(
        checkpoint=checkpoint,
        output=output,
        task=args.task,
        model_name=model_name,
        num_observations=num_observations,
        num_actions=num_actions,
        recurrent=recurrent,
        preprocessor_baked=not args.no_preprocessor,
        opset=args.opset,
        batch_size=args.batch_size,
        verification=verification,
    )
    detail["model"]["parameter_count"] = sum(p.numel() for p in model.parameters())
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[EXPORT] {model_name} -> {output}")
    print(f"[EXPORT] task            : {detail['model']['task']}")
    print(f"[EXPORT] observations    : ({num_observations},) float32"
          + ("" if recurrent is None else "  + hidden state"))
    print(f"[EXPORT] actions         : ({num_actions},) float32")
    print(f"[EXPORT] parameters      : {detail['model']['parameter_count']:,}")
    print(f"[EXPORT] preprocessor    : "
          f"{'baked in' if not args.no_preprocessor else 'NOT applied (expects normalized input)'}")
    if verification.get("performed"):
        action_error = verification["max_abs_action_error"]
        hidden_text = (
            f"  hidden {verification['max_abs_hidden_error']:.3e}"
            if recurrent is not None
            else ""
        )
        print(f"[EXPORT] parity          : action {action_error:.3e}{hidden_text}"
              f"  atol {args.atol:.1e}  "
              f"{'PASS' if verification['passed'] else 'FAIL'}")
    else:
        print(f"[EXPORT] parity          : skipped ({verification.get('reason')})")
    print(f"[EXPORT] metadata        : {metadata_path}")

    if verification.get("performed") and not verification["passed"]:
        print("[EXPORT] the ONNX graph does not reproduce the torch policy", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    # Setup problems are reported as one-line diagnostics: a missing params/
    # directory or a checkpoint that does not match its config is a user error,
    # not a bug to be debugged from a traceback.
    try:
        return _export(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[EXPORT] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
