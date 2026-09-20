#!/usr/bin/env python3
"""Run resumable one-factor Emma drive-response sweeps."""

import argparse
import csv
import json
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--parameter", choices=("armature", "damping", "effort"), required=True)
parser.add_argument("--values", required=True, help="comma-separated floats")
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--position-iterations", type=int, default=4)
parser.add_argument("--velocity-iterations", type=int, default=2)
args = parser.parse_args()

repo = Path(__file__).resolve().parents[2]
runner = Path("/home/jz/isaacsim/python.sh")
flag = {
    "armature": "--wheel-armature",
    "damping": "--wheel-damping",
    "effort": "--wheel-effort-limit",
}[args.parameter]
values = [float(item) for item in args.values.split(",")]
args.output_root.mkdir(parents=True, exist_ok=True)
records = []

for value in values:
    label = ("%g" % value).replace("-", "m").replace(".", "p")
    output = (args.output_root / (args.parameter + "_" + label)).resolve()
    summary_path = output / "summary.json"
    log_path = args.output_root / (args.parameter + "_" + label + ".log")
    if not summary_path.exists():
        command = [
            str(runner), "scripts/wheel_response_test.py",
            "--task", "Emma-RL-ObsNav-Direct-v0", "--headless",
            "--position-iterations", str(args.position_iterations),
            "--velocity-iterations", str(args.velocity_iterations),
            flag, str(value),
            "--linear_accelerations", "0.6",
            "--angular_accelerations", "1.2",
            "--output_dir", str(output),
        ]
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=str(repo), stdout=stream,
                                       stderr=subprocess.STDOUT)
        if completed.returncode or not summary_path.exists():
            print("[SWEEP] %s=%g FAILED (exit=%d, log=%s)" %
                  (args.parameter, value, completed.returncode, log_path), flush=True)
            continue
    summary = json.loads(summary_path.read_text())
    linear = summary["runs"]["linear_a0p6"]["signals"]
    angular = summary["runs"]["angular_a1p2"]["signals"]
    row = {
        "parameter": args.parameter,
        "value": value,
        "position_iterations": summary["solver_position_iteration_count"],
        "velocity_iterations": summary["solver_velocity_iteration_count"],
        "vx_lag_ms": linear["vx_mps"]["lag_ms"],
        "vx_rmse": linear["vx_mps"]["rmse"],
        "wz_lag_ms": angular["wz_radps"]["lag_ms"],
        "wz_rmse": angular["wz_radps"]["rmse"],
        "left_linear_lag_ms": linear["left_radps"]["lag_ms"],
        "right_linear_lag_ms": linear["right_radps"]["lag_ms"],
        "left_angular_lag_ms": angular["left_radps"]["lag_ms"],
        "right_angular_lag_ms": angular["right_radps"]["lag_ms"],
        "wheel_armature_kgm2": summary["wheel_armature_kgm2"],
        "wheel_damping": summary["wheel_velocity_damping"],
        "wheel_effort_limit_nm": summary["wheel_effort_limit_nm"],
        "output": str(output),
    }
    records.append(row)
    print("[SWEEP] %s=%g vx=%.1fms wz=%.1fms vx_rmse=%.5f wz_rmse=%.5f" %
          (args.parameter, value, row["vx_lag_ms"], row["wz_lag_ms"],
           row["vx_rmse"], row["wz_rmse"]), flush=True)

(args.output_root / (args.parameter + "_summary.json")).write_text(
    json.dumps(records, indent=2) + "\n")
if records:
    with (args.output_root / (args.parameter + "_summary.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
