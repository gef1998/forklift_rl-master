#!/usr/bin/env python
"""Shared observation-free action calibration profile and report helpers.

This file is intentionally Python 2/3 compatible and is installed byte-for-byte
in the simulation and robot repositories. Adapters supply requested, commanded
and feedback streams; this module owns the profile, kinematics, lag estimates,
CSV schema and plots.
"""
from __future__ import print_function

import csv
import io
import json
import math
import os
import re


STREAM_FIELDS = [
    "run_id", "axis", "acceleration", "sequence", "segment", "time_s",
    "message_stamp_s", "receipt_time_s", "action_0", "action_1",
    "vx_mps", "wz_radps", "left_radps", "right_radps",
]


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except AttributeError:  # Python 2
        return value is not None and not math.isnan(float(value)) and not math.isinf(float(value))
    except (TypeError, ValueError):
        return False


def parse_positive_csv(value, name):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not finite(number) or number <= 0.0:
            raise ValueError("%s values must be finite and positive" % name)
        result.append(number)
    if not result:
        raise ValueError("%s must contain at least one value" % name)
    return result


def _append_hold(samples, value, duration, dt, segment):
    count = max(1, int(round(duration / dt))) if duration > 0.0 else 0
    for _ in range(count):
        samples.append((value, segment))


def _append_sine_ramp(samples, start, end, max_action_acceleration, dt, segment):
    # Shifted half-cosine velocity: zero slope at both endpoints. Its maximum
    # slope is |end-start|*pi/(2*T), so T follows directly from the requested
    # maximum acceleration.
    duration = abs(end - start) * math.pi / (2.0 * max_action_acceleration)
    count = max(1, int(math.ceil(duration / dt)))
    for index in range(1, count + 1):
        phase = float(index) / float(count)
        blend = 0.5 * (1.0 - math.cos(math.pi * phase))
        samples.append((start + (end - start) * blend, segment))


def generate_profile(max_action_acceleration, dt, hold_seconds):
    """Return 0 -> +1 -> -1 -> 0 using sinusoidal velocity ramps."""
    if not all(finite(x) and float(x) > 0.0 for x in (max_action_acceleration, dt)):
        raise ValueError("acceleration and dt must be finite and positive")
    if not finite(hold_seconds) or hold_seconds < 0.0:
        raise ValueError("hold_seconds must be finite and non-negative")
    samples = []
    _append_hold(samples, 0.0, hold_seconds, dt, "stationary_start")
    _append_sine_ramp(samples, 0.0, 1.0, max_action_acceleration, dt, "zero_to_positive")
    _append_hold(samples, 1.0, hold_seconds, dt, "positive_full")
    _append_sine_ramp(samples, 1.0, -1.0, max_action_acceleration, dt, "positive_to_negative")
    _append_hold(samples, -1.0, hold_seconds, dt, "negative_full")
    _append_sine_ramp(samples, -1.0, 0.0, max_action_acceleration, dt, "negative_to_zero")
    _append_hold(samples, 0.0, hold_seconds, dt, "stationary_end")
    return samples


def build_runs(linear_accelerations, angular_accelerations, dt, hold_seconds,
               max_linear_velocity, max_angular_velocity):
    runs = []
    for axis, accelerations, limit in (
            ("linear", linear_accelerations, max_linear_velocity),
            ("angular", angular_accelerations, max_angular_velocity)):
        if not finite(limit) or limit <= 0.0:
            raise ValueError("velocity limits must be finite and positive")
        for acceleration in accelerations:
            action_acceleration = float(acceleration) / float(limit)
            profile = generate_profile(action_acceleration, dt, hold_seconds)
            token = ("%.6f" % acceleration).rstrip("0").rstrip(".").replace(".", "p")
            runs.append({
                "run_id": "%s_a%s" % (axis, token),
                "axis": axis,
                "acceleration": float(acceleration),
                "acceleration_unit": "m/s^2" if axis == "linear" else "rad/s^2",
                "action_acceleration_per_s": action_acceleration,
                "dt_s": float(dt),
                "hold_seconds": float(hold_seconds),
                "duration_s": len(profile) * float(dt),
                "samples": profile,
            })
    return runs


def action_components(axis, value):
    if axis == "linear":
        return float(value), 0.0
    if axis == "angular":
        return 0.0, float(value)
    raise ValueError("axis must be linear or angular")


def action_to_motion(action_0, action_1, max_linear_velocity,
                     max_angular_velocity, wheel_radius, wheel_base,
                     wheel_max_velocity):
    values = (action_0, action_1, max_linear_velocity, max_angular_velocity,
              wheel_radius, wheel_base, wheel_max_velocity)
    if not all(finite(x) for x in values) or min(wheel_radius, wheel_base, wheel_max_velocity) <= 0.0:
        raise ValueError("action conversion parameters are invalid")
    action_0 = max(-1.0, min(1.0, float(action_0)))
    action_1 = max(-1.0, min(1.0, float(action_1)))
    vx = action_0 * float(max_linear_velocity)
    wz = action_1 * float(max_angular_velocity)
    left = (vx - 0.5 * float(wheel_base) * wz) / float(wheel_radius)
    right = (vx + 0.5 * float(wheel_base) * wz) / float(wheel_radius)
    scale = min(1.0, float(wheel_max_velocity) /
                max(abs(left), abs(right), 1.0e-12))
    left *= scale
    right *= scale
    vx = 0.5 * float(wheel_radius) * (left + right)
    wz = float(wheel_radius) * (right - left) / float(wheel_base)
    return {
        "vx_mps": vx, "wz_radps": wz,
        "left_radps": left, "right_radps": right,
        "projection_scale": scale,
    }


def stream_row(run, sequence, segment, time_s, motion, action=None,
               message_stamp_s=None, receipt_time_s=None):
    action = action or (None, None)
    return {
        "run_id": run["run_id"], "axis": run["axis"],
        "acceleration": run["acceleration"], "sequence": sequence,
        "segment": segment or "", "time_s": time_s,
        "message_stamp_s": message_stamp_s, "receipt_time_s": receipt_time_s,
        "action_0": action[0], "action_1": action[1],
        "vx_mps": motion.get("vx_mps"), "wz_radps": motion.get("wz_radps"),
        "left_radps": motion.get("left_radps"),
        "right_radps": motion.get("right_radps"),
    }


def _write_csv(path, rows):
    mode = "wb" if __import__("sys").version_info[0] == 2 else "w"
    with open(path, mode) as stream:
        writer = csv.DictWriter(stream, fieldnames=STREAM_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in STREAM_FIELDS})


def _write_json(path, value):
    temporary = path + ".tmp"
    with io.open(temporary, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False) + u"\n")
    getattr(os, "replace", os.rename)(temporary, path)


def _arrays(rows, field):
    import numpy as np
    pairs = [(float(row["time_s"]), float(row[field])) for row in rows
             if finite(row.get("time_s")) and finite(row.get(field))]
    if not pairs:
        return np.asarray([]), np.asarray([])
    pairs.sort(key=lambda item: item[0])
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def estimate_lag(command_rows, feedback_rows, field, max_lag_s=1.0):
    import numpy as np
    command_t, command_v = _arrays(command_rows, field)
    feedback_t, feedback_v = _arrays(feedback_rows, field)
    if command_t.size < 8 or feedback_t.size < 8:
        return None
    if float(np.max(command_v) - np.min(command_v)) < 1.0e-8:
        return None  # The inactive axis carries no response-lag information.
    unique = np.diff(np.unique(command_t))
    dt = float(np.median(unique)) if unique.size else .02
    lag_step = max(.001, min(.01, dt / 4.0))
    best = None
    for lag in np.arange(0.0, max_lag_s + .5 * lag_step, lag_step):
        target_t = command_t + lag
        mask = (target_t >= feedback_t[0]) & (target_t <= feedback_t[-1])
        if np.count_nonzero(mask) < 8:
            continue
        actual = np.interp(target_t[mask], feedback_t, feedback_v)
        error = actual - command_v[mask]
        rmse = float(np.sqrt(np.mean(error * error)))
        candidate = (rmse, float(lag), float(np.mean(np.abs(error))),
                     float(np.max(np.abs(error))), int(np.count_nonzero(mask)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    return {"lag_ms": best[1] * 1000.0, "rmse": best[0],
            "mae": best[2], "peak_abs_error": best[3], "samples": best[4]}


def _plot_run(directory, run, requested, commanded, feedback, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    groups = [(requested, "requested", "#9aa0a6", "--"),
              (commanded, "mux/sim command", "#2876b8", "-"),
              (feedback, "feedback", "#d34848", "-")]
    for ax, fields, title, unit in (
            (axes[0], ("left_radps", "right_radps"), "Left / right wheel speed", "rad/s"),
            (axes[1], ("vx_mps",), "Body linear velocity", "m/s"),
            (axes[2], ("wz_radps",), "Body angular velocity", "rad/s")):
        for rows, label, color, style in groups:
            for field in fields:
                t, values = _arrays(rows, field)
                if t.size:
                    suffix = " " + ("left" if field.startswith("left") else
                                      "right" if field.startswith("right") else field.split("_")[0])
                    ax.plot(t, values, style, color=color, alpha=.88,
                            label=label + suffix)
        if not _arrays(feedback, fields[0])[0].size:
            ax.text(.02, .08, "feedback unavailable", color="#a03030",
                    transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    req_t, req_action = _arrays(requested,
                                "action_0" if run["axis"] == "linear" else "action_1")
    if req_t.size:
        axes[3].plot(req_t, req_action, color="#7057a3", label="signed action")
    delay_t, delays = [], []
    for row in feedback:
        if finite(row.get("receipt_time_s")) and finite(row.get("message_stamp_s")):
            delay = float(row["receipt_time_s"]) - float(row["message_stamp_s"])
            if -0.05 <= delay <= 10.0:
                delay_t.append(float(row["time_s"]))
                delays.append(delay * 1000.0)
    delay_axis = axes[3].twinx()
    if delay_t:
        delay_axis.plot(delay_t, delays, color="#47886e", alpha=.7,
                        label="feedback receipt - msg stamp")
        delay_axis.set_ylabel("transport [ms]")
        delay_axis.legend(loc="lower right", fontsize=8)
    axes[3].set_ylabel("action [-1,1]")
    axes[3].set_xlabel("run time [s]")
    axes[3].grid(True, alpha=.25)
    axes[3].legend(loc="upper right", fontsize=8)

    lag_text = []
    for field in ("left_radps", "right_radps", "vx_mps", "wz_radps"):
        result = summary.get(field)
        if result:
            lag_text.append("%s=%.1fms" % (field.replace("_radps", "").replace("_mps", ""),
                                             result["lag_ms"]))
    fig.suptitle("%s | max acceleration %.4g %s | %s" %
                 (run["run_id"], run["acceleration"], run["acceleration_unit"],
                  ", ".join(lag_text) if lag_text else "feedback lag unavailable"))
    fig.tight_layout(rect=(0, 0, 1, .96))
    safe = re.sub("[^A-Za-z0-9_.-]", "_", run["run_id"])
    fig.savefig(os.path.join(directory, safe + ".png"), dpi=140)
    fig.savefig(os.path.join(directory, safe + ".svg"))
    plt.close(fig)


def write_artifacts(directory, metadata, runs, requested, commanded, feedback):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    _write_csv(os.path.join(directory, "requested.csv"), requested)
    _write_csv(os.path.join(directory, "commanded.csv"), commanded)
    _write_csv(os.path.join(directory, "feedback.csv"), feedback)
    summaries = {}
    for run in runs:
        run_id = run["run_id"]
        command_rows = [row for row in commanded if row["run_id"] == run_id]
        feedback_rows = [row for row in feedback if row["run_id"] == run_id]
        signal_summary = {}
        for field in ("left_radps", "right_radps", "vx_mps", "wz_radps"):
            result = estimate_lag(command_rows, feedback_rows, field)
            if result is not None:
                signal_summary[field] = result
        summaries[run_id] = {
            "axis": run["axis"], "acceleration": run["acceleration"],
            "acceleration_unit": run["acceleration_unit"],
            "requested_samples": sum(row["run_id"] == run_id for row in requested),
            "command_samples": len(command_rows), "feedback_samples": len(feedback_rows),
            "signals": signal_summary,
        }
        _plot_run(directory, run, [row for row in requested if row["run_id"] == run_id],
                  command_rows, feedback_rows, signal_summary)
    result = dict(metadata)
    result["profile"] = "sinusoidal half-cosine: stationary -> +full -> -full -> stationary"
    result["runs"] = summaries
    _write_json(os.path.join(directory, "summary.json"), result)
    return result
