"""Repeatable Emma drive regression; run with Isaac Sim's python.sh.

Samples each physics step and saves both velocities and poses. The default
probe bypasses navigation termination and the safety shield, settles first,
and never saves the USD. --benchmark measures ordinary env.step instead.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", type=float, default=1.0)
parser.add_argument("--turn", action="store_true")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--settle", type=float, default=0.5)
parser.add_argument("--reverse-at", type=float)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--legacy", action="store_true")
parser.add_argument("--no-acceleration-limits", action="store_true")
parser.add_argument("--linear-command-scale", type=float)
parser.add_argument("--effort-limit", type=float, default=4.15)
parser.add_argument("--pos-iters", type=int, default=8)
parser.add_argument("--vel-iters", type=int, default=4)
parser.add_argument("--contacts", action="store_true")
parser.add_argument("--smoke", action="store_true", help="check shield priority and per-env reset")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0 or args.num_envs <= 0 or args.settle < 0:
    parser.error("steps/env count must be positive and settle nonnegative")

launcher = AppLauncher(args)
app = launcher.app
import gymnasium as gym
import torch
import omni.usd
from pxr import UsdPhysics
import emma_rl.tasks
from isaaclab_tasks.utils import parse_env_cfg
from emma_rl.tasks.direct.emma.emma_obs_nav.emma_obs_nav_env import EmmaObsNavEnv

task = "Emma-RL-ObsNav-Direct-v0"
cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
cfg.seed = 0
cfg.action_control_mode = "wheel_velocity"
cfg.safety_control.enabled = False
for field in ("enable_lidar_noise", "enable_lidar_dropout", "enable_dynamics_noise", "enable_action_noise", "enable_init_noise"):
    setattr(cfg.domain_rand, field, False)
cfg.domain_rand.wheel_slip_noise_std = 0.0
cfg.domain_rand.action_noise_std = 0.0
cfg.lidar_sensor_mode = "synchronous"
cfg.collision_clearance_threshold = 0.0
cfg.goal_reach_dist = 1e-6
if args.legacy:
    cfg.drive_wheel_collision_shape = "cylinder"
    cfg.caster_collision_radius = None
    cfg.drive_enable_external_forces_every_iteration = False
    cfg.enable_motion_acceleration_limits = False
if args.no_acceleration_limits:
    cfg.enable_motion_acceleration_limits = False
if args.linear_command_scale is not None:
    cfg.linear_acceleration_command_scale = args.linear_command_scale
for name in ("wheel_left_actuator", "wheel_right_actuator"):
    cfg.robot_cfg.actuators[name].effort_limit_sim = args.effort_limit
    cfg.robot_cfg.actuators[name].effort_limit = args.effort_limit
props = cfg.robot_cfg.spawn.articulation_props
props.solver_position_iteration_count = args.pos_iters
props.solver_velocity_iteration_count = args.vel_iters
if args.contacts:
    cfg.robot_cfg.spawn.activate_contact_sensors = True
    original_setup = EmmaObsNavEnv._setup_scene
    def setup_with_contacts(self):
        original_setup(self)
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        self.scene.sensors["drive_probe"] = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/EMMA400K_body/.*",
            update_period=0.0, history_length=1, debug_vis=False,
        ))
    EmmaObsNavEnv._setup_scene = setup_with_contacts

env = gym.make(task, cfg=cfg)
env.reset()
u = env.unwrapped
r = u.robot
for obj in u.scene.rigid_objects.values():
    pose = obj.data.root_state_w[:, :7].clone()
    pose[:, 2] = -50.0
    obj.write_root_pose_to_sim(pose)
pose = r.data.default_root_state[:, :7].clone()
pose[:, :3] = u.scene.env_origins
pose[:, 2] += .045
pose[:, 3:] = pose.new_tensor([1., 0., 0., 0.])
r.write_root_pose_to_sim(pose)
r.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=u.device))

checks = {}
stage = omni.usd.get_context().get_stage()
if not args.legacy:
    for i in {0, args.num_envs - 1}:
        for name in ("dwheel_left", "dwheel_right", "caster_fl", "caster_fr", "caster_rl", "caster_rr"):
            parent = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/EMMA400K_body/{name}")
            proxy = stage.GetPrimAtPath(str(parent.GetPath()) + "/navigation_contact")
            assert parent.GetTypeName() == "Cylinder", "visible tire was changed"
            assert proxy.IsValid(), f"missing cloned proxy {i}/{name}"
            assert not UsdPhysics.CollisionAPI(parent).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Get()
    checks["proxies_cloned_visuals_preserved"] = True

zero = torch.zeros(args.num_envs, 2, device=u.device)
def advance(action, record=None, clock=0.0):
    u._pre_physics_step(action)
    for _ in range(cfg.decimation):
        u._sim_step_counter += 1
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        clock += u.physics_dt
        if record is not None:
            record.append(sample(clock))
    u.episode_length_buf += 1
    u.common_step_counter += 1
    return clock

def sample(t):
    row = dict(t=t, joint_pos=r.data.joint_pos[0].tolist(),
               joint_vel=r.data.joint_vel[0].tolist(),
               target=r.data.joint_vel_target[0, :2].tolist(),
               root_pos=r.data.root_pos_w[0].tolist(),
               root_quat=r.data.root_quat_w[0].tolist(),
               velocity_b=r.data.root_lin_vel_b[0].tolist(),
               angular_b=r.data.root_ang_vel_b[0].tolist())
    if args.contacts:
        row["contact_net_w"] = u.scene.sensors["drive_probe"].data.net_forces_w[0].tolist()
    return row

for _ in range(math.ceil(args.settle / u.step_dt)):
    advance(zero)
action = zero.clone()
action[:, 0] = args.target / cfg.wheel_max_velocity
action[:, 1] = (-args.target if args.turn else args.target) / cfg.wheel_max_velocity
trace = []
summary = {}
if args.benchmark:
    for _ in range(40):
        env.step(action)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    summary = dict(seconds=elapsed, env_steps_per_second=args.num_envs * args.steps / elapsed)
else:
    trace.append(sample(0.0))
    t = 0.0
    for _ in range(args.steps):
        sign = -1 if args.reverse_at is not None and t >= args.reverse_at - 1e-8 else 1
        t = advance(action * sign, trace, t)
    tail = [row for row in trace if row["t"] >= max(0, t - 2.0) - 1e-8]
    expected_yaw = -2 * args.target * cfg.wheel_radius / cfg.wheel_base
    final_sign = -1 if args.reverse_at is not None and t > args.reverse_at else 1
    mean_yaw = statistics.fmean(x["angular_b"][2] for x in tail)
    changes = []
    command_changes = []
    for a,b in zip(trace, trace[1:]):
        dt = b["t"] - a["t"]
        changes.append(((b["velocity_b"][0]-a["velocity_b"][0])/dt,
                        (b["angular_b"][2]-a["angular_b"][2])/dt))
        dl,dr = (b["target"][k]-a["target"][k] for k in (0,1))
        command_changes.append(((dl+dr)*cfg.wheel_radius/(2*dt),
                                (dr-dl)*cfg.wheel_radius/(cfg.wheel_base*dt)))
    summary = dict(yaw_mean=mean_yaw,
        tracking_pct=100*mean_yaw/(expected_yaw*final_sign) if args.turn else None,
        forward_mean=statistics.fmean(x["velocity_b"][0] for x in tail),
        wheel_mean=[statistics.fmean(x["joint_vel"][k] for x in tail) for k in (0,1)],
        wheel_sd=[statistics.pstdev(x["joint_vel"][k] for x in tail) for k in (0,1)],
        measured_peak_acceleration=[max(abs(x[k]) for x in changes) for k in (0,1)],
        command_peak_acceleration=[max(abs(x[k]) for x in command_changes) for k in (0,1)])
    if cfg.enable_motion_acceleration_limits:
        assert summary["command_peak_acceleration"][0] <= cfg.max_linear_acceleration + 1e-4
        assert summary["command_peak_acceleration"][1] <= cfg.max_angular_acceleration + 1e-4
        checks["nominal_command_slew_bounds"] = True

if args.smoke:
    cfg.safety_control.enabled = True
    u._has_cached_scan[:] = False
    u._pre_physics_step(torch.ones_like(action))
    u._apply_action()
    assert torch.count_nonzero(u._wheel_vel_cmd).item() == 0
    checks["shield_stops_without_interpolation_delay"] = True
    cfg.safety_control.enabled = False
    u._pre_physics_step(action)
    u._apply_action()
    other = u._wheel_vel_cmd[1:].clone()
    u._reset_idx(torch.tensor([0], device=u.device))
    for buffer in (u._wheel_vel_cmd, u._joint_vel_cmd, u._wheel_target_start):
        assert torch.count_nonzero(buffer[0]).item() == 0
    torch.testing.assert_close(u._wheel_vel_cmd[1:], other)
    checks["reset_clears_only_selected_env_commands"] = True

result = dict(args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
    drive_config={name: getattr(cfg, name) for name in (
        "drive_wheel_collision_shape", "caster_collision_radius",
        "drive_enable_external_forces_every_iteration", "enable_motion_acceleration_limits",
        "max_linear_acceleration", "max_angular_acceleration",
        "linear_acceleration_command_scale", "angular_acceleration_command_scale",
    )},
    dt=cfg.sim.dt, decimation=cfg.decimation, joint_names=r.data.joint_names,
    body_names=r.data.body_names,
    contact_names=u.scene.sensors["drive_probe"].body_names if args.contacts else [],
    checks=checks, summary=summary, trace=trace)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("[drive_probe]", json.dumps(dict(checks=checks, summary=summary)), flush=True)
print("[drive_probe] saved", args.out, flush=True)
os._exit(0)
