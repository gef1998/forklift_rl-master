# Emma ObsNav task reward

The ranges-based `Emma-RL-ObsNav-Direct-v0` task uses six additive terms. Geometry,
observations, actuator constraints, optional safety projection, and asynchronous
sensor randomization are unchanged. CostmapNav is a separate task and is not
modified by this reward change.

## Objective and default units

For distance `d`, minimum current footprint clearance `c`, normalized raw policy
action `a`, and control interval `dt`:

```text
progress       = 5.0 * (d_previous - d_current)
proximity      = -0.25 * dt * clamp((0.35 - c) / (0.35 - 0.05), 0, 1)^2
action_change  = -0.002 * mean((a - a_previous)^2)
time           = -0.10 * dt
arrival        = +25 on successful termination
failure        = -30 on collision, falling, or leaving the arena
```

Progress uses metres, running costs use seconds, and action smoothness is charged
per command change. The first action after reset has no smoothness cost. Applied control noise is
not included in the action-change penalty. A
physical failure suppresses progress and arrival credit for that transition.
Timeouts receive no extra failure penalty. Goal/failure masks are shared with
termination logic so a failed robot cannot also receive a success bonus.

Clearance comes from the same current, clean, footprint-adjusted scan used for
collision detection. The policy still receives its configured noisy, delayed
scan. Clearance at or above 0.35 m has zero proximity cost; a 0.5 m side clearance
therefore remains useful corridor space. The collision threshold remains 0.05 m.

PPO `discount_factor` is 0.999 at the default 50 Hz control rate. Its approximate
effective horizon is 20 s, versus 2 s at the old 0.99. For a 10 s arrival the
terminal reward multiplier is approximately 0.606 instead of 0.00657. This is
necessary context for removing large intermediate movement bonuses; it does not
guarantee learning a maze with a feed-forward, locally observed policy.

## What was removed and why

- VFH reference selection, passage masks, opening scores, heading hysteresis,
  reference-angle rate limiting, persistent state, configuration and diagnostics.
  These were a heuristic reward teacher, not policy inputs or the safety shield.
- Positive-only progress: away-and-back transitions previously earned fresh
  approach credit. Signed distance changes telescope over an undiscounted closed
  path. Discounting still changes trajectory preferences; this is not a formal
  discount-invariant potential-shaping guarantee.
- Turn and aligned-forward bonuses: maximum forward credit alone was 0.5 per
  step (25/s). Rewarding useful activity can outweigh finishing the task.
- Activity/idle and backward penalties: time already costs something, and a
  differential-drive robot needs to reverse to escape some wall arrangements.
- The 1.2 m directional, rotational and goal-masked repulsion model. It is
  replaced by one bounded penalty near actual collision clearance.
- Separate linear/angular action-jitter weights and an unused shield-intervention
  reward setting. One weak action-change term remains.

The play action recorder now exports four panels and omits its obsolete VFH
reference columns. No zero-valued compatibility diagnostics are kept.

## Reference and adaptation

[NavRL training source, pinned revision 3725bcc](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/isaac-training/training/scripts/env.py#L547-L587)
uses velocity projected toward the goal, logarithmic obstacle distances, a
velocity-change penalty, a constant positive term, and UAV height penalties.
Emma adopts direct goal progress and a small smoothness cost. It uses local
bounded clearance costs and terminal success/failure for its episodic ground
navigation task. The UAV height term and positive survival constant are omitted.

## Validation and use

Run `python tests/test_navigation_reward_scaling.py` or the existing
`bash launch.sh test-nav-reward` entry point. Tests exercise signed progress,
closed paths, retreat, stationary rotation, control-rate units, corridor
clearance, reset smoothness, bounded costs and failure/success priority.

Old policy checkpoints retain compatible observation/action dimensions and can
still be played. Their critic and optimizer were trained on a different return;
start a fresh training run to assess the new objective. Compare held-out seeds,
success/collision/timeout rates, path length and time to goal rather than raw
returns across reward versions. Short smoke tests verify execution and numerical
behavior; they are not evidence of higher learned success rates.

Validation on 2026-09-17: all 14 behavioral tests passed. An 8-environment
Isaac Sim smoke test passed real reward/observation steps, mixed success/failure
priority, partial resets, four-panel play recording, and two PPO updates over
32 vector steps. Policy parameters changed and remained finite. This was an
execution check, not a full training comparison. Logs and pre-change backups
are in `/tmp/emma_reward_20260917/` on the development server.
