"""Task-based navigation rewards, independent of Isaac Sim and local planners.

Signed distance change integrates NavRL's goal-directed velocity idea over the
actual transition. Running costs use seconds; smoothness is charged per action
change. No heading, forward-motion, survival or reverse-motion bonus is needed.
"""

from __future__ import annotations

import torch


def navigation_reward_terms(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    minimum_clearance: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    has_previous_action: torch.Tensor,
    goal_reached: torch.Tensor,
    failure: torch.Tensor,
    *,
    step_dt: float,
    progress_per_meter: float,
    clearance_threshold: float,
    collision_clearance: float,
    proximity_cost_per_second: float,
    time_cost_per_second: float,
    action_change_scale: float,
    goal_bonus: float,
    failure_penalty: float,
) -> dict[str, torch.Tensor]:
    """Return six additive terms with shape ``(num_envs,)``.

    Clearance is current, noise-free distance outside the robot footprint.
    Collision/fall/out-of-bounds failure takes precedence over goal success and
    suppresses progress on that transition. A timeout is not a physical failure.
    The first action after reset has no preceding policy action to regularize.
    """
    progress = progress_per_meter * (previous_distance - current_distance)
    progress = torch.where(failure | ~torch.isfinite(progress), 0.0, progress)
    clearance = torch.nan_to_num(
        minimum_clearance, nan=collision_clearance,
        posinf=clearance_threshold, neginf=collision_clearance,
    )
    proximity = ((clearance_threshold - clearance)
                 / (clearance_threshold - collision_clearance)).clamp(0.0, 1.0).square()
    action_change = (action - previous_action).square().mean(dim=-1)
    action_change = torch.where(has_previous_action, action_change, 0.0)
    return {
        "progress": progress,
        "proximity": -proximity_cost_per_second * step_dt * proximity,
        "action_change": -action_change_scale * action_change,
        "time": torch.full_like(current_distance, -time_cost_per_second * step_dt),
        "goal": (goal_reached & ~failure).to(current_distance.dtype) * goal_bonus,
        "failure": failure.to(current_distance.dtype) * failure_penalty,
    }
