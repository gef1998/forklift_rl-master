"""Tensor-only collision tolerance state for the navigation curriculum."""

from __future__ import annotations

import torch


def collision_curriculum_step(
    minimum_clearance: torch.Tensor,
    contact_active: torch.Tensor,
    contact_duration_s: torch.Tensor,
    episode_event_count: torch.Tensor,
    terminate_immediately: torch.Tensor,
    max_nonterminal_events: torch.Tensor,
    *,
    step_dt: float,
    collision_clearance: float,
    release_clearance: float,
    severe_clearance: float,
    max_continuous_collision_time_s: float,
) -> dict[str, torch.Tensor]:
    """Advance debounced collision state and decide which contacts are fatal.

    A collision event is counted only when contact starts. The active latch is
    released above ``release_clearance`` so LiDAR noise around the collision
    threshold cannot generate many events from one wall contact.
    """

    reference_shape = minimum_clearance.shape
    for name, value in (
        ("contact_active", contact_active),
        ("contact_duration_s", contact_duration_s),
        ("episode_event_count", episode_event_count),
        ("terminate_immediately", terminate_immediately),
        ("max_nonterminal_events", max_nonterminal_events),
    ):
        if value.shape != reference_shape:
            raise ValueError(f"{name} must match minimum_clearance shape")

    collision = minimum_clearance <= collision_clearance
    released = minimum_clearance > release_clearance
    collision_event = collision & ~contact_active
    next_event_count = episode_event_count + collision_event.to(
        episode_event_count.dtype
    )
    next_contact_duration = torch.where(
        collision,
        contact_duration_s + step_dt,
        torch.where(released, torch.zeros_like(contact_duration_s), contact_duration_s),
    )
    next_contact_active = (contact_active | collision) & ~released

    severe_collision = collision & (minimum_clearance <= severe_clearance)
    sustained_collision = collision & (
        next_contact_duration >= max_continuous_collision_time_s
    )
    budget_exceeded = collision & (
        next_event_count > max_nonterminal_events
    )
    terminal_collision = collision & (
        terminate_immediately
        | severe_collision
        | sustained_collision
        | budget_exceeded
    )
    nonterminal_collision_event = collision_event & ~terminal_collision

    return {
        "collision": collision,
        "collision_event": collision_event,
        "nonterminal_collision_event": nonterminal_collision_event,
        "terminal_collision": terminal_collision,
        "severe_collision": severe_collision,
        "sustained_collision": sustained_collision,
        "budget_exceeded": budget_exceeded,
        "contact_active": next_contact_active,
        "contact_duration_s": next_contact_duration,
        "episode_event_count": next_event_count,
    }
