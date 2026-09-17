"""Outcome-based curriculum state, with no dependency on the simulator."""
from dataclasses import dataclass
import math


@dataclass
class NavigationCurriculumCfg:
    enabled: bool = True
    initial_level: int = 0
    evaluation_level: int = 3
    evaluation_noise_scale: float = 0.0
    replay_probability: float = 0.20
    noise_ranges: tuple = ((0.0, 0.05), (0.05, 0.15), (0.10, 0.35), (0.35, 1.0))
    window_size: int = 512
    required_windows: int = 2
    min_level_steps: int = 2000
    promote_success_rate: float = 0.80
    promote_failure_rate: float = 0.10
    demote_success_rate: float = 0.35
    demote_failure_rate: float = 0.40


class NavigationCurriculum:
    """Decide from independent outcome windows of current-level episodes only.

    Replay episodes and episodes started before the latest level transition are
    excluded. Changes affect future resets; the caller owns per-episode levels.
    The state_dict API allows skrl to checkpoint this state alongside the agent.
    """

    def __init__(self, cfg: NavigationCurriculumCfg, *, supported: bool = True):
        self.cfg = cfg
        self.enabled = cfg.enabled and supported
        if cfg.initial_level not in (0, 1, 2, 3) or cfg.evaluation_level not in (0, 1, 2, 3):
            raise ValueError("curriculum levels must be 0, 1, 2, or 3")
        for name in ("window_size", "required_windows", "min_level_steps"):
            value = getattr(cfg, name)
            if not isinstance(value, int) or value < (0 if name == "min_level_steps" else 1):
                raise ValueError(f"invalid curriculum {name}")
        for name in ("replay_probability", "evaluation_noise_scale", "promote_success_rate",
                     "promote_failure_rate", "demote_success_rate", "demote_failure_rate"):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (cfg.demote_success_rate >= cfg.promote_success_rate
                or cfg.promote_failure_rate >= cfg.demote_failure_rate):
            raise ValueError("curriculum thresholds must provide hysteresis")
        if len(cfg.noise_ranges) != 4 or any(
            len(r) != 2 or not 0 <= r[0] <= r[1] <= 1 for r in cfg.noise_ranges
        ):
            raise ValueError("four valid noise ranges are required")
        if cfg.replay_probability >= 1:
            raise ValueError("replay_probability must leave some current-level tasks")
        if any(cfg.noise_ranges[i][j] > cfg.noise_ranges[i + 1][j]
               for i in range(3) for j in range(2)):
            raise ValueError("noise ranges must increase with curriculum level")
        self.level = cfg.initial_level if self.enabled else cfg.evaluation_level
        self.generation = 0
        self.steps_since_change = 0
        self.promote_streak = self.demote_streak = 0
        self.window = []
        self.last_success_rate = self.last_failure_rate = 0.0

    def advance(self):
        if self.enabled:
            self.steps_since_change += 1

    def record(self, levels, generations, successes, failures):
        """Record completed episodes; return whether a level transition occurred."""
        if not self.enabled or self.steps_since_change < self.cfg.min_level_steps:
            return False
        if len({len(x) for x in (levels, generations, successes, failures)}) != 1:
            raise ValueError("curriculum outcome arrays must have matching lengths")
        for level, generation, success, failure in zip(levels, generations, successes, failures):
            if level != self.level or generation != self.generation:
                continue
            self.window.append((bool(success) and not failure, bool(failure)))
            if len(self.window) < self.cfg.window_size:
                continue
            self.last_success_rate = sum(x[0] for x in self.window) / len(self.window)
            self.last_failure_rate = sum(x[1] for x in self.window) / len(self.window)
            self.window.clear()  # Non-overlapping windows, not repeated checks of one batch.
            promote = (self.last_success_rate >= self.cfg.promote_success_rate
                       and self.last_failure_rate <= self.cfg.promote_failure_rate)
            demote = (self.last_success_rate <= self.cfg.demote_success_rate
                      or self.last_failure_rate >= self.cfg.demote_failure_rate)
            self.promote_streak = self.promote_streak + 1 if promote else 0
            self.demote_streak = self.demote_streak + 1 if demote else 0
            next_level = self.level
            if self.promote_streak >= self.cfg.required_windows:
                next_level = min(3, self.level + 1)
            elif self.demote_streak >= self.cfg.required_windows:
                next_level = max(0, self.level - 1)
            if next_level != self.level:
                self.level = next_level
                self.generation += 1
                self.steps_since_change = 0
                self.promote_streak = self.demote_streak = 0
                # Remaining outcomes belong to the previous generation.
                return True
        return False

    def state_dict(self):
        return dict(version=1, window_size=self.cfg.window_size, level=self.level, generation=self.generation,
                    steps_since_change=self.steps_since_change,
                    promote_streak=self.promote_streak, demote_streak=self.demote_streak,
                    window=list(self.window), last_success_rate=self.last_success_rate,
                    last_failure_rate=self.last_failure_rate)

    def load_state_dict(self, state):
        if not self.enabled:
            return  # Evaluation stays at its explicitly configured level.
        if state.get("version") != 1 or state.get("level") not in (0, 1, 2, 3):
            raise ValueError("unsupported curriculum checkpoint")
        for name in ("generation", "steps_since_change", "promote_streak", "demote_streak"):
            if not isinstance(state.get(name), int) or state[name] < 0:
                raise ValueError(f"invalid curriculum checkpoint {name}")
        for name in ("last_success_rate", "last_failure_rate"):
            if not math.isfinite(state[name]) or not 0 <= state[name] <= 1:
                raise ValueError(f"invalid curriculum checkpoint {name}")
        window = state.get("window", [])
        if not isinstance(window, (list, tuple)) or any(
            not isinstance(x, (list, tuple)) or len(x) != 2 for x in window
        ):
            raise ValueError("invalid curriculum checkpoint window")
        self.level = state["level"]
        for name in ("generation", "steps_since_change", "promote_streak", "demote_streak",
                     "last_success_rate", "last_failure_rate"):
            setattr(self, name, state[name])
        # Changing the window size at resume must not reuse a partial old window.
        self.window = [(bool(s) and not bool(f), bool(f)) for s, f in window]
        if state.get("window_size") != self.cfg.window_size or len(self.window) >= self.cfg.window_size:
            self.window.clear()
            self.promote_streak = self.demote_streak = 0
