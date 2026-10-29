"""Shared finite LiDAR ground covering the actual cloned environment bounds."""
import math

import numpy as np
import torch
import trimesh


def make_lidar_ground(env_origins: torch.Tensor | np.ndarray, arena_size: float) -> trimesh.Trimesh:
    """Cover every square arena using its actual XY origin, including offsets.

    Environment count need not be a perfect square and layouts may be irregular.
    Preserve the existing slab: top z=0, bottom z=-0.02. This creates only the
    sensor mesh; it does not resize the physical ground or individual arenas.
    """
    if not math.isfinite(arena_size) or arena_size <= 0:
        raise ValueError("arena_size must be finite and positive")
    if isinstance(env_origins, torch.Tensor):
        env_origins = env_origins.detach().cpu().numpy()
    origins = np.asarray(env_origins, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[0] == 0 or origins.shape[1] not in (2,3):
        raise ValueError("env_origins must have nonempty shape (N, 2) or (N, 3)")
    xy = origins[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError("environment XY origins must be finite")
    lower = xy.min(axis=0)-arena_size/2
    upper = xy.max(axis=0)+arena_size/2
    center = (lower+upper)/2
    ground = trimesh.creation.box(extents=[*(upper-lower), .02])
    ground.apply_translation([*center, -.01])
    return ground
