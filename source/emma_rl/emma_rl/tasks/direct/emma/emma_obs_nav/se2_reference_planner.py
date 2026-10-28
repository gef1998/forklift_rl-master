"""Offline differential-drive Hybrid A* for privileged training references.

Continuous SE(2) states, hashed discretization, exact unicycle primitives and
curvature-bounded tangent connections. No Isaac/ROS dependency. Geometric
feasibility only: a path is not a time-parameterized expert action sequence.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import heapq
import math
import time

import numpy as np


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class PlannerConfig:
    half_length: float = .425
    half_width: float = .275
    margin: float = .01
    xy_resolution: float = .02
    yaw_resolution_deg: float = 2.
    primitive_length: float = .15
    primitive_turn_deg: float = 15.
    min_turning_radius: float = .50
    analytic_radius_factors: tuple[float, ...] = (1., 1.1, 1.5, 2.)
    inplace_rotation_fallback: bool = True
    inplace_rotation_penalty: float = 2.0
    collision_step: float = .01
    collision_angle_deg: float = 1.
    output_step: float = .05
    output_angle_deg: float = 5.
    reverse_penalty: float = 1.5
    rotation_cost: float = .35  # equivalent metres per radian; never zero
    heuristic_weight: float = 1.5
    max_expansions: int = 12000
    max_seconds: float = .75
    cache_size: int = 512
    max_points: int = 1024

    def __post_init__(self):
        factors = tuple(self.analytic_radius_factors)
        if (not factors or factors[0] != 1. or any(not math.isfinite(f) or f < 1 for f in factors)
                or any(a >= b for a,b in zip(factors,factors[1:]))):
            raise ValueError('analytic_radius_factors must start at 1 and increase finitely')
        object.__setattr__(self, 'analytic_radius_factors', factors)
        for name in ('half_length', 'half_width', 'xy_resolution', 'yaw_resolution_deg',
                     'primitive_length', 'primitive_turn_deg', 'collision_step',
                     'collision_angle_deg', 'output_step', 'output_angle_deg',
                     'rotation_cost', 'heuristic_weight', 'max_seconds',
                     'min_turning_radius', 'inplace_rotation_penalty'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.margin) or self.margin < 0 or self.reverse_penalty < 1:
            raise ValueError('invalid margin/reverse penalty')
        if min(self.max_expansions, self.cache_size) < 1 or self.max_points < 2:
            raise ValueError('planner budgets must be positive')
        if self.yaw_resolution_deg > 180 or not math.isfinite(self.reverse_penalty):
            raise ValueError('invalid heading resolution/reverse penalty')


class CollisionWorld:
    """OBBs [x,y,yaw,hx,hy], circles [x,y,r], and inside arena boundary."""
    def __init__(self, boxes=(), circles=(), inner_half=4.85):
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
        self.circles = np.asarray(circles, dtype=float).reshape(-1, 3)
        self.inner_half = float(inner_half)
        if (not np.isfinite(self.boxes).all() or not np.isfinite(self.circles).all()
                or not math.isfinite(inner_half) or inner_half <= 0
                or (self.boxes[:, 3:] <= 0).any() or (self.circles[:, 2:] <= 0).any()):
            raise ValueError('invalid collision geometry')
        self.axes = np.stack((np.cos(self.boxes[:, 2]), np.sin(self.boxes[:, 2])), axis=-1)
        self.normals = np.stack((-self.axes[:, 1], self.axes[:, 0]), axis=-1)
        h = hashlib.sha256()
        for a in (self.boxes, self.circles, np.array([inner_half])):
            h.update(a.tobytes())
        self.fingerprint = h.hexdigest()

    def valid(self, poses, cfg, padding=0.):
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        u, v = np.stack((c, s), -1), np.stack((-s, c), -1)
        hl, hw = cfg.half_length + cfg.margin + padding, cfg.half_width + cfg.margin + padding
        extent = np.stack((hl * abs(c) + hw * abs(s), hl * abs(s) + hw * abs(c)), -1)
        valid = np.isfinite(poses).all(-1) & (abs(poses[:, :2]) + extent < self.inner_half).all(-1)
        if len(self.boxes):
            d = self.boxes[None, :, :2] - poses[:, None, :2]
            du, dv = np.einsum('nbi,ni->nb', d, u), np.einsum('nbi,ni->nb', d, v)
            a, b = u @ self.axes.T, u @ self.normals.T
            e, f = v @ self.axes.T, v @ self.normals.T
            sep_u = abs(du) > hl + abs(a) * self.boxes[:, 3] + abs(b) * self.boxes[:, 4]
            sep_v = abs(dv) > hw + abs(e) * self.boxes[:, 3] + abs(f) * self.boxes[:, 4]
            sep_a = abs(np.einsum('nbi,bi->nb', d, self.axes)) > self.boxes[:, 3] + hl * abs(a) + hw * abs(e)
            sep_b = abs(np.einsum('nbi,bi->nb', d, self.normals)) > self.boxes[:, 4] + hl * abs(b) + hw * abs(f)
            valid &= (sep_u | sep_v | sep_a | sep_b).all(-1)
        if len(self.circles):
            d = self.circles[None, :, :2] - poses[:, None, :2]
            qx = np.maximum(abs(np.einsum('nbi,ni->nb', d, u)) - hl, 0.)
            qy = np.maximum(abs(np.einsum('nbi,ni->nb', d, v)) - hw, 0.)
            valid &= (qx*qx + qy*qy > self.circles[:, 2]**2).all(-1)
        return valid


def motion(start, distance, turn, linear_step, angular_step):
    """Sample an exact constant-curvature primitive, including pure rotation."""
    count = max(1, int(math.ceil(abs(distance)/linear_step)), int(math.ceil(abs(turn)/angular_step)))
    f = np.linspace(0., 1., count + 1)
    theta = start[2] + turn * f
    if abs(turn) < 1e-10:
        x = start[0] + distance * f * math.cos(start[2])
        y = start[1] + distance * f * math.sin(start[2])
    else:
        radius = distance / turn
        x = start[0] + radius * (np.sin(theta) - math.sin(start[2]))
        y = start[1] - radius * (np.cos(theta) - math.cos(start[2]))
    return np.stack((x, y, wrap(theta)), -1)


@dataclass
class PlanResult:
    poses: np.ndarray
    directions: np.ndarray  # primitive gear ending at each sample; 0 = rotate
    success: bool
    reason: str
    expansions: int = 0
    seconds: float = 0.
    fallback_attempted: bool = False
    failed_leg: int = -1  # index of failed bank leg; -1 on success / standalone query

    @property
    def arc_lengths(self):
        if not len(self.poses):
            return np.empty(0)
        return np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1))]

    @property
    def rotation_count(self):
        if len(self.poses) < 2:
            return 0
        stationary = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1) < 1e-8
        rotating = stationary & (abs(wrap(np.diff(self.poses[:, 2]))) > 1e-8)
        return int(np.count_nonzero(rotating & ~np.r_[False, rotating[:-1]]))

    @property
    def gear_changes(self):
        moving = self.directions[self.directions != 0]
        return int(np.count_nonzero(np.diff(moving)))

    @property
    def requires_stop(self):
        return bool(self.rotation_count or self.gear_changes)

    @property
    def minimum_moving_radius(self):
        if len(self.poses) < 2:
            return math.inf
        chords = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1)
        angles = abs(wrap(np.diff(self.poses[:, 2])))
        curved = (chords > 1e-8) & (angles > 1e-8)
        return float(np.min(chords[curved]/(2*np.sin(angles[curved]/2)))) if curved.any() else math.inf


def tangent_connections(start, goal, goal_yaw, radius):
    """Geometric Dubins candidates, generated in either one constant gear.

    Construct common tangents of start/goal turning circles (CSC) and
    intersections of radius-2R circles (CCC). Commands are (arc length, yaw
    change); no heading reset, smoothing shortcut or endpoint snap is used.
    Reverse candidates use the travel heading yaw+pi, with signed distance.
    Mixed-gear connections are discovered by the Hybrid A* search.
    """
    def positive(angle):
        value = angle % (2*math.pi)
        return 0. if value < 1e-10 or 2*math.pi-value < 1e-10 else value
    def normal(angle):
        return np.array([-math.sin(angle), math.cos(angle)])
    commands = []
    for gear in (1, -1):
        h0 = start[2] + (math.pi if gear < 0 else 0.)
        h1 = goal_yaw + (math.pi if gear < 0 else 0.)
        for first in (1, -1):
            c0 = start[:2] + first*radius*normal(h0)
            for last in (1, -1):
                c1 = goal + last*radius*normal(h1)
                delta = c1-c0
                distance = float(np.linalg.norm(delta))
                offset = (first-last)*radius
                if distance < 1e-10:
                    if first == last:
                        angle = positive(first*(h1-h0))
                        commands.append([(gear*radius*angle, first*angle)])
                    continue
                if abs(offset) <= distance+1e-10:
                    tangent = math.atan2(delta[1],delta[0]) + math.asin(np.clip(offset/distance,-1,1))
                    a = positive(first*(tangent-h0))
                    b = positive(last*(h1-tangent))
                    straight = math.sqrt(max(0., distance*distance-offset*offset))
                    commands.append([(gear*radius*a, first*a), (gear*straight, 0.),
                                     (gear*radius*b, last*b)])
                if first != last or distance > 4*radius+1e-10:
                    continue
                midpoint = (c0+c1)/2
                height = math.sqrt(max(0., 4*radius*radius-distance*distance/4))
                perpendicular = np.array([-delta[1],delta[0]])/distance
                for side in (1, -1):
                    middle = midpoint+side*height*perpendicular
                    p0, p1 = (c0+middle)/2, (c1+middle)/2
                    t0 = math.atan2(p0[1]-c0[1],p0[0]-c0[0])+first*math.pi/2
                    t1 = math.atan2(p1[1]-middle[1],p1[0]-middle[0])-first*math.pi/2
                    angles = (positive(first*(t0-h0)), positive(-first*(t1-t0)), positive(first*(h1-t1)))
                    commands.append([(gear*radius*a, turn*a) for a,turn in zip(angles,(first,-first,first))])
    return commands


class HybridAStar:
    def __init__(self, world, cfg=PlannerConfig()):
        self.world, self.cfg = world, cfg

    def obstacle_heuristic(self, goal):
        """Coarse holonomic distance guidance only, never a collision verdict.

        Under-inflation leaves narrow doors visible to the heuristic. Fine SE2
        sweep checks still use the complete rectangle and configured margin.
        """
        resolution = .08
        size = int(math.ceil(2*self.world.inner_half/resolution))
        origin = -self.world.inner_half
        xx, yy = np.meshgrid(origin+(np.arange(size)+.5)*resolution,
                             origin+(np.arange(size)+.5)*resolution, indexing='ij')
        proxy = PlannerConfig(half_length=.15, half_width=.15, margin=0.)
        free = self.world.valid(np.c_[xx.ravel(), yy.ravel(), np.zeros(size*size)], proxy).reshape(size, size)
        target = np.clip(((goal-origin)/resolution).astype(int), 0, size-1)
        distances = np.full((size, size), np.inf)
        distances[tuple(target)] = 0.
        queue = [(0., *target)]
        moves = [(i, j, math.hypot(i, j)*resolution) for i in (-1,0,1) for j in (-1,0,1) if i or j]
        while queue:
            cost, i, j = heapq.heappop(queue)
            if cost > distances[i, j]:
                continue
            for di, dj, length in moves:
                ni, nj = i+di, j+dj
                if 0 <= ni < size and 0 <= nj < size and free[ni, nj] and cost+length < distances[ni, nj]:
                    distances[ni, nj] = cost+length
                    heapq.heappush(queue, (cost+length, ni, nj))
        def estimate(pose):
            i, j = np.clip(((pose[:2]-origin)/resolution).astype(int), 0, size-1)
            return distances[i, j] if np.isfinite(distances[i, j]) else float(np.linalg.norm(pose[:2]-goal))
        return estimate

    def checked_motion(self, start, distance, turn, *, allow_inplace=False):
        cfg = self.cfg
        if abs(distance) < 1e-10:
            if abs(turn) > 1e-10 and not allow_inplace:
                return None
        elif abs(turn) > abs(distance)/cfg.min_turning_radius + 1e-10:
            return None
        fine = motion(start, distance, turn, cfg.collision_step, math.radians(cfg.collision_angle_deg))
        # Bound every point of the moving footprint between adjacent samples.
        # This conservative swept-volume padding also protects in-place turns.
        n = len(fine) - 1
        padding = (abs(distance) + math.hypot(cfg.half_length + cfg.margin,
                                            cfg.half_width + cfg.margin) * abs(turn)) / (2*n)
        if not self.world.valid(fine, cfg, padding).all():
            # The 1 cm / 1 deg sweep bound can reject a physically clear turn
            # near a side wall. Refine only these near-contact cases. Every
            # accepted interval still has its own conservative sweep bound;
            # never accept merely because its unpadded sample points are free.
            if not self.world.valid(fine, cfg).all():
                return None
            for factor in (2,4):
                fine = motion(start,distance,turn,cfg.collision_step/factor,
                              math.radians(cfg.collision_angle_deg)/factor)
                padding = (abs(distance)+math.hypot(cfg.half_length+cfg.margin,
                           cfg.half_width+cfg.margin)*abs(turn))/(2*(len(fine)-1))
                if self.world.valid(fine,cfg,padding).all():
                    break
                if not self.world.valid(fine,cfg).all():
                    return None
            else:
                return None
        return motion(start, distance, turn, cfg.output_step, math.radians(cfg.output_angle_deg))

    def connector(self, start, goal, goal_yaw=None, *, allow_inplace=False, clearance_hints=True,
                  radius_factor=1.):
        """Try tangent straight/arc expansions in cost order; check every sweep."""
        goal = np.asarray(goal, dtype=float)
        delta = goal-start[:2]
        distance = float(np.linalg.norm(delta))
        phi = math.atan2(delta[1], delta[0]) if distance > 1e-10 else start[2]
        candidates = []
        headings = (goal_yaw,) if goal_yaw is not None else (phi, start[2])
        for heading in headings:
            candidates.extend(tangent_connections(start, goal, heading, self.cfg.min_turning_radius*radius_factor))
        for gear in (1, -1):
            travel = start[2] + (math.pi if gear < 0 else 0.)
            a = math.cos(travel)*delta[0]+math.sin(travel)*delta[1]
            b = -math.sin(travel)*delta[0]+math.cos(travel)*delta[1]
            if goal_yaw is None:
                if abs(b) < 1e-9 and a >= 0:
                    candidates.append([(gear*a, 0.)])
                elif abs(b) > 1e-9:
                    turn = 2*math.atan2(b,a)
                    radius = (a*a+b*b)/(2*b)
                    if abs(radius) >= self.cfg.min_turning_radius and abs(turn) <= math.pi+1e-9:
                        candidates.append([(gear*radius*turn, turn)])
            # A tangent pair of equal/opposite arcs can use a gentler radius
            # than R_min, without fixing artificial approach/departure poses.
            elif abs(wrap(goal_yaw-start[2])) < 1e-8 and a > 0 and abs(b) > 1e-9:
                turn = 2*math.atan2(b,a)
                radius = (a*a+b*b)/(4*b)
                if abs(radius) >= self.cfg.min_turning_radius:
                    length = radius*turn
                    candidates.append([(gear*length, turn), (gear*length, -turn)])
            if allow_inplace:
                heading = phi + (math.pi if gear < 0 else 0.)
                commands = [(0.,wrap(heading-start[2])), (gear*distance,0.)]
                if goal_yaw is not None:
                    commands.append((0.,wrap(goal_yaw-heading)))
                candidates.append(commands)
        def cost(commands):
            return sum(abs(length)*(self.cfg.reverse_penalty if length < 0 else 1.)
                       + self.cfg.rotation_cost*abs(turn)
                       + (self.cfg.inplace_rotation_penalty if abs(length) < 1e-10 and abs(turn) > 1e-10 else 0.)
                       for length,turn in commands)
        for commands in sorted(candidates, key=cost):
            pose = np.array(start, dtype=float)
            points, gears = [pose[None]], [np.array([0],dtype=np.int8)]
            for length,turn in commands:
                if abs(length)+abs(turn) < 1e-10:
                    continue
                part = self.checked_motion(pose,length,turn,allow_inplace=allow_inplace)
                if part is None:
                    break
                points.append(part[1:])
                gears.append(np.full(len(part)-1,np.sign(length),dtype=np.int8))
                pose=part[-1]
            else:
                if np.linalg.norm(pose[:2]-goal) > 1e-7:
                    continue
                if goal_yaw is not None and abs(wrap(pose[2]-goal_yaw)) > 1e-7:
                    continue
                return cost(commands),np.concatenate(points),np.concatenate(gears)
        if radius_factor == 1.:
            # R_min is a lower bound, not the only legal analytic turn. Larger
            # circles often clear a long rectangular body's outer corner.
            for factor in self.cfg.analytic_radius_factors[1:]:
                candidate = self.connector(start,goal,goal_yaw,allow_inplace=allow_inplace,
                                           clearance_hints=False,radius_factor=factor)
                if candidate is not None:
                    return candidate
        if clearance_hints:
            # Optional straight run-in/run-out candidates protect the body tail
            # while crossing a narrow wall. These are candidate connections,
            # not extra mandatory gate poses; all sweeps remain checked.
            lead = self.cfg.half_length+self.cfg.margin+.15
            hinted = []
            suffix_options = (0.,lead) if goal_yaw is not None else (0.,)
            for gear in (1,-1):
                for before in (0.,lead):
                    first = self.checked_motion(start,gear*before,0.)
                    if first is None:
                        continue
                    for after in suffix_options:
                        if before+after < 1e-10:
                            continue
                        target = goal if not after else goal-gear*after*np.array([math.cos(goal_yaw),math.sin(goal_yaw)])
                        middle = self.connector(first[-1],target,goal_yaw,
                                                allow_inplace=allow_inplace,clearance_hints=False)
                        if middle is None:
                            continue
                        last = self.checked_motion(middle[1][-1],gear*after,0.)
                        if last is None:
                            continue
                        poses = np.concatenate((first,middle[1][1:],last[1:]))
                        gears = np.r_[np.full(len(first),gear if before else 0,dtype=np.int8),
                                      middle[2][1:],np.full(len(last)-1,gear if after else 0,dtype=np.int8)]
                        score = middle[0]+(before+after)*(self.cfg.reverse_penalty if gear<0 else 1.)
                        hinted.append((score,poses,gears))
            if hinted:
                return min(hinted,key=lambda item:item[0])
        return None

    def plan(self, start, goal, goal_yaw=None):
        """Search moving arcs first; permit stop/rotate only in a second pass.

        Each pass has its own finite time/expansion budget. The radius bound
        applies to every moving primitive and analytic connector in both passes.
        """
        started = time.perf_counter()
        smooth = self._plan(start, goal, goal_yaw, allow_inplace=False)
        if smooth.success or not self.cfg.inplace_rotation_fallback or smooth.reason in (
                'invalid_query', 'goal_outside_arena', 'start_collision', 'goal_collision'):
            return smooth
        fallback = self._plan(start, goal, goal_yaw, allow_inplace=True)
        fallback.expansions += smooth.expansions
        fallback.seconds = time.perf_counter()-started
        fallback.fallback_attempted = True
        if fallback.success and fallback.rotation_count:
            fallback.reason = 'rotation_fallback'
        return fallback

    def _plan(self, start, goal, goal_yaw=None, *, allow_inplace):
        started = time.perf_counter()
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        cfg = self.cfg
        def result(poses, gears, ok, reason, expanded=0):
            return PlanResult(np.asarray(poses).reshape(-1, 3), np.asarray(gears, dtype=np.int8), ok,
                              reason, expanded, time.perf_counter()-started)
        if (start.shape != (3,) or goal.shape != (2,) or not np.isfinite(start).all()
                or not np.isfinite(goal).all() or (goal_yaw is not None and not math.isfinite(goal_yaw))):
            return result([], [], False, 'invalid_query')
        if (abs(goal) >= self.world.inner_half).any():
            return result([], [], False, 'goal_outside_arena')
        if not self.world.valid(start[None], cfg)[0]:
            return result([], [], False, 'start_collision')
        if goal_yaw is not None and not self.world.valid([[*goal, goal_yaw]], cfg)[0]:
            return result([], [], False, 'goal_collision')
        direct = self.connector(start, goal, goal_yaw, allow_inplace=allow_inplace)
        if direct is not None:
            return result(direct[1], direct[2], True, 'analytic')
        angle_step = math.radians(cfg.primitive_turn_deg)
        # Translation arcs and optional fallback rotations, including reverse. Continuous
        # endpoint poses are retained; quantization is only for duplicate keys.
        turns = (0., -angle_step, angle_step, -2*angle_step, 2*angle_step,
                 math.radians(-2), math.radians(2), math.radians(-5), math.radians(5))
        max_turn = cfg.primitive_length/cfg.min_turning_radius
        turns = sorted(set(max(-max_turn,min(max_turn,t)) for t in turns))
        primitives = [(gear*cfg.primitive_length, turn) for gear in (1, -1) for turn in turns]
        if allow_inplace:
            primitives += [(0., angle_step), (0., -angle_step)]
        def key(p):
            return (int(round(p[0]/cfg.xy_resolution)), int(round(p[1]/cfg.xy_resolution)),
                    int(round(wrap(p[2])/math.radians(cfg.yaw_resolution_deg))) % int(round(360/cfg.yaw_resolution_deg)))
        distance_estimate = self.obstacle_heuristic(goal)
        def heuristic(p):
            return distance_estimate(p) + (cfg.rotation_cost*abs(wrap(p[2]-goal_yaw)) if goal_yaw is not None else 0.)
        # Immutable nodes: improvements cannot change an already expanded
        # parent's pose while descendants still refer to its old edge.
        nodes = [(start, -1, None, None, 0.)]
        best = {key(start): 0.}
        heap = [(cfg.heuristic_weight*heuristic(start), 0)]
        expanded = 0
        while heap and expanded < cfg.max_expansions:
            if time.perf_counter()-started > cfg.max_seconds:
                return result([], [], False, 'time_budget', expanded)
            _, idx = heapq.heappop(heap)
            pose, parent, edge, gear, g = nodes[idx]
            if g > best.get(key(pose), math.inf) + 1e-9:
                continue
            expanded += 1
            connection = self.connector(pose, goal, goal_yaw, allow_inplace=allow_inplace)
            if connection is not None:
                parts, gears = [connection[1][1:]], [connection[2][1:]]
                while nodes[idx][1] != -1:
                    _, parent, edge, gear, _ = nodes[idx]
                    parts.append(edge[1:]); gears.append(np.full(len(edge)-1, gear, dtype=np.int8)); idx = parent
                parts.append(start[None]); gears.append(np.array([0], dtype=np.int8))
                return result(np.concatenate(parts[::-1]), np.concatenate(gears[::-1]), True, 'search', expanded)
            for distance, turn in primitives:
                edge = self.checked_motion(pose, distance, turn, allow_inplace=allow_inplace)
                if edge is None:
                    continue
                q = edge[-1]
                cost = g + abs(distance)*(cfg.reverse_penalty if distance < 0 else 1.) + cfg.rotation_cost*abs(turn)
                if abs(distance) < 1e-10:
                    cost += cfg.inplace_rotation_penalty
                k = key(q)
                if cost >= best.get(k, math.inf)-1e-9:
                    continue
                best[k] = cost
                nodes.append((q, idx, edge, int(np.sign(distance)), cost))
                heapq.heappush(heap, (cost + cfg.heuristic_weight*heuristic(q), len(nodes)-1))
        return result([], [], False, 'expansion_budget' if heap else 'no_path', expanded)


class DetourReferenceBank:
    """Lazy cache of checked gate-to-gate legs; actual start/yaw are never cached away."""
    def __init__(self, cfg=PlannerConfig()):
        self.cfg = cfg
        self.cache = OrderedDict()
        self.stats = dict(requests=0, successes=0, failures=0, cache_hits=0, expansions=0, seconds=0.)
        self.stats.update(stop_paths=0, rotation_paths=0)
        self.failure_reasons = {}

    def plan(self, world, start, goal, gates):
        # gates: [x, y, half_gap, approach_guard], ordered from start to goal.
        started = time.perf_counter()
        self.stats['requests'] += 1
        sign = 1 if goal[1] >= start[1] else -1
        heading = sign*math.pi/2
        # Keep the gate centre/tangent constraint, remove artificial fixed
        # approach/exit poses. Collision sweeps decide where turning can start.
        anchors = [[x, y, heading] for x, y, _, _ in gates]
        anchors.append([*goal, None])
        planner = HybridAStar(world, self.cfg)
        pose = np.asarray(start, dtype=float)
        points, gears, expansions, reason = [pose[None]], [np.array([0], dtype=np.int8)], 0, 'ok'
        fallback_attempted = False
        for index, anchor in enumerate(anchors):
            # Only exact canonical gate legs are cached. Key includes all
            # geometry and planner settings; endpoints/headings are never rounded.
            cacheable = 0 < index < len(anchors)-1
            key = (world.fingerprint, self.cfg, tuple(pose), tuple(anchor))
            part = self.cache.get(key) if cacheable else None
            if part is not None:
                self.stats['cache_hits'] += 1; self.cache.move_to_end(key)
            else:
                part = planner.plan(pose, np.asarray(anchor[:2]), anchor[2])
                expansions += part.expansions
                if cacheable and part.success:
                    self.cache[key] = part
                    while len(self.cache) > self.cfg.cache_size:
                        self.cache.popitem(last=False)
            if not part.success:
                fallback_attempted |= part.fallback_attempted
                reason = part.reason; break
            fallback_attempted |= part.fallback_attempted
            # Canonical endpoint snap is only floating-point noise, not a
            # geometric shortcut. It stabilizes exact cache keys across resets.
            if np.linalg.norm(part.poses[-1, :2]-anchor[:2]) > 1e-7:
                reason = 'endpoint_mismatch'; break
            part.poses[-1, :2] = anchor[:2]
            if anchor[2] is not None:
                if abs(wrap(part.poses[-1, 2]-anchor[2])) > 1e-7:
                    reason = 'heading_mismatch'; break
                part.poses[-1, 2] = anchor[2]
            points.append(part.poses[1:]); gears.append(part.directions[1:]); pose = part.poses[-1]
        poses, directions = np.concatenate(points), np.concatenate(gears)
        if len(poses) > self.cfg.max_points:
            reason = 'point_budget'
        success = reason == 'ok'
        self.stats['successes' if success else 'failures'] += 1
        self.stats['expansions'] += expansions
        seconds = time.perf_counter()-started; self.stats['seconds'] += seconds
        if not success:
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
        result = PlanResult(poses if success else np.empty((0, 3)), directions if success else np.empty(0, np.int8),
                            success, reason, expansions, seconds, fallback_attempted)
        if not success:
            result.failed_leg = index
        if success:
            self.stats['stop_paths'] += int(result.requires_stop)
            self.stats['rotation_paths'] += int(result.rotation_count > 0)
        return result


def world_from_detour_layout(layout, config):
    """Read the same continuous wall lengths and mixed shapes used by PhysX/Warp."""
    boxes, circles = [], []
    for pose, active, length in zip(layout['collision_wall_poses'], layout['collision_wall_active'], layout['collision_wall_lengths']):
        if active:
            boxes.append([*pose, length/2, config.interior_wall_thickness/2])
    for i in np.flatnonzero(layout['active']):
        x, y = layout['positions'][i]
        if config._obstacle_types[i] == 'cylinder':
            circles.append([x, y, config._obstacle_radii[i]])
        else:
            sx, sy = config._obstacle_cube_sizes[i]
            boxes.append([x, y, layout['yaws'][i], sx/2, sy/2])
    return CollisionWorld(boxes, circles, config.arena_size/2-config.wall_thickness)


def ordered_detour_gates(layout, level, start, goal):
    lane = int(layout['curriculum_lane_ids'][level])
    centers = layout['gate_centers'][lane]
    widths = layout['passage_widths'][lane]
    low, high = sorted((start[1], goal[1]))
    gates = [[x, y, width/2, layout['gate_guard']] for (x, y), width in zip(centers, widths) if low < y < high]
    gates.sort(key=lambda g: g[1], reverse=bool(goal[1] < start[1]))
    return np.asarray(gates, dtype=float).reshape(-1, 4)


def gate_arc_lengths(poses, gates):
    lengths = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    prefix = np.r_[0., np.cumsum(lengths)]
    arcs = []
    for x, y, _, _ in gates:
        d = np.linalg.norm(poses[:, :2] - (x, y), axis=1)
        i = int(d.argmin())
        if d[i] > 1e-5:
            raise ValueError('planned route did not reach its gate')
        arcs.append(prefix[i])
    return np.asarray(arcs)
