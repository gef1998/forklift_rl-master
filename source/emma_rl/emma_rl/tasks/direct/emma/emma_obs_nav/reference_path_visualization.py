"""Optional PNG and USD views of the exact reset-time reference result.

No planner invocation here. Matplotlib and USD are imported only on demand.
"""
from pathlib import Path
import math

import numpy as np


COLORS = dict(forward=(.10, .75, .30), reverse=(1., .42, .05),
              heading=(.15, .45, 1.), gates=(.05, .70, .80),
              start=(.05, .55, .20), goal=(.85, .10, .65), stop=(.9, .12, .10))


def rectangle(pose, half_length, half_width):
    x, y, yaw = pose
    corners = np.array([[-half_length, -half_width], [half_length, -half_width],
                        [half_length, half_width], [-half_length, half_width]])
    c, s = math.cos(yaw), math.sin(yaw)
    return corners @ np.array([[c, s], [-s, c]]) + [x, y]


def reference_lines(plan, start, goal, gates):
    """Independent local XY segments, shared by PNG and the USD overlay."""
    lines = {key: [] for key in COLORS}
    if plan.success and len(plan.poses) > 1:
        xy = plan.poses[:, :2]
        segments = np.stack((xy[:-1], xy[1:]), axis=1)
        moving = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) > 1e-8
        for gear, name in ((1, 'forward'), (-1, 'reverse')):
            lines[name] = segments[moving & (plan.directions[1:] == gear)].tolist()
        # Red crosses mark explicit zero-speed events, not ordinary arc joins.
        rotating = ~moving & (np.abs((np.diff(plan.poses[:,2])+math.pi)%(2*math.pi)-math.pi)>1e-8)
        stop_indices = list(np.flatnonzero(rotating & ~np.r_[False,rotating[:-1]]))
        moving_ids = np.flatnonzero(plan.directions != 0)
        if len(moving_ids)>1:
            stop_indices.extend((moving_ids[1:][np.diff(plan.directions[moving_ids]) != 0]-1).tolist())
        for i in stop_indices:
            point=xy[i]
            for delta in (np.array([.10,.10]),np.array([.10,-.10])):
                lines['stop'].append([point-delta,point+delta])
        # Body-heading arrows, spaced by travelled distance; they intentionally
        # oppose the direction of travel during reverse motion.
        last = -1.
        for pose, arc in zip(plan.poses, plan.arc_lengths):
            if arc-last < .65:
                continue
            last = arc
            p = pose[:2]
            u = np.array([math.cos(pose[2]), math.sin(pose[2])])
            v = np.array([-u[1], u[0]])
            tip = p + .24*u
            lines['heading'].extend([[p, tip], [tip, tip-.08*u+.05*v],
                                     [tip, tip-.08*u-.05*v]])
    for x, y, half_gap, _ in gates:
        lines['gates'].append([[x-half_gap, y], [x+half_gap, y]])
    for name, point in (('start', np.asarray(start)[:2]), ('goal', np.asarray(goal)[:2])):
        for delta in (np.array([.16, .16]), np.array([.16, -.16])):
            lines[name].append([point-delta, point+delta])
    return {name: np.asarray(value, dtype=float).reshape(-1, 2, 2) for name, value in lines.items()}


def export_reference_png(filename, world, plan, start, goal, gates, planner_cfg, *, env_id, level):
    """Render the full actual layout, including an explicitly labelled failure."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Circle, Rectangle
    from matplotlib.lines import Line2D

    fig = Figure(figsize=(9, 10), dpi=160, facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.16)
    half = world.inner_half
    ax.add_patch(Rectangle((-half, -half), 2*half, 2*half,
                           facecolor='#f7f9fc', edgecolor='#495464', linewidth=3))
    for x, y, yaw, hx, hy in world.boxes:
        ax.add_patch(Polygon(rectangle((x, y, yaw), hx, hy),
                             facecolor='#8994a4', edgecolor='#455263', linewidth=.8))
    for x, y, radius in world.circles:
        ax.add_patch(Circle((x, y), radius, facecolor='#a3adba', edgecolor='#455263'))
    for name, segments in reference_lines(plan, start, goal, gates).items():
        ax.add_collection(LineCollection(segments, colors=[COLORS[name]],
                                         linewidths=2.2 if name in ('forward', 'reverse') else 1.2,
                                         linestyles='dashed' if name == 'gates' else 'solid', zorder=4))
    if plan.success:
        last = -1.
        for pose, arc in zip(plan.poses, plan.arc_lengths):
            if arc-last >= .85:
                last = arc
                ax.add_patch(Polygon(rectangle(pose, planner_cfg.half_length, planner_cfg.half_width),
                                     fill=False, edgecolor='#3d75be', alpha=.4, linewidth=.8, zorder=3))
    ax.add_patch(Polygon(rectangle(start, planner_cfg.half_length, planner_cfg.half_width),
                         fill=False, edgecolor=COLORS['start'], linewidth=1.7, zorder=5))
    for label, point in (('START', start), ('GOAL', goal)):
        ax.annotate(label, point[:2], xytext=(8, 7), textcoords='offset points',
                    fontsize=9, weight='bold', zorder=6)
    for i, (x, y, gap, _) in enumerate(gates):
        ax.annotate(f'G{i+1}: {2*gap:.2f} m', (x, y), xytext=(0, 10),
                    textcoords='offset points', fontsize=8, ha='center', color='#006b78',
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1), zorder=6)
    status = ('VALID / STOP REQUIRED' if plan.requires_stop else 'VALID / NO STOP') if plan.success else f'FAILED: {plan.reason}'
    length = float(plan.arc_lengths[-1]) if plan.success and len(plan.poses) else 0.
    ax.set_title(f'Reference path | env {env_id} | level {level} | {status}\n'
                 f'{len(plan.poses)} poses | {length:.2f} m | {1000*plan.seconds:.1f} ms\n'
                 f'R_min={planner_cfg.min_turning_radius:.2f} m | rotations={plan.rotation_count} | gear changes={plan.gear_changes}',
                 fontsize=12, pad=15)
    ax.set(xlim=(-half-.3, half+.3), ylim=(-half-.3, half+.3),
           xlabel='Local X (m)', ylabel='Local Y (m)', aspect='equal')
    ax.grid(alpha=.15)
    handles = [Line2D([0], [0], color=COLORS[key], lw=2, label=label) for key, label in
               (('forward', 'Forward'), ('reverse', 'Reverse'), ('heading', 'Body heading'),
                ('gates', 'Gate opening'), ('stop', 'Stop / rotate or change gear'))]
    handles.append(Line2D([0], [0], color='#3d75be', alpha=.5, lw=1, label='Robot footprint'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    fig.text(.5, .025, 'Geometric reference, not a timed command trajectory. Footprints exclude the extra collision margin.',
             ha='center', fontsize=8, color='#526071')
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    fig.clear()
    return filename


class ReferencePathOverlay:
    """Own non-physical USD curves outside cloned envs; no global debug clear."""
    root = '/World/ReferencePathDebug'

    def __init__(self, stage):
        self.stage = stage

    def update(self, env_id, origin, plan, start, goal, gates, height=.12):
        from pxr import Gf, UsdGeom
        path = f'{self.root}/env_{env_id}'
        root = UsdGeom.Xform.Define(self.stage, path)
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*map(float, origin)))
        root.MakeVisible()
        root.GetPrim().SetCustomDataByKey('reference_status', 'valid' if plan.success else plan.reason)
        root.GetPrim().SetCustomDataByKey('requires_stop', bool(getattr(plan,'requires_stop',False)))
        for name, segments in reference_lines(plan, start, goal, gates).items():
            curves = UsdGeom.BasisCurves.Define(self.stage, f'{path}/{name}')
            curves.CreateTypeAttr('linear')
            curves.CreateWrapAttr('nonperiodic')
            curves.CreateCurveVertexCountsAttr([2]*len(segments))
            xy = segments.reshape(-1, 2)
            points = np.c_[xy, np.full(len(xy), height)]
            curves.CreatePointsAttr(points.astype(np.float32).tolist())
            curves.CreateWidthsAttr([.035 if name in ('forward', 'reverse') else .022])
            curves.SetWidthsInterpolation('constant')
            curves.CreateDisplayColorAttr([COLORS[name]])
            curves.CreateDisplayOpacityAttr([1.])
            if len(segments):
                curves.MakeVisible()
            else:
                curves.MakeInvisible()  # Clear a previous successful path on failure.

    def hide(self, env_id):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(f'{self.root}/env_{env_id}')
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
