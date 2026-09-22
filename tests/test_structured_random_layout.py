"""Run directly with the Isaac Lab Python; no simulator or pytest required."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/structured_random_geometry.py"
spec = importlib.util.spec_from_file_location("structured_random_geometry", MODULE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
RADII = np.array([.50 if i % 2 == 0 else math.hypot(.5, .5) / 2 for i in range(15)])


class StructuredLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layouts = [geometry.sample_connected_layout(RADII, seed=i) for i in range(64)]

    def test_seed_reproducibility_and_distribution(self):
        again = geometry.sample_connected_layout(RADII, seed=0)
        for name in ("positions", "active", "wall_poses", "wall_active", "spawn_pairs"):
            np.testing.assert_array_equal(again[name], self.layouts[0][name])
        signatures = {layout["wall_active"].tobytes() for layout in self.layouts}
        self.assertGreater(len(signatures), 30)
        self.assertGreater(len({int(x["active"].sum()) for x in self.layouts}), 3)

    def test_clearances_and_connected_spawn_banks(self):
        _, _, pitch = geometry.wall_graph(10, .15, 3)
        axis = np.linspace(-4.85, 4.85, math.ceil(9.7 / .15) + 1)
        xx, yy = np.meshgrid(axis, axis)
        points = np.stack((xx, yy), axis=-1)
        step = axis[1] - axis[0]
        for layout in self.layouts:
            p, active = layout["positions"], layout["active"]
            r = RADII[active]
            distance = np.linalg.norm(p[active, None] - p[active][None], axis=-1)
            edge = distance - r[:, None] - r[None]
            np.fill_diagonal(edge, np.inf)
            self.assertTrue((edge >= 1.5 - 1e-6).all())
            wall_clear = geometry.point_clearance(p, p, RADII, np.zeros_like(active),
                layout["wall_poses"], layout["wall_active"], layout["wall_lengths"], .15, 4.85)
            self.assertTrue((wall_clear[active] >= r + .30 - 1e-6).all())
            pairs = layout["spawn_pairs"]
            clear = geometry.point_clearance(pairs, p, RADII, active,
                layout["wall_poses"], layout["wall_active"], layout["wall_lengths"], .15, 4.85)
            self.assertTrue((clear[:, 0] >= .70).all())
            self.assertTrue((clear[:, 1] >= .60).all())
            self.assertTrue((np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=-1) > 5.2).all())
            grid_clear = geometry.point_clearance(points, p, RADII, active,
                layout["wall_poses"], layout["wall_active"], layout["wall_lengths"], .15, 4.85)
            free = grid_clear >= layout["navigation_clearance"] + step / math.sqrt(2)
            # Independent propagation from the first spawn verifies every bank
            # endpoint is reachable, without calling the generator's flood fill.
            indices = np.rint((pairs[..., ::-1] + 4.85) / step).astype(int)
            reached = np.zeros_like(free)
            reached[tuple(indices[0, 0])] = True
            while True:
                expanded = reached.copy()
                expanded[1:] |= reached[:-1]
                expanded[:-1] |= reached[1:]
                expanded[:, 1:] |= reached[:, :-1]
                expanded[:, :-1] |= reached[:, 1:]
                expanded &= free
                if np.array_equal(expanded, reached):
                    break
                reached = expanded
            self.assertTrue(reached[indices[..., 0], indices[..., 1]].all())

    def test_graphs_include_serpentine_loops_and_open_runs(self):
        serpentine = loop = straight = detour = False
        perimeter = {(0, 1), (1, 2), (2, 5), (5, 8), (7, 8), (6, 7), (3, 6), (0, 3)}
        for layout in self.layouts:
            edges = {tuple(e) for e in layout["open_edges"]}
            # A perimeter cycle around at least two central wall sections is
            # a ring corridor, rather than merely an empty room.
            central_walls = sum(4 in edge for edge, active in zip(
                geometry.wall_graph(10, .15, 3)[0], layout["wall_active"]) if active)
            loop |= perimeter <= edges and central_walls >= 2
            straight |= any({(r * 3, r * 3 + 1), (r * 3 + 1, r * 3 + 2)} <= edges for r in range(3))
            pairs = layout["spawn_pairs"]
            lines = pairs[:, :1] + (pairs[:, 1:] - pairs[:, :1]) * np.linspace(0, 1, 65)[None, :, None]
            args = (layout["wall_poses"], layout["wall_active"], layout["wall_lengths"], .15, 4.85)
            wall_clear = geometry.point_clearance(lines, layout["positions"], RADII,
                                                   np.zeros(15, dtype=bool), *args).min(axis=1)
            all_clear = geometry.point_clearance(lines, layout["positions"], RADII,
                                                  layout["active"], *args).min(axis=1)
            detour |= bool(((wall_clear >= layout["navigation_clearance"])
                            & (all_clear < layout["navigation_clearance"])).any())
            neighbours = [[] for _ in range(9)]
            for a, b in edges:
                neighbours[a].append(b)
                neighbours[b].append(a)
            for start in range(9):
                paths = [[start]]
                seen = {start}
                for path in paths:
                    for nxt in neighbours[path[-1]]:
                        if nxt in seen:
                            continue
                        seen.add(nxt)
                        extended = path + [nxt]
                        paths.append(extended)
                        coords = np.array([(i % 3, i // 3) for i in extended])
                        directions = np.diff(coords, axis=0)
                        turns = [a[0] * b[1] - a[1] * b[0] for a, b in zip(directions, directions[1:])]
                        serpentine |= 1 in turns and -1 in turns and len(extended) >= 5
        self.assertTrue(serpentine, "missing alternating turns on shortest graph paths")
        self.assertTrue(loop, "missing perimeter ring corridors")
        self.assertTrue(straight, "missing straight connected runs")
        self.assertTrue(detour, "missing clutter obstructing otherwise straight routes")

    def test_narrow_passages_are_part_of_the_random_distribution(self):
        layouts_with_gates = [x for x in self.layouts if len(x["narrow_gate_widths"])]
        self.assertGreater(len(layouts_with_gates), 32)
        forced_pair_count = 0
        for layout in layouts_with_gates:
            self.assertLessEqual(len(layout["narrow_gate_widths"]), 2)
            self.assertTrue(((layout["narrow_gate_widths"] >= 1.20)
                             & (layout["narrow_gate_widths"] <= 1.50)).all())
            # Each gate is represented by exactly two short wall slots.
            self.assertGreaterEqual(layout["wall_active"].sum(),
                                    2 * len(layout["narrow_gate_widths"]))
            self.assertEqual(layout["narrow_pair_mask"].shape, (128,))
            forced_pair_count += layout["narrow_pair_mask"].sum()
        self.assertGreater(forced_pair_count, 128 * len(self.layouts) * .15)

    def test_empty_clutter_and_invalid_geometry(self):
        layout = geometry.sample_connected_layout([], seed=4)
        self.assertEqual(layout["positions"].shape, (0, 2))
        for options in ({"grid_size": 0}, {"grid_size": 8}, {"grid_resolution": 0},
                        {"min_goal_distance": 30}, {"loop_probability_range": (-1, 1)},
                        {"narrow_passage_width_range": (1.0, 4.0)}):
            with self.assertRaises(ValueError):
                geometry.sample_connected_layout(RADII, seed=1, **options)


if __name__ == "__main__":
    unittest.main()
