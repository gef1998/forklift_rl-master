"""Offline tests for stage-one SE2 references. No simulator, GPU or ROS needed."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
if not ROOT.exists():
    ROOT = Path(__file__).parent  # standalone offline staging
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

p = load('se2_reference_planner')
geometry = load('straight_detour_geometry')
reward = load('reference_path')


def layout(seed=0, narrow=False):
    full = (10-.3)/3
    ranges = ((.6,.6),)*6 if narrow else ((1.8,2.3),(1.4,2.1),(1.25,1.8),(1.1,1.7),(1.1,1.6),(.8,1.5))
    data = geometry.sample_straight_detour_layout([.25,.225*2**.5]*6,
        [full]*12+[(full-1.5)/2]*4, seed=seed, passage_width_range=(min(a for a,b in ranges),max(b for a,b in ranges)),
        curriculum_passage_width_ranges=ranges, barrier_y_span=2.2, spawn_y=3.95)
    config = SimpleNamespace(_obstacle_types=['cylinder','cube']*6,_obstacle_radii=[.25,0]*6,
        _obstacle_cube_sizes=[(0,0),(.45,.45)]*6,interior_wall_thickness=.15,arena_size=10.,wall_thickness=.15)
    return data, config


class PlannerTests(unittest.TestCase):
    def test_env4071_seed42_side_wall_sweep_budget_regression(self):
        path=Path(__file__).parent/'data/timeout_env4071_seed42.json'
        if not path.exists():path=Path(__file__).parent/'timeout_env4071_seed42.json'
        data=json.loads(path.read_text())
        world=p.CollisionWorld(data['boxes'],data['circles'],data['inner_half'])
        cfg=p.PlannerConfig(xy_resolution=.01,yaw_resolution_deg=1.,max_seconds=.5)
        result=p.DetourReferenceBank(cfg).plan(world,np.array(data['start']),
                                             np.array(data['goal']),np.array(data['gates']))
        self.assertTrue(result.success,result.reason)
        self.assertEqual(result.expansions,0)
        self.assertFalse(result.requires_stop)
        self.assertGreaterEqual(result.minimum_moving_radius,.5-1e-7)
        self.assertTrue(world.valid(result.poses,cfg).all())

    def test_refined_sweep_still_rejects_real_collision_and_supports_larger_radius(self):
        # A motion along the boundary is truly in collision, not a padding-only rejection.
        cfg=p.PlannerConfig()
        planner=p.HybridAStar(p.CollisionWorld(inner_half=1.),cfg)
        self.assertIsNone(planner.checked_motion(np.array([0.,.72,0.]),.1,0.))
        with self.assertRaises(ValueError):p.PlannerConfig(analytic_radius_factors=(1.,.9))
        with self.assertRaises(ValueError):p.PlannerConfig(analytic_radius_factors=(1.,float('inf')))
        self.assertEqual(p.PlannerConfig(analytic_radius_factors=[1.,1.5]).analytic_radius_factors,(1.,1.5))
        larger=p.HybridAStar(p.CollisionWorld([[0,0,0,.1,.8]],inner_half=4),cfg).plan([-1.5,0,0],[1.5,0])
        self.assertTrue(larger.success,larger.reason)
        self.assertEqual(larger.expansions,0)
        self.assertGreater(larger.minimum_moving_radius,.5+1e-6)
    def test_tangent_candidates_reach_endpoint_with_bounded_radius(self):
        rng=np.random.default_rng(14)
        for radius in (.35,.50,.70):
            for _ in range(20):
                start=rng.uniform(-2,2,3)
                goal=rng.uniform(-2,2,2)
                yaw=float(rng.uniform(-math.pi,math.pi))
                candidates=p.tangent_connections(start,goal,yaw,radius)
                self.assertGreaterEqual(len(candidates),4)
                for commands in candidates:
                    pose=start.copy()
                    for length,turn in commands:
                        if abs(turn)>1e-9:self.assertGreaterEqual(abs(length/turn),radius-1e-9)
                        pose=p.motion(pose,length,turn,.1,.1)[-1]
                    np.testing.assert_allclose(pose[:2],goal,atol=1e-7)
                    self.assertAlmostEqual(p.wrap(pose[2]-yaw),0.,places=7)

    def test_radius_applies_to_motion_and_analytic_connectors(self):
        cfg=p.PlannerConfig(min_turning_radius=.5,inplace_rotation_fallback=False)
        planner=p.HybridAStar(p.CollisionWorld(),cfg)
        self.assertIsNone(planner.checked_motion(np.zeros(3),.1,.3))
        self.assertIsNone(planner.checked_motion(np.zeros(3),.1,.3,allow_inplace=True))
        plan=planner.plan([0,0,0],[2,1],0.)
        self.assertTrue(plan.success,plan.reason)
        self.assertFalse(plan.requires_stop)
        self.assertFalse(plan.fallback_attempted)
        self.assertGreaterEqual(plan.minimum_moving_radius,.5-1e-7)
        self.assertTrue((np.linalg.norm(np.diff(plan.poses[:,:2],axis=0),axis=1)>1e-8).all())
        delta=np.diff(plan.poses[:,:2],axis=0)
        travel_yaw=np.arctan2(delta[:,1],delta[:,0])
        mean_yaw=plan.poses[:-1,2]+p.wrap(np.diff(plan.poses[:,2]))/2
        expected=mean_yaw+np.where(plan.directions[1:]<0,math.pi,0.)
        np.testing.assert_allclose(p.wrap(travel_yaw-expected),0.,atol=1e-7)

    def test_rotation_is_only_a_marked_second_pass_fallback(self):
        w=p.CollisionWorld(inner_half=.60)
        cfg=p.PlannerConfig(min_turning_radius=.50,max_expansions=1,max_seconds=.5)
        smooth=p.HybridAStar(w,p.PlannerConfig(min_turning_radius=.50,max_expansions=1,
                                             max_seconds=.5,inplace_rotation_fallback=False)).plan([0,0,0],[0,0],math.pi/2)
        self.assertFalse(smooth.success)
        plan=p.HybridAStar(w,cfg).plan([0,0,0],[0,0],math.pi/2)
        self.assertTrue(plan.success,plan.reason)
        self.assertTrue(plan.fallback_attempted)
        self.assertTrue(plan.requires_stop)
        self.assertEqual(plan.rotation_count,1)
        self.assertEqual(plan.reason,'rotation_fallback')
        self.assertTrue(w.valid(plan.poses,cfg).all())

    def test_rectangular_footprint_rotation_and_mixed_geometry(self):
        # 0.60 m corridor: aligned robot fits, rotated robot does not.
        w = p.CollisionWorld([[0,.375,0,3,.075],[0,-.375,0,3,.075]], inner_half=4)
        cfg = p.PlannerConfig()
        np.testing.assert_array_equal(w.valid([[0,0,0],[0,0,math.pi/2]],cfg),[True,False])
        self.assertIsNone(p.HybridAStar(w).checked_motion(np.zeros(3),0,math.pi/2))
        circle = p.CollisionWorld(circles=[[.50,0,.10]])
        self.assertFalse(circle.valid([[0,0,0]],cfg)[0])
        self.assertTrue(circle.valid([[-1,0,0]],cfg)[0])

    def test_six_tenths_corridor_forward_and_reverse(self):
        w = p.CollisionWorld([[0,.375,0,3,.075],[0,-.375,0,3,.075]], inner_half=4)
        for goal in ([2,0],[-2,0]):
            plan=p.HybridAStar(w).plan([0,0,0],goal)
            self.assertTrue(plan.success,plan.reason)
            self.assertTrue(w.valid(plan.poses,p.PlannerConfig()).all())
            np.testing.assert_allclose(plan.poses[-1,:2],goal,atol=1e-8)
            if goal[0]<0:self.assertTrue((plan.directions<0).any())

    def test_six_tenths_portal_with_s_curve_and_body_tail_clearance(self):
        world=p.CollisionWorld([[-1.8,0,0,2.2,.075],[2.5,0,0,1.5,.075]],inner_half=4.5)
        cfg=p.PlannerConfig(min_turning_radius=.5,inplace_rotation_fallback=False,max_seconds=2)
        for sign in (1,-1):
            start=np.array([-.7,-2*sign,sign*math.pi/2])
            goal=np.array([-.7,2*sign])
            gates=np.array([[.7,0,.3,.6]])
            plan=p.DetourReferenceBank(cfg).plan(world,start,goal,gates)
            self.assertTrue(plan.success,plan.reason)
            self.assertFalse(plan.requires_stop)
            self.assertGreaterEqual(plan.minimum_moving_radius,.5-1e-7)
            self.assertTrue(world.valid(plan.poses,cfg).all())
            self.assertEqual(len(p.gate_arc_lengths(plan.poses,gates)),1)

    def test_swept_motion_cannot_jump_thin_wall(self):
        w=p.CollisionWorld([[0,0,0,.005,1]],inner_half=4)
        self.assertIsNone(p.HybridAStar(w).checked_motion(np.array([-1.,0.,0.]),2.,0.))

    def test_search_detours_wide_wall(self):
        w=p.CollisionWorld([[0,0,0,.1,.8]],inner_half=4)
        cfg=p.PlannerConfig(max_seconds=8,analytic_radius_factors=(1.,))
        plan=p.HybridAStar(w,cfg).plan([-1.5,0,0],[1.5,0])
        self.assertTrue(plan.success,plan.reason)
        self.assertGreater(plan.expansions,0)
        self.assertGreater(abs(plan.poses[:,1]).max(),.8)
        self.assertTrue(w.valid(plan.poses,cfg).all())

    def test_failure_never_returns_partial_or_direct_path(self):
        w=p.CollisionWorld([[0,0,0,.1,4]],inner_half=4)
        cfg=p.PlannerConfig(max_expansions=1,max_seconds=2)
        plan=p.HybridAStar(w,cfg).plan([-1.5,0,0],[1.5,0])
        self.assertFalse(plan.success)
        self.assertEqual(len(plan.poses),0)
        plan=p.HybridAStar(w,cfg).plan([0,0,0],[1.5,0])
        self.assertEqual(plan.reason,'start_collision')

    def test_detour_cache_preserves_actual_start_heading_and_geometry(self):
        data,config=layout()
        w=p.world_from_detour_layout(data,config)
        start,goal=data['curriculum_pairs'][4,0]
        yaw=math.copysign(math.pi/2,goal[1]-start[1])
        start=np.r_[start,yaw]
        gates=p.ordered_detour_gates(data,4,start,goal)
        bank=p.DetourReferenceBank(p.PlannerConfig(max_seconds=3))
        first=bank.plan(w,start,goal,gates)
        self.assertTrue(first.success,first.reason)
        self.assertTrue(np.all(np.diff(p.gate_arc_lengths(first.poses,gates))>0))
        altered=start.copy();altered[0]+=.03;altered[2]+=.07
        second=bank.plan(w,altered,goal,gates)
        self.assertTrue(second.success,second.reason)
        np.testing.assert_allclose(second.poses[0],altered)
        self.assertGreater(bank.stats['cache_hits'],0)
        closed=p.CollisionWorld(np.r_[w.boxes,[[*gates[0,:2],0,.2,.2]]],w.circles,w.inner_half)
        self.assertNotEqual(w.fingerprint,closed.fingerprint)
        blocked=bank.plan(closed,start,goal,gates)
        self.assertFalse(blocked.success)
        self.assertEqual(len(blocked.poses),0)

    def test_budget_and_configuration_validation(self):
        with self.assertRaises(ValueError):p.PlannerConfig(min_turning_radius=0)
        with self.assertRaises(ValueError):p.PlannerConfig(min_turning_radius=float('nan'))
        with self.assertRaises(ValueError):p.PlannerConfig(max_points=1)
        with self.assertRaises(ValueError):p.PlannerConfig(yaw_resolution_deg=1000)
        bank=p.DetourReferenceBank(p.PlannerConfig(max_points=2))
        plan=bank.plan(p.CollisionWorld(),[0,0,0],[2,0],[])
        self.assertFalse(plan.success)
        self.assertEqual(plan.reason,'point_budget')


class ProgressTests(unittest.TestCase):
    def project(self, xy, prev_xy, prev_s, path=None, gates=None, passed=0, yaw=math.pi/2):
        if path is None:path=[[0,0],[0,1],[0,2]]
        if gates is None:gates=[[0,1,.3,1.]]
        return reward.project_continuous_progress(
            torch.tensor([xy],dtype=torch.float32),torch.tensor([prev_xy],dtype=torch.float32),
            torch.tensor([path],dtype=torch.float32),torch.tensor([len(path)]),torch.tensor([prev_s],dtype=torch.float32),
            torch.tensor([gates],dtype=torch.float32),torch.tensor([len(gates)]),torch.tensor([passed]),
            torch.tensor([yaw]),torch.ones(1),half_length=.425,half_width=.275,collision_margin=.01)

    def test_stationary_rotation_has_no_progress(self):
        s,_,passed=self.project([0,.95],[0,.95],.93,yaw=0.)
        self.assertAlmostEqual(s.item(),.93,places=6)
        self.assertEqual(passed.item(),0)

    def test_valid_gate_crossing_and_signed_backtracking(self):
        s,_,passed=self.project([0,1.01],[0,.99],.99)
        self.assertEqual(passed.item(),1)
        self.assertAlmostEqual(s.item(),1.01,places=5)
        back,_,_=self.project([0,.99],[0,1.01],1.01,passed=1)
        self.assertLess(back.item(),1.01)

    def test_crossing_wall_or_wrong_heading_cannot_unlock_gate(self):
        for xy,prev,yaw in (([.3,1.01],[.3,.99],math.pi/2),([0,1.01],[0,.99],0.)):
            s,_,passed=self.project(xy,prev,.99,yaw=yaw)
            self.assertEqual(passed.item(),0)
            self.assertLessEqual(s.item(),1.)

    def test_teleport_does_not_credit_future_route_or_gate(self):
        s,_,passed=self.project([0,1.9],[0,.9],.9)
        self.assertAlmostEqual(s.item(),.9,places=6)
        self.assertEqual(passed.item(),0)

    def test_close_parallel_segments_do_not_jump(self):
        s,_,_=self.project([.09,.8],[.09,.79],.79,
            path=[[0,0],[0,2],[.1,2],[.1,0]],gates=[[0,10,2,10]])
        self.assertLess(s.item(),.82)


if __name__=='__main__':unittest.main()
