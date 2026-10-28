"""Velocity reward semantics: moving direction, stops, local path and dt."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch

root=Path(__file__).resolve().parents[1]/'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
if not root.exists():root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reference_path',root/'reference_path.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class VelocityDirectionTests(unittest.TestCase):
    def cost(self, velocities, tangents=None, valid=None, dt=.02, scale=.15):
        velocities=torch.tensor(velocities,dtype=torch.float32)
        if tangents is None:tangents=[[1.,0.]]*len(velocities)
        if valid is None:valid=[True]*len(velocities)
        return r.reference_velocity_direction_cost(velocities,torch.tensor(tangents,dtype=torch.float32),
            torch.tensor(valid),step_dt=dt,cost_per_second=scale,min_speed=.05,full_speed=.30,tolerance_deg=15.)

    def test_parallel_forward_and_backward_use_signed_progress_not_duplicate_cost(self):
        cost,cosine,weight=self.cost([[.3,0],[-.3,0],[0,.3]])
        torch.testing.assert_close(cost,torch.tensor([0.,0.,-.003]))
        torch.testing.assert_close(cosine,torch.tensor([1.,-1.,0.]))
        # Existing progress contributes +/- 10*v*dt; combined motion preference
        # remains forward > sideways > backward, without a stationary bonus.
        total=10*.02*torch.tensor([.3,-.3,0.])+cost
        self.assertGreater(total[0].item(),total[2].item())
        self.assertGreater(total[2].item(),total[1].item())

    def test_stops_and_invalid_references_are_zero_and_finite(self):
        cost,_,weight=self.cost([[0,0],[.02,0],[float('nan'),0],[0,.3],[0,.3]],
            tangents=[[1,0],[1,0],[1,0],[0,0],[1,0]],valid=[True,True,True,True,False])
        torch.testing.assert_close(cost,torch.zeros(5))
        torch.testing.assert_close(weight,torch.zeros(5))

    def test_speed_fade_angle_deadband_disable_and_time_scaling(self):
        cost,_,weight=self.cost([[0,.175],[.3*math.cos(math.radians(10)),.3*math.sin(math.radians(10))]])
        self.assertAlmostEqual(weight[0].item(),.5,places=6)
        self.assertAlmostEqual(cost[0].item(),-.0015,places=6)
        self.assertAlmostEqual(cost[1].item(),0.,places=7)
        fine=self.cost([[0,.3]],dt=.01)[0]
        coarse=self.cost([[0,.3]],dt=.02)[0]
        torch.testing.assert_close(2*fine,coarse)
        torch.testing.assert_close(self.cost([[0,.3]],scale=0)[0],torch.zeros(1))

    def test_reverse_travel_and_corner_tangent_follow_path_order(self):
        points=torch.tensor([[1.,.5],[.5,0.]])
        paths=torch.tensor([[[0.,0],[1.,0],[1.,1]],[[0.,0],[1.,0],[1.,1]]])
        _,_,tangent=r.project_points_to_polylines(points,paths,torch.tensor([3,3]),return_tangent=True)
        old=r.project_points_to_polylines(points,paths,torch.tensor([3,3]))
        new=r.project_points_to_polylines(points,paths,torch.tensor([3,3]),return_tangent=True)
        torch.testing.assert_close(old[0],new[0]);torch.testing.assert_close(old[1],new[1])
        torch.testing.assert_close(tangent,torch.tensor([[0.,1],[1.,0]]))
        # Legacy path reverse traversal flips the travel tangent, not vehicle yaw.
        cost,cosine,_=self.cost([[0,.3],[-.3,0]],tangents=(tangent*torch.tensor([[1.],[-1.]])).tolist())
        torch.testing.assert_close(cost,torch.zeros(2));torch.testing.assert_close(cosine,torch.ones(2))

    def test_continuous_projection_keeps_same_branch_and_teleport_is_inactive(self):
        def query(point,previous):
            return r.project_continuous_progress(torch.tensor([point]),torch.tensor([previous]),
                torch.tensor([[[0.,0],[0,2],[.1,2],[.1,0]]]),torch.tensor([4]),torch.tensor([.79]),
                torch.tensor([[[0.,10.,2.,10.]]]),torch.tensor([1]),torch.tensor([0]),
                torch.tensor([math.pi/2]),torch.ones(1),half_length=.425,half_width=.275,
                collision_margin=.01,return_tangent=True)
        _,_,_,tangent=query([.09,.8],[.09,.79])
        torch.testing.assert_close(tangent,torch.tensor([[0.,1.]]))
        _,_,_,teleport=query([.09,1.8],[.09,.79])
        torch.testing.assert_close(teleport,torch.zeros(1,2))


if __name__=='__main__':unittest.main()
