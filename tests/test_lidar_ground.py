"""Regression coverage for non-square clone counts and translated layouts."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import torch

root=Path(__file__).resolve().parents[1]/'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
if not root.exists():root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('lidar_ground',root/'lidar_ground.py')
ground_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(ground_module)


def clone_origins(count,spacing=10.):
    # Actual GridCloner layout: floor(sqrt(N)) followed by ceil rows/columns.
    per_row=int(math.sqrt(count));rows=math.ceil(count/per_row);cols=math.ceil(count/rows)
    ids=np.arange(count)
    return np.c_[(rows-1)*spacing/2-(ids//cols)*spacing,
                 (ids%cols)*spacing-(cols-1)*spacing/2]


class LidarGroundTests(unittest.TestCase):
    def assert_covered(self,origins,size,expected_extent):
        ground=ground_module.make_lidar_ground(origins,size)
        xy=np.asarray(origins)[:,:2]
        self.assertTrue((ground.bounds[0,:2]<=xy-size/2+1e-8).all())
        self.assertTrue((ground.bounds[1,:2]>=xy+size/2-1e-8).all())
        np.testing.assert_allclose(ground.extents,expected_extent,atol=1e-8)
        np.testing.assert_allclose(ground.bounds[:,2],[-.02,0.],atol=1e-10)
        self.assertTrue(ground.is_watertight)
        return ground

    def test_single_and_square_training_layout(self):
        self.assert_covered(clone_origins(1),10.,[10.,10.,.02])
        self.assert_covered(clone_origins(4096),10.,[640.,640.,.02])

    def test_12800_environments_cover_outermost_room_edges(self):
        origins=clone_origins(12800)
        ground=self.assert_covered(origins,10.,[1140.,1130.,.02])
        np.testing.assert_allclose(ground.bounds[:,:2],[[-570.,-565.],[570.,565.]])
        # The former 1130 x 1130 ground misses the last 5 m on both X edges.
        self.assertGreater((origins[:,0]+5).max(),1130/2)
        self.assertLess((origins[:,0]-5).min(),-1130/2)

    def test_offset_and_irregular_layouts_and_warehouse_size(self):
        origins=np.array([[100.,20.,0.],[130.,28.,0.],[104.,-5.,0.]])
        ground=self.assert_covered(origins,10.,[40.,43.,.02])
        np.testing.assert_allclose(ground.bounds[:,:2],[[95.,-10.],[135.,33.]])
        translated=clone_origins(12800,16.)+[123.,-87.]
        ground=self.assert_covered(translated,16.,[1824.,1808.,.02])
        np.testing.assert_allclose(ground.centroid[:2],[123.,-87.],atol=1e-8)

    def test_tensor_inputs_are_detached_and_bad_bounds_fail_explicitly(self):
        origins=torch.tensor([[100.,-20.]],requires_grad=True)
        ground=ground_module.make_lidar_ground(origins,10.)
        np.testing.assert_allclose(ground.bounds[:,:2],[[95.,-25.],[105.,-15.]])
        for bad in (np.empty((0,2)),np.zeros((2,1)),[[float('nan'),0.]]):
            with self.assertRaises(ValueError):ground_module.make_lidar_ground(bad,10.)
        for bad in (0.,-1.,float('inf')):
            with self.assertRaises(ValueError):ground_module.make_lidar_ground([[0.,0.]],bad)


if __name__=='__main__':unittest.main()
