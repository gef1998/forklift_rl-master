"""Raw-cache geometry and real environment queue/reset methods, without Isaac."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
spec = importlib.util.spec_from_file_location('raw_cache_lidar', SOURCE/'async_lidar.py')
lidar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lidar)

@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('sectors', [120,180])
@pytest.mark.parametrize('rays', [720,1440])
@pytest.mark.parametrize('precomputed', [False,True])
def test_raw_scan_retains_dense_geometry_after_rotation(device,sectors,rays,precomputed):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    angles=torch.arange(rays,device=device,dtype=torch.float64)[None]*(2*math.pi/rays)
    ranges=2.+.4*torch.sin(angles*7)
    ranges[:,::11]=20.
    pose=torch.tensor([[.1,-.03,.031]],device=device,dtype=torch.float64)
    current=torch.tensor([[.17,.04,.071]],device=device,dtype=torch.float64)
    edges=lidar.hybrid_180_sector_edges(device=device,dtype=torch.float64) if sectors==180 else None
    directions=torch.stack((torch.cos(angles),torch.sin(angles)),dim=-1) if precomputed else None
    result=lidar.warp_cached_raw_ranges_and_clearances(ranges,pose,current,angles,sectors,120,20.,.425,.275,0.,0.,edges,ray_directions=directions)
    # Oracle: all original physical endpoints, transformed in world coordinates.
    points=lidar.transform_points_se2(lidar.ranges_to_points(ranges,angles),pose,current)
    expected=lidar.points_to_ranges_and_clearances(points,sectors,20.,.425,.275,0.,0.,ranges<20.,edges)
    shield=lidar.points_to_ranges_and_clearances(points,120,20.,.425,.275,0.,0.,ranges<20.)[1]
    for a,b in zip(result,(*expected,shield)):torch.testing.assert_close(a,b,atol=1e-10,rtol=0.)

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_raw_compensation_avoids_false_free_bin_and_policy_dropout_is_independent(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    angles=torch.tensor([[.1,.9]],device=device)*math.pi/180
    ranges=torch.tensor([[1.,1.1]],device=device)
    source=torch.zeros((1,3),device=device);target=source.clone();target[:,2]=.2*math.pi/180
    edges=lidar.hybrid_180_sector_edges(device=device)
    raw=lidar.warp_cached_raw_ranges_and_clearances(ranges,source,target,angles,180,120,20.,.425,.275,0.,0.,edges)
    # The closer .1-degree ray moves right of the nose; the .9-degree ray
    # still occupies the first left-front sector and must not disappear.
    assert raw[0][0,0].item()==pytest.approx(1.1,abs=1e-6)
    assert raw[1][0,0].item()<1.
    dropped=lidar.warp_cached_raw_ranges_and_clearances(ranges,source,target,angles,180,120,20.,.425,.275,0.,0.,edges,torch.zeros_like(ranges,dtype=torch.bool))
    assert dropped[1].min().item()>19.
    torch.testing.assert_close(dropped[2],raw[2])

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_large_world_origin_does_not_perturb_stationary_raw_scan(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    angles=torch.arange(720,device=device)[None]*(math.pi/360)
    ranges=torch.full_like(angles,2.)
    edges=lidar.hybrid_180_sector_edges(device=device)
    zero=torch.zeros((1,3),device=device)
    far=torch.tensor([[5000.,-5000.,.7]],device=device)
    a=lidar.warp_cached_raw_ranges_and_clearances(ranges,zero,zero,angles,180,120,20.,.425,.275,0.,0.,edges)
    b=lidar.warp_cached_raw_ranges_and_clearances(ranges,far,far,angles,180,120,20.,.425,.275,0.,0.,edges)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-6,rtol=0.)
    assert (b[0]<20.).all()

def environment_methods():
    tree=ast.parse((SOURCE/'emma_obs_nav_env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EmmaObsNavEnv')
    names={'_enqueue_lidar_scans','_deliver_pending_lidar','_configure_lidar_schedule'}
    funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace={'torch':torch,'math':math,'sample_sensor_periods':lidar.sample_sensor_periods}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SOURCE/'emma_obs_nav_env.py'),'exec'),namespace)
    return type('QueueHarness',(),{name:namespace[name] for name in names})

def queue_fixture():
    obj=environment_methods()();obj.device='cpu';obj.num_envs=2;obj.step_dt=.02
    obj.cfg=SimpleNamespace(lidar_warp_before_pooling=True,lidar_sensor_mode='fixed',lidar_fixed_frequency_hz=10.,lidar_max_distance=20.)
    obj._cached_scan_ranges=torch.full((2,720),20.)
    obj._cached_policy_dropout=torch.zeros((2,180),dtype=torch.bool)
    obj._cached_sensor_pose=torch.zeros((2,3));obj._cached_timestamp=torch.zeros(2)
    obj._has_cached_scan=torch.zeros(2,dtype=torch.bool)
    obj._pending_scan_ranges=torch.full((2,4,720),20.)
    obj._pending_policy_dropout=torch.zeros((2,4,180),dtype=torch.bool)
    obj._pending_sensor_pose=torch.zeros((2,4,3))
    obj._pending_generation_timestamp=torch.zeros((2,4));obj._pending_delivery_timestamp=torch.full((2,4),float('inf'))
    obj._pending_valid=torch.zeros((2,4),dtype=torch.bool);obj._pending_write_index=torch.zeros(2,dtype=torch.long)
    obj._episode_noise_scale=torch.ones(2);obj._sample_lidar_latency=lambda n:torch.zeros(n)
    for name in ['_sensor_period','_sensor_latency','_next_sensor_timestamp']:setattr(obj,name,torch.zeros(2))
    obj._sensor_schedule_initialized=torch.zeros(2,dtype=torch.bool)
    return obj

def test_raw_queue_delivers_newest_eligible_scan_and_its_capture_dropout():
    obj=queue_fixture();raw=torch.ones((2,720));drop=torch.zeros((2,180),dtype=torch.bool);drop[0,4]=True
    pose=torch.zeros((2,3));ids=torch.tensor([0])
    obj._enqueue_lidar_scans(ids,raw,None,None,None,pose,torch.tensor([.1,.1]),torch.tensor([.2,.2]),drop)
    obj._enqueue_lidar_scans(ids,raw*2,None,None,None,pose,torch.tensor([.15,.15]),torch.tensor([.18,.18]),~drop)
    delivered=obj._deliver_pending_lidar(torch.tensor([.21,.21]))
    assert delivered.tolist()==[True,False]
    assert obj._cached_scan_ranges[0].eq(2.).all()
    assert obj._cached_timestamp[0].item()==pytest.approx(.15)
    assert torch.equal(obj._cached_policy_dropout[0],(~drop)[0])
    assert not obj._pending_valid.any()

def test_episode_reset_clears_raw_cache_queue_and_dropout_for_only_that_env():
    obj=queue_fixture();obj._has_cached_scan[:]=True;obj._cached_scan_ranges[:]=1.
    obj._cached_policy_dropout[:]=True;obj._pending_valid[:]=True
    obj._configure_lidar_schedule(torch.tensor([0]),torch.tensor([3.,3.]))
    assert obj._cached_scan_ranges[0].eq(20.).all() and obj._cached_scan_ranges[1].eq(1.).all()
    assert not obj._cached_policy_dropout[0].any() and obj._cached_policy_dropout[1].all()
    assert not obj._pending_valid[0].any() and obj._pending_valid[1].all()
    assert obj._has_cached_scan.tolist()==[False,True]
