import importlib.util,sys,types,math
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
pkg=types.ModuleType('dual_lidar_test_package');pkg.__path__=[str(SOURCE)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.dual_lidar',SOURCE/'dual_lidar.py')
dual=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dual;spec.loader.exec_module(dual)

def config():
    return NS(front_lidar=dual.front_mount(),back_lidar=dual.back_mount(),lidar_pending_queue_size=4,
        lidar_max_distance=20.,lidar_offset_x=0.,lidar_offset_y=0.,robot_half_length=.425,robot_half_width=.275,
        lidar_output_sectors=180,lidar_safety_output_sectors=120,lidar_sensor_mode='fixed',
        lidar_fixed_frequency_hz=10.,lidar_frequency_choices_hz=(10.,20.),lidar_latency_range_s=(.06,.06),
        lidar_scan_dropout_prob=0.,lidar_jitter_max_steps=0,lidar_warp_enabled=True,sim=NS(dt=.01),decimation=2,
        domain_rand=NS(enable_lidar_noise=False,lidar_noise_std=.03,enable_lidar_dropout=False,lidar_dropout_prob=.02))

def state(device='cpu',n=2):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dual_lidar_test_package.async_lidar import hybrid_180_sector_edges
    cfg=config();return dual.DualLidarObservation(n,cfg,device,hybrid_180_sector_edges(device=device))

def test_tf_mounts_and_native_sampling_indices_match_recorded_geometry():
    a,b=dual.front_mount(),dual.back_mount()
    assert (a.ray_count,b.ray_count)==(420,200)
    starts,dirs=dual.mounted_rays(a,'cpu',torch.float64)
    torch.testing.assert_close(starts[0],torch.tensor(a.position,dtype=torch.float64))
    assert math.degrees(math.atan2(dirs[210,1],dirs[210,0]))==pytest.approx(a.angle_min_deg+420*a.angle_increment_deg+a.yaw_deg,abs=1e-12)
    bs,bd=dual.mounted_rays(b,'cpu',torch.float64)
    assert bs[0,0].item()==pytest.approx(-.380840938976)
    assert bs[0,2].item()==pytest.approx(.2)
    assert math.degrees(math.atan2(bd[100,1],bd[100,0]))==pytest.approx(b.angle_min_deg+400*b.angle_increment_deg+b.yaw_deg,abs=1e-12)
    a.sample_stride=0
    with pytest.raises(ValueError):_ = a.ray_count

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_separate_clocks_deliver_each_source_and_only_query_due_rays(device):
    obj=state(device);pose=torch.zeros((2,3),device=device);noise=torch.ones(2,device=device);calls=[]
    def capture(ids):
        calls.append(tuple(len(x) for x in ids))
        return [torch.ones((len(x),m.ray_count),device=device) for x,m in zip(ids,obj.mounts)]
    obj.step(pose,torch.zeros(2,device=device),noise,capture)
    obj.next_capture[:]=torch.tensor([.1,.16],device=device)
    obj.step(pose,torch.full((2,),.04,device=device),noise,capture)
    assert calls==[(2,2)]
    obj.step(pose,torch.full((2,),.10,device=device),noise,capture)
    assert calls[-1]==(2,0)
    obj.step(pose,torch.full((2,),.16,device=device),noise,capture)
    assert calls[-1]==(0,2)
    torch.testing.assert_close(obj.source_stamp,torch.tensor([[.1,0.],[.1,0.]],device=device))
    assert obj.aoi[0].item()==pytest.approx(.16,abs=1e-6)
    obj.step(pose,torch.full((2,),.22,device=device),noise,capture)
    assert calls[-1]==(2,0)
    assert obj.source_stamp[0,1].item()==pytest.approx(.16,abs=1e-6)
    assert obj.aoi[0].item()==pytest.approx(.12,abs=1e-6)
    assert obj.ray_queries==2*(420+200+420+200+420)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_policy_dropout_does_not_erase_safety_and_reset_is_per_env(device):
    obj=state(device);obj.cfg.domain_rand.enable_lidar_dropout=True;obj.cfg.domain_rand.lidar_dropout_prob=1.
    pose=torch.zeros((2,3),device=device);noise=torch.ones(2,device=device)
    def capture(ids):return [torch.ones((len(x),m.ray_count),device=device) for x,m in zip(ids,obj.mounts)]
    _,_,policy,shield=obj.step(pose,torch.zeros(2,device=device),noise,capture)
    assert policy.min().item()>19.
    assert shield.min().item()<1.
    saved=obj.ranges[1].clone()
    obj.reset(torch.tensor([0],device=device),torch.ones(2,device=device),noise)
    assert obj.ranges[0].eq(20.).all()
    torch.testing.assert_close(obj.ranges[1],saved)
    assert obj.has_cache.tolist()==[False,True]
    assert not obj.source_has[0].any()
    for q in obj.queues:assert not q['valid'][0].any()

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_no_returns_or_nonfinite_ranges_never_become_phantom_obstacles(device):
    obj=state(device);pose=torch.tensor([[.2,.1,.05],[.2,.1,.05]],device=device)
    def capture(ids):return [torch.full((len(x),m.ray_count),float('nan'),device=device) for x,m in zip(ids,obj.mounts)]
    _,_,policy,shield=obj.step(pose,torch.zeros(2,device=device),torch.ones(2,device=device),capture)
    assert torch.isfinite(policy).all() and torch.isfinite(shield).all()
    assert policy.min().item()>19. and shield.min().item()>19.
