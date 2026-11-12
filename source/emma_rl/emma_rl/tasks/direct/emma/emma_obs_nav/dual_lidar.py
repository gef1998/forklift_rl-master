"""Mounted front/rear observation scans, independent of collision truth.

Mounts and native scan metadata come from 2026-11-09-22-54-49.bag TF/LaserScan.
No Isaac imports: geometry, delivery and reset can be tested without a simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .async_lidar import (
    _sector_indices, advance_sensor_schedule, sample_sensor_periods,
    sensor_due_mask, warp_cached_raw_ranges_and_clearances,
)


@dataclass
class MountedLidarCfg:
    position: tuple[float, float, float]
    yaw_deg: float
    angle_min_deg: float
    angle_increment_deg: float
    native_rays: int
    sample_stride: int

    @property
    def ray_count(self):
        if self.sample_stride < 1 or self.native_rays < 1:
            raise ValueError("mounted LiDAR ray count and stride must be positive")
        return (self.native_rays + self.sample_stride - 1) // self.sample_stride


def front_mount():
    return MountedLidarCfg((.364452970190296, -.0006679106426595418, 0.),
        -.481695039143, -104.99999950679661, .2499958092364564, 840, 2)


def back_mount():
    return MountedLidarCfg((-.380840938976, .00387318860567, .2),
        179.498420146, -100.0000005060238, .2499999980634083, 800, 4)


def mounted_rays(cfg, device, dtype=torch.float32):
    """Preserve native beam indices/angles when subsampling, as deployment does."""
    indices = torch.arange(0, cfg.native_rays, cfg.sample_stride, device=device, dtype=dtype)
    angles = (indices * math.radians(cfg.angle_increment_deg)
              + math.radians(cfg.angle_min_deg) + math.radians(cfg.yaw_deg))
    directions = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    starts = torch.tensor(cfg.position, device=device, dtype=dtype).expand(cfg.ray_count, -1).clone()
    return starts, directions


class DualLidarObservation:
    """Two capture clocks/queues; one virtual body-frame observation contract."""
    def __init__(self, num_envs, cfg, device, policy_edges):
        self.n, self.cfg, self.device, self.edges = num_envs, cfg, device, policy_edges
        self.mounts = (cfg.front_lidar, cfg.back_lidar)
        self.geometry = [mounted_rays(m, device) for m in self.mounts]
        counts = [m.ray_count for m in self.mounts]
        self.slices = [slice(0, counts[0]), slice(counts[0], sum(counts))]
        self.ray_count = sum(counts)
        self.ranges = torch.full((num_envs, self.ray_count), cfg.lidar_max_distance, device=device)
        self.directions = torch.zeros((num_envs, self.ray_count, 2), device=device)
        self.policy_keep = torch.ones((num_envs, self.ray_count), dtype=torch.bool, device=device)
        self.poses = torch.zeros((num_envs, 2, 3), device=device)
        self.source_stamp = torch.zeros((num_envs, 2), device=device)
        self.source_has = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.timestamp = torch.zeros(num_envs, device=device)
        self.has_cache = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.period = torch.full((num_envs, 2), .1, device=device)
        self.latency = torch.zeros((num_envs, 2), device=device)
        self.next_capture = torch.zeros((num_envs, 2), device=device)
        self.generated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.delivered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.aoi = torch.zeros(num_envs, device=device)
        self.ray_queries = 0
        self.queues = []
        q = cfg.lidar_pending_queue_size
        for count in counts:
            self.queues.append(dict(
                ranges=torch.full((num_envs,q,count),cfg.lidar_max_distance,device=device),
                keep=torch.ones((num_envs,q,count),dtype=torch.bool,device=device),
                pose=torch.zeros((num_envs,q,3),device=device),
                stamp=torch.zeros((num_envs,q),device=device),
                delivery=torch.full((num_envs,q),float('inf'),device=device),
                valid=torch.zeros((num_envs,q),dtype=torch.bool,device=device),
                write=torch.zeros(num_envs,dtype=torch.long,device=device)))

    def reset(self, ids, now, noise):
        cfg = self.cfg; n = len(ids); mode = cfg.lidar_sensor_mode.lower()
        if mode == 'synchronous':
            period = torch.full((n,2), cfg.sim.dt*cfg.decimation, device=self.device)
            latency = torch.zeros_like(period); phase = torch.zeros_like(period)
        else:
            nominal = 1./cfg.lidar_fixed_frequency_hz
            if mode == 'fixed':
                period = torch.full((n,2),nominal,device=self.device)
            elif mode == 'random':
                sampled = sample_sensor_periods(n*2,cfg.lidar_frequency_choices_hz,self.device).reshape(n,2)
                period = nominal + noise[ids,None]*(sampled-nominal)
            else:
                raise ValueError('unknown LiDAR sensor mode: '+mode)
            phase = torch.rand_like(period)*period
            low,high = cfg.lidar_latency_range_s
            latency = (low+torch.rand_like(period)*(high-low))*noise[ids,None]
        self.period[ids]=period;self.latency[ids]=latency
        self.next_capture[ids]=now[ids,None]+phase
        self.ranges[ids]=cfg.lidar_max_distance;self.policy_keep[ids]=True
        self.directions[ids]=0.;self.poses[ids]=0.
        self.source_stamp[ids]=now[ids,None];self.timestamp[ids]=now[ids]
        self.source_has[ids]=False;self.has_cache[ids]=False;self.initialized[ids]=True
        for q in self.queues:
            q['valid'][ids]=False;q['delivery'][ids]=float('inf');q['write'][ids]=0

    def _virtual_points(self, source, ranges):
        starts, dirs = self.geometry[source]
        offset = ranges.new_tensor([self.cfg.lidar_offset_x,self.cfg.lidar_offset_y])
        return starts[None,:,:2] + ranges[:,:,None]*dirs[None,:,:2] - offset

    def _prepare(self, source, clean, noise):
        cfg=self.cfg;physical=(clean>=.05)&(clean<cfg.lidar_max_distance)&torch.isfinite(clean)
        clean=torch.where(physical,clean,torch.full_like(clean,cfg.lidar_max_distance))
        ranges=clean
        if cfg.domain_rand.enable_lidar_noise:
            ranges=torch.where(physical,clean+torch.randn_like(clean)*cfg.domain_rand.lidar_noise_std*noise[:,None],clean)
        ranges=ranges.clamp(0.,cfg.lidar_max_distance)
        points=self._virtual_points(source,ranges)
        norm=torch.linalg.vector_norm(points,dim=-1)
        body=points+ranges.new_tensor([cfg.lidar_offset_x,cfg.lidar_offset_y])
        inside=(body[...,0].abs()<cfg.robot_half_length)&(body[...,1].abs()<cfg.robot_half_width)
        valid=physical&(ranges>=.05)&(ranges<cfg.lidar_max_distance)&(norm>1e-6)&(norm<cfg.lidar_max_distance)&~inside
        ranges=torch.where(valid,ranges,torch.full_like(ranges,cfg.lidar_max_distance))
        keep=torch.ones_like(valid)
        if cfg.domain_rand.enable_lidar_dropout:
            dropout=torch.rand((len(ranges),cfg.lidar_output_sectors),device=self.device)<cfg.domain_rand.lidar_dropout_prob*noise[:,None]
            angles=torch.atan2(points[...,1],points[...,0])
            index=_sector_indices(angles,cfg.lidar_output_sectors,self.edges)
            keep=~dropout.gather(1,index)
        return ranges,keep

    def step(self, current_pose, now, noise, capture):
        cfg=self.cfg
        uninit=~self.initialized
        if uninit.any():self.reset(torch.where(uninit)[0],now,noise)
        due=sensor_due_mask(now[:,None],self.next_capture)
        force=~self.source_has
        generate=due|force
        self.generated.copy_(generate.any(1));self.delivered.zero_()
        accepted=generate.clone()
        if cfg.lidar_sensor_mode.lower()!='synchronous' and cfg.lidar_scan_dropout_prob>0:
            accepted &= torch.rand_like(self.latency)>=cfg.lidar_scan_dropout_prob*noise[:,None]
        accepted |= force
        ids=[torch.where(accepted[:,source])[0] for source in range(2)]
        if any(len(x) for x in ids):
            # One shared mesh query can concatenate only these due rays.
            scans=capture(ids)
            for source,envs in enumerate(ids):
                if not len(envs):continue
                self.ray_queries+=len(envs)*self.mounts[source].ray_count
                ranges,keep=self._prepare(source,scans[source],noise[envs])
                q=self.queues[source];slots=q['write'][envs]%q['valid'].shape[1]
                q['ranges'][envs,slots]=ranges;q['keep'][envs,slots]=keep
                q['pose'][envs,slots]=current_pose[envs];q['stamp'][envs,slots]=now[envs]
                delivery=now[envs]+self.latency[envs,source]
                q['delivery'][envs,slots]=torch.where(force[envs,source],now[envs],delivery)
                q['valid'][envs,slots]=True;q['write'][envs]+=1
        jitter=torch.zeros_like(self.latency)
        if cfg.lidar_sensor_mode.lower()!='synchronous' and cfg.lidar_jitter_max_steps>0:
            jitter=torch.randint(-cfg.lidar_jitter_max_steps,cfg.lidar_jitter_max_steps+1,self.latency.shape,device=self.device)*noise[:,None]
        self.next_capture.copy_(advance_sensor_schedule(now[:,None].expand_as(self.next_capture),self.next_capture,self.period,due,cfg.sim.dt*cfg.decimation,jitter))
        for source,q in enumerate(self.queues):
            eligible=q['valid']&(q['delivery']<=now[:,None]+1e-7)
            mask=eligible.any(1)
            if not mask.any():continue
            stamp,slots=torch.where(eligible,q['stamp'],float('-inf')).max(1)
            envs=torch.where(mask)[0];slot=slots[envs];sl=self.slices[source]
            physical=q['ranges'][envs,slot];points=self._virtual_points(source,physical)
            norm=torch.linalg.vector_norm(points,dim=-1)
            valid=(physical<cfg.lidar_max_distance)&(norm>1e-6)&(norm<cfg.lidar_max_distance)
            self.ranges[envs,sl]=torch.where(valid,norm,cfg.lidar_max_distance)
            self.directions[envs,sl]=points/norm.clamp_min(1e-6)[:,:,None]
            self.policy_keep[envs,sl]=q['keep'][envs,slot]
            self.poses[envs,source]=q['pose'][envs,slot]
            self.source_stamp[envs,source]=stamp[envs];self.source_has[envs,source]=True
            self.delivered |= mask
            q['valid'][eligible]=False;q['delivery'][eligible]=float('inf')
        self.has_cache.copy_(self.source_has.all(1));self.timestamp.copy_(self.source_stamp.amin(1))
        self.aoi.copy_((now-self.timestamp).clamp_min(0.))
        results=[]
        for source,sl in enumerate(self.slices):
            pose=self.poses[:,source]
            target=current_pose if cfg.lidar_warp_enabled else pose
            results.append(warp_cached_raw_ranges_and_clearances(
                self.ranges[:,sl],pose,target,None,cfg.lidar_output_sectors,cfg.lidar_safety_output_sectors,
                cfg.lidar_max_distance,cfg.robot_half_length,cfg.robot_half_width,cfg.lidar_offset_x,cfg.lidar_offset_y,
                self.edges,self.policy_keep[:,sl],self.directions[:,sl]))
        merged=[torch.minimum(a,b) for a,b in zip(*results)]
        return merged[0],self.aoi,merged[1],merged[2]
