#!/usr/bin/env python
"""Read-only ROS1 calibration extraction; no node, control, or parameter writes."""
from __future__ import print_function
import argparse,json,math
import rosbag,rospy,tf
p=argparse.ArgumentParser();p.add_argument('bag');a=p.parse_args()
buffer=tf.Transformer(True,rospy.Duration(120.))
scans={};counts={}
with rosbag.Bag(a.bag) as bag:
 for topic,msg,t in bag.read_messages():
  if topic=='/tf_static':
   for value in msg.transforms:buffer.setTransform(value,'recorded_tf_static')
  if topic in ['/scan_emma_nav_front','/scan_emma_nav_back']:
   meta=(msg.header.frame_id,msg.angle_min,msg.angle_increment,len(msg.ranges),msg.range_min,msg.range_max)
   if topic in scans and scans[topic]!=meta:raise ValueError('scan metadata varies: '+topic)
   scans[topic]=meta;counts[topic]=counts.get(topic,0)+1
result={'source':a.bag,'reference_frame':'base_footprint','mounts':{}}
for topic,stride in [('/scan_emma_nav_front',2),('/scan_emma_nav_back',4)]:
 frame,amin,inc,n,rmin,rmax=scans[topic]
 xyz,quat=buffer.lookupTransform('base_footprint',frame,rospy.Time(0))
 r,p,yaw=tf.transformations.euler_from_quaternion(quat)
 result['mounts'][topic]={'frame':frame,'position_m':xyz,'rpy_deg':[math.degrees(v) for v in (r,p,yaw)],
  'angle_min_deg':math.degrees(amin),'native_angle_increment_deg':math.degrees(inc),'native_rays':n,
  'sample_stride':stride,'sampled_rays':(n+stride-1)//stride,'sampled_angle_increment_deg':math.degrees(inc)*stride,
  'messages':counts[topic],'range_min_m':rmin,'range_max_m':rmax}
print(json.dumps(result,indent=2,sort_keys=True))
