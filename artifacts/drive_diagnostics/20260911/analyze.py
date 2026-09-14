import json
import math
from pathlib import Path
from statistics import fmean as mean, pstdev as sd

root = Path(__file__).parent
results = []
for path in sorted(root.glob('*.json')):
    data = json.loads(path.read_text())
    if 'trace' not in data:
        continue
    rows = data['trace']
    ts = [x['t'] for x in rows]
    ix = [i for i,t in enumerate(ts) if t >= 4.0 - 1e-6]
    if not ix:
        continue
    selected = [rows[i] for i in ix]
    jv = [x['joint_vel'] for x in selected]
    wz = [x['root_ang_b'][2] for x in selected]
    yaw=[]
    for row in rows:
        qw,qx,qy,qz = row['root_quat']
        a = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        if yaw:
            a = yaw[-1] + (a-yaw[-1]+math.pi)%(2*math.pi)-math.pi
        yaw.append(a)
    start, end = ix[0]-1, ix[-1]
    duration = ts[end] - ts[start]
    w_pose = (yaw[end]-yaw[start])/duration
    wheel_pose = [sum((rows[i]['joint_pos'][k]-rows[i-1]['joint_pos'][k]+math.pi)%(2*math.pi)-math.pi for i in range(start+1,end+1))/duration for k in range(2)]
    wheel_mean = [mean(x[k] for x in jv) for k in range(2)]
    expected = -2*data['args']['target']*.05/.43
    if not data['args']['turn']:
        expected = math.nan
    entry = dict(case=path.stem, tracking_pct=100*mean(wz)/expected,
        yaw=mean(wz), yaw_sd=sd(wz), yaw_from_pose=w_pose,
        wheel_mean=wheel_mean, wheel_sd=[sd(x[k] for x in jv) for k in range(2)],
        wheel_from_pose=wheel_pose,
        body_over_wheel=mean(wz)/((wheel_mean[1]-wheel_mean[0])*.05/.43),
        v_mean=mean(x['root_vel_b'][0] for x in selected),
        normal_mean={}, normal_sd={})
    if data['contact_names']:
        for k, name in enumerate(data['contact_names']):
            fs = [x['contact_net_w'][k][2] for x in selected]
            entry['normal_mean'][name] = mean(fs)
            entry['normal_sd'][name] = sd(fs)
    results.append(entry)
print(json.dumps(results, indent=2))
