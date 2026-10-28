"""Replay a saved privileged reference query without launching Isaac Sim."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot',type=Path)
parser.add_argument('--max_seconds',type=float,default=None,help='Override per-leg/pass budget for diagnosis.')
parser.add_argument('--png',type=Path,default=None)
args=parser.parse_args()
pkg=Path(__file__).resolve().parents[2]/'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
def load(name):
    spec=importlib.util.spec_from_file_location(name,pkg/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
p=load('se2_reference_planner')
with np.load(args.snapshot,allow_pickle=False) as data:
    config=json.loads(str(data['planner_config_json'].item()))
    if args.max_seconds is not None:config['max_seconds']=args.max_seconds
    cfg=p.PlannerConfig(**config)
    world=p.CollisionWorld(data['boxes'],data['circles'],float(data['inner_half']))
    start,goal,gates=data['start'],data['goal'],data['gates']
    result=p.DetourReferenceBank(cfg).plan(world,start,goal,gates)
    print(json.dumps(dict(success=result.success,reason=result.reason,failed_leg=result.failed_leg,
        seconds=result.seconds,expansions=result.expansions,poses=len(result.poses),
        rotation_count=result.rotation_count,gear_changes=result.gear_changes),indent=2))
    if args.png is not None:
        load('reference_path_visualization').export_reference_png(args.png,world,result,start,goal,gates,cfg,
            env_id=int(data['env_id']),level=int(data['level']))
        print(args.png.resolve())
sys.exit(0 if result.success else 2)
