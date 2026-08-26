#!/usr/bin/env python3
"""RayCaster 版本诊断: 对比 isaaclab 2.1.1 vs 2.3.2 的行为差异.

用法:
  $ISAAC_PYTHON tools/diag_raycaster.py

测试内容:
  1. import fabric / RayCaster 是否成功
  2. make_plane(2e6,2e6) → convert_to_warp_mesh 耗时
  3. make_plane(200,200)  → convert_to_warp_mesh 耗时
  4. Plane prim 创建 + RayCaster._initialize_impl() 耗时
  5. RayCaster.update() 耗时
"""

from __future__ import annotations
import argparse, math, sys, time, traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import BpearlPatternCfg


def log(msg):
    print(msg, flush=True)


def test_1_check_versions():
    """打印版本信息 + fabric import 测试."""
    log("=" * 60)
    log("TEST 1: Version check + fabric import")
    log("=" * 60)

    import isaaclab
    log(f"  isaaclab version: {isaaclab.__version__}")
    log(f"  isaaclab path:    {isaaclab.__file__}")

    # Check if MultiMeshRayCaster is importable (__init__.py imports it)
    try:
        from isaaclab.sensors.ray_caster import MultiMeshRayCaster
        log(f"  MultiMeshRayCaster import: OK")
        log(f"  __init__ includes MultiMesh: YES")
    except Exception as e:
        log(f"  MultiMeshRayCaster import: FAIL — {e}")
        log(f"  __init__ includes MultiMesh: FAIL")

    # Check fabric import
    try:
        from isaaclab.utils.warp import fabric
        log(f"  fabric.py import: OK")
    except Exception as e:
        log(f"  fabric.py import: FAIL — {e}")

    # Check warp version
    import warp as wp
    log(f"  warp version: {wp.__version__ if hasattr(wp, '__version__') else 'N/A'}")


def test_2_make_plane_warp_conversion():
    """测试不同 plane 大小的 make_plane + convert_to_warp_mesh 耗时."""
    log("\n" + "=" * 60)
    log("TEST 2: make_plane → convert_to_warp_mesh timing")
    log("=" * 60)

    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")

    for plane_size in [(2e6, 2e6), (200, 200), (30, 30)]:
        t0 = time.time()
        mesh = make_plane(size=plane_size, height=0.0, center_zero=True)
        log(f"  make_plane({plane_size}): {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces, t={time.time()-t0:.3f}s")

        t0 = time.time()
        try:
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
            log(f"    convert_to_warp_mesh: OK, t={time.time()-t0:.3f}s")
        except Exception as e:
            log(f"    convert_to_warp_mesh: FAIL — {e}, t={time.time()-t0:.3f}s")


def test_3_raycaster_init_with_plane():
    """Plane prim 场景 + RayCaster._initialize_impl() 耗时."""
    log("\n" + "=" * 60)
    log("TEST 3: RayCaster._initialize_impl() with Plane prim")
    log("=" * 60)

    # Create minimal scene
    from pxr import UsdGeom
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    if not stage.GetPrimAtPath("/World/groundPlane"):
        stage.DefinePrim("/World/groundPlane", "Xform")
    UsdGeom.Plane.Define(stage, "/World/groundPlane/Plane")
    log("  Plane prim created at /World/groundPlane/Plane")

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01, render_interval=2, physics_material=sim_utils.RigidBodyMaterialCfg()
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.reset()
    log("  SimulationContext created + reset")

    # Create RayCaster
    lidar_cfg = RayCasterCfg(
        prim_path="/World/groundPlane",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
        ray_alignment="world",
        pattern_cfg=BpearlPatternCfg(
            horizontal_fov=360.0, horizontal_res=5.0, vertical_ray_angles=[0.0]
        ),
        mesh_prim_paths=["/World/groundPlane"],
        max_distance=12.0,
    )

    t0 = time.time()
    lidar = RayCaster(lidar_cfg)
    log(f"  RayCaster created: t={time.time()-t0:.3f}s")

    # step 1: _initialize_impl (includes _initialize_warp_meshes which does make_plane)
    t0 = time.time()
    try:
        lidar._initialize_impl()
        log(f"  _initialize_impl(): OK, num_rays={lidar.num_rays}, t={time.time()-t0:.3f}s")
    except Exception as e:
        log(f"  _initialize_impl(): FAIL — {e}, t={time.time()-t0:.3f}s")
        return

    # step 2: update (includes first Warp kernel launch)
    t0 = time.time()
    try:
        lidar.update(0.01)
        has_data = lidar.data.ray_hits_w is not None
        log(f"  update(): OK, has_data={has_data}, t={time.time()-t0:.3f}s")
        if has_data:
            log(f"    ray_hits_w shape: {list(lidar.data.ray_hits_w.shape)}")
    except Exception as e:
        log(f"  update(): FAIL — {e}, t={time.time()-t0:.3f}s")


def test_4_raycaster_with_obstacleworld_mesh():
    """模拟 ObstacleWorld.build_warp_mesh + inject 的完整流程."""
    log("\n" + "=" * 60)
    log("TEST 4: Full RayCaster flow with ObstacleWorld mesh injection")
    log("=" * 60)

    from pxr import UsdGeom
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh
    import trimesh

    # Create scene
    stage = get_current_stage()
    if not stage.GetPrimAtPath("/World/groundPlane"):
        stage.DefinePrim("/World/groundPlane", "Xform")
    UsdGeom.Plane.Define(stage, "/World/groundPlane/Plane")
    log("  Plane prim OK")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=2,
                                       physics_material=sim_utils.RigidBodyMaterialCfg())
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.reset()
    log("  Sim OK")

    lidar_cfg = RayCasterCfg(
        prim_path="/World/groundPlane",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
        ray_alignment="world",
        pattern_cfg=BpearlPatternCfg(
            horizontal_fov=360.0, horizontal_res=5.0, vertical_ray_angles=[0.0]
        ),
        mesh_prim_paths=["/World/groundPlane"],
        max_distance=12.0,
    )

    lidar = RayCaster(lidar_cfg)
    log("  RayCaster created")

    # Init lidar (creates default plane warp mesh)
    t0 = time.time()
    lidar._initialize_impl()
    log(f"  _initialize_impl done: t={time.time()-t0:.3f}s")

    # Build replacement warp mesh (simulate ObstacleWorld)
    t0 = time.time()
    ground = trimesh.creation.box(extents=[30.0, 30.0, 0.02])
    ground.apply_translation([0, 0, -0.01])
    # Add a couple cylinders (simulate obstacles)
    for i in range(5):
        cyl = trimesh.creation.cylinder(radius=0.3, height=3.0, sections=24)
        cyl.apply_translation([2.0 + i * 2.0, 1.0, 1.5])
        ground = trimesh.util.concatenate([ground, cyl])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wp_mesh = convert_to_warp_mesh(ground.vertices, ground.faces, device=device)
    log(f"  build_warp_mesh: {len(ground.vertices)} verts, {len(ground.faces)} faces, t={time.time()-t0:.3f}s")

    # Inject
    mesh_key = lidar_cfg.mesh_prim_paths[0]
    lidar.meshes[mesh_key] = wp_mesh
    log(f"  mesh injected → meshes[{mesh_key}]")

    # Update
    t0 = time.time()
    lidar.update(0.01)
    log(f"  update done: t={time.time()-t0:.3f}s")
    if lidar.data.ray_hits_w is not None:
        dist = torch.norm(
            lidar.data.ray_hits_w[..., :2] - lidar.data.pos_w[:, None, :2], dim=-1
        )
        log(f"  dist min={dist.min().item():.3f} max={dist.max().item():.3f}")
    else:
        log(f"  WARN: no ray_hits_w!")


def main():
    log("=" * 60)
    log("  RayCaster Diagnostic — isaaclab version comparison")
    log("=" * 60)

    test_1_check_versions()

    try:
        test_2_make_plane_warp_conversion()
    except Exception as e:
        log(f"  TEST 2 FAILED: {e}")
        traceback.print_exc()

    try:
        test_3_raycaster_init_with_plane()
    except Exception as e:
        log(f"  TEST 3 FAILED: {e}")
        traceback.print_exc()

    try:
        test_4_raycaster_with_obstacleworld_mesh()
    except Exception as e:
        log(f"  TEST 4 FAILED: {e}")
        traceback.print_exc()

    log("\n" + "=" * 60)
    log("  Done")
    log("=" * 60)
    simulation_app.close()


if __name__ == "__main__":
    main()
