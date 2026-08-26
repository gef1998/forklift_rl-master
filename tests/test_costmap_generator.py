#!/usr/bin/env python3
"""Costmap 生成器独立验证脚本 — 无需 Isaac Sim 环境.

测试: 圆形/矩形投影、膨胀、dropout、body-frame 变换、空障碍物.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source",
    "emma_rl", "emma_rl", "tasks", "direct", "emma"))

# 直接加载 costmap_generator_nav 模块 (绕过 __init__.py 的 isaaclab 依赖)
import importlib.util
_mod_path = os.path.join(os.path.dirname(__file__), "..", "source",
    "emma_rl", "emma_rl", "tasks", "direct",
    "emma", "emma_costmap_nav", "costmap_generator_nav.py")
spec = importlib.util.spec_from_file_location("costmap_generator_nav", _mod_path)
cgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn)

import math

import torch

generate_static_costmap = cgn.generate_static_costmap


def test_circles():
    print("=" * 50)
    print("Test 1: Circle obstacles")
    N = 2
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    robot_quat = torch.tensor([[1., 0., 0., 0.]] * N)

    circle_centers = torch.tensor([
        [[3.0, 0.0], [0.0, -4.0], [6.0, 6.0]],
        [[2.0, 1.0], [0.0, 3.0], [-5.0, -5.0]],
    ])
    circle_radii = torch.tensor([0.3, 0.5, 0.2])

    costmap = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=circle_centers, circle_radii=circle_radii,
        rect_centers_w=None, rect_half_l=None, rect_half_w=None, rect_yaws_w=None,
        grid_size=12.0, map_res=64, inflation=0.25,
    )
    print(f"  Shape: {costmap.shape}, occ env0={costmap[0].mean().item():.4f}, env1={costmap[1].mean().item():.4f}")
    assert costmap.shape == (N, 1, 64, 64)
    print("  PASSED\n")


def test_rectangles():
    print("=" * 50)
    print("Test 2: Rectangle obstacles (with rotation)")
    N = 1
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    robot_quat = torch.tensor([[1., 0., 0., 0.]])

    rect_centers = torch.tensor([[[3.0, 0.0], [0.0, 5.0]]])
    rect_half_l = torch.tensor([2.0, 2.0])
    rect_half_w = torch.tensor([0.3, 0.3])
    rect_yaws = torch.tensor([0.0, math.pi / 4])

    costmap = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        rect_centers_w=rect_centers, rect_half_l=rect_half_l,
        rect_half_w=rect_half_w, rect_yaws_w=rect_yaws,
        circle_centers_w=None, circle_radii=None,
        grid_size=12.0, map_res=64, inflation=0.25,
    )
    print(f"  Shape: {costmap.shape}, occ={costmap[0].mean().item():.4f}")
    assert costmap.shape == (N, 1, 64, 64)
    print("  PASSED\n")


def test_mixed():
    print("=" * 50)
    print("Test 3: Mixed (rectangles + circles)")
    N = 1
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    robot_quat = torch.tensor([[1., 0., 0., 0.]])

    rect_centers = torch.tensor([[[3.0, 0.0]]])
    rect_half_l = torch.tensor([2.0]); rect_half_w = torch.tensor([0.3]); rect_yaws = torch.tensor([0.0])
    circle_centers = torch.tensor([[[6.0, 0.0], [0.0, 4.0]]])
    circle_radii = torch.tensor([0.4, 0.3])

    costmap = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=circle_centers, circle_radii=circle_radii,
        rect_centers_w=rect_centers, rect_half_l=rect_half_l,
        rect_half_w=rect_half_w, rect_yaws_w=rect_yaws,
        grid_size=12.0, map_res=64, inflation=0.25,
    )
    print(f"  Shape: {costmap.shape}, occ={costmap[0].mean().item():.4f}")
    assert costmap.shape == (N, 1, 64, 64)
    print("  PASSED\n")


def test_dropout():
    print("=" * 50)
    print("Test 4: Dropout effect")
    N = 1
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    robot_quat = torch.tensor([[1., 0., 0., 0.]])
    circle_centers = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]]])
    circle_radii = torch.tensor([0.3, 0.3, 0.3, 0.3])

    cm0 = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=circle_centers, circle_radii=circle_radii,
        rect_centers_w=None, rect_half_l=None, rect_half_w=None, rect_yaws_w=None,
        grid_size=12.0, map_res=64, inflation=0.25, dropout_prob=0.0,
    )
    cm1 = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=circle_centers, circle_radii=circle_radii,
        rect_centers_w=None, rect_half_l=None, rect_half_w=None, rect_yaws_w=None,
        grid_size=12.0, map_res=64, inflation=0.25, dropout_prob=0.5,
    )
    print(f"  Without dropout: {cm0[0].mean().item():.4f}")
    print(f"  With dropout 0.5: {cm1[0].mean().item():.4f}")
    assert cm0.shape == cm1.shape == (N, 1, 64, 64)
    assert cm1.mean() <= cm0.mean()
    print("  PASSED\n")


def test_body_frame():
    print("=" * 50)
    print("Test 5: Body-frame transform (Emma rotated 90°)")
    N = 1
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    half_yaw = math.pi / 4
    robot_quat = torch.tensor([[math.cos(half_yaw), 0., 0., math.sin(half_yaw)]])

    circle_centers = torch.tensor([[[6.0, 0.0]]])  # world +X → body -Y
    circle_radii = torch.tensor([0.5])

    costmap = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=circle_centers, circle_radii=circle_radii,
        rect_centers_w=None, rect_half_l=None, rect_half_w=None, rect_yaws_w=None,
        grid_size=12.0, map_res=64, inflation=0.25,
    )
    print(f"  Shape: {costmap.shape}, occ={costmap[0].mean().item():.4f}")
    print("  → Obstacle should appear at bottom of image (body -Y = left side)")
    assert costmap.shape == (N, 1, 64, 64)
    print("  PASSED\n")


def test_empty():
    print("=" * 50)
    print("Test 6: Empty obstacles")
    N = 4
    robot_pos = torch.zeros(N, 3); robot_pos[:, 2] = 0.045
    robot_quat = torch.tensor([[1., 0., 0., 0.]] * N)

    costmap = generate_static_costmap(
        robot_pos_w=robot_pos, robot_quat_w=robot_quat,
        circle_centers_w=torch.zeros(N, 0, 2), circle_radii=torch.zeros(0),
        rect_centers_w=None, rect_half_l=None, rect_half_w=None, rect_yaws_w=None,
        grid_size=12.0, map_res=64, inflation=0.25,
    )
    print(f"  Shape: {costmap.shape}, all_zeros={(costmap == 0).all().item()}")
    assert costmap.shape == (N, 1, 64, 64)
    assert (costmap == 0).all()
    print("  PASSED\n")


if __name__ == "__main__":
    print("Costmap Generator Validation Suite\n")
    try:
        test_circles()
        test_rectangles()
        test_mixed()
        test_dropout()
        test_body_frame()
        test_empty()
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
