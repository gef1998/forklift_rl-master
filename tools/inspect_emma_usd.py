"""Inspect Emma_no_ground.usd articulation structure via Isaac Sim.

Usage:
    EMMA_USD_PATH=/path/to/Emma_no_ground.usd $ISAAC_PYTHON tools/inspect_emma_usd.py
Output written to: /tmp/emma_inspect_result.txt
"""

import os
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage

DEFAULT_USD_PATH = Path(__file__).resolve().parents[1] / "assets" / "Emma_no_ground.usd"
USD_PATH = os.environ.get("EMMA_USD_PATH", str(DEFAULT_USD_PATH))
OUTPUT_FILE = "/tmp/emma_inspect_result.txt"

# Use pxr directly
from pxr import Usd, UsdPhysics, UsdGeom


def inspect_via_pxr(f):
    """Inspect USD via pxr directly (before sim init)."""
    stage = Usd.Stage.Open(USD_PATH)
    if not stage:
        f.write("ERROR: Failed to open stage!\n")
        return

    f.write("=" * 80 + "\n")
    f.write("USD PRIM INSPECTION (via pxr)\n")
    f.write("=" * 80 + "\n")

    # All prims related to physics/joints/wheels
    joints = []
    wheels = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()

        attrs = []

        # Articulation root
        if prim.IsA(UsdPhysics.ArticulationRootAPI):
            attrs.append("ARTICULATION_ROOT")

        # Revolute joints
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis = prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else "?"
            lower = prim.GetAttribute("physics:lowerLimit").Get() if prim.HasAttribute("physics:lowerLimit") else "?"
            upper = prim.GetAttribute("physics:upperLimit").Get() if prim.HasAttribute("physics:upperLimit") else "?"
            attrs.append(f"RevoluteJoint(axis={axis}, limits=[{lower}, {upper}])")
            joints.append((path, "Revolute", axis, lower, upper))

            # Check for drive API
            if prim.HasAttribute("drive:angular:physics:damping"):
                damping = prim.GetAttribute("drive:angular:physics:damping").Get()
                stiffness = prim.GetAttribute("drive:angular:physics:stiffness").Get()
                attrs.append(f"AngularDrive(damping={damping}, stiffness={stiffness})")

        if prim.IsA(UsdPhysics.PrismaticJoint):
            axis = prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else "?"
            lower = prim.GetAttribute("physics:lowerLimit").Get() if prim.HasAttribute("physics:lowerLimit") else "?"
            upper = prim.GetAttribute("physics:upperLimit").Get() if prim.HasAttribute("physics:upperLimit") else "?"
            attrs.append(f"PrismaticJoint(axis={axis}, limits=[{lower}, {upper}])")
            joints.append((path, "Prismatic", axis, lower, upper))

        if prim.IsA(UsdPhysics.FixedJoint):
            attrs.append("FixedJoint")
            joints.append((path, "Fixed", None, None, None))

        # Rigid body
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            attrs.append("RigidBody")

        # Mass
        if prim.HasAttribute("physics:mass"):
            mass = prim.GetAttribute("physics:mass").Get()
            if mass is not None:
                attrs.append(f"mass={mass:.4f}")

        # Cylinder geometry = potential wheel
        if prim.IsA(UsdGeom.Cylinder):
            radius = prim.GetAttribute("radius").Get() if prim.HasAttribute("radius") else None
            height = prim.GetAttribute("height").Get() if prim.HasAttribute("height") else None
            attrs.append(f"Cylinder(radius={radius}, height={height})")
            if any(kw in path.lower() for kw in ["wheel", "drive"]):
                wheels.append((path, radius, height))

        # Any prim with "wheel" in path
        if "wheel" in path.lower() and not attrs:
            attrs.append(f"[type={type_name}]")

        if attrs:
            depth = max(path.count("/") - 1, 0)
            indent = "  " * depth
            f.write(f"{indent}{path}  [{type_name}]\n")
            for a in attrs:
                f.write(f"{indent}  -> {a}\n")

    # Summary
    f.write("\n" + "=" * 80 + "\n")
    f.write("JOINTS SUMMARY\n")
    f.write("=" * 80 + "\n")
    for j in joints:
        f.write(f"  {j[0]}: type={j[1]}, axis={j[2]}, limits=[{j[3]}, {j[4]}]\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("POTENTIAL WHEELS\n")
    f.write("=" * 80 + "\n")
    for w in wheels:
        f.write(f"  {w[0]}: radius={w[1]}, height={w[2]}\n")

    # NEW: traverse all prims to find any mesh reference with "wheel" or "drive" in the name
    f.write("\n" + "=" * 80 + "\n")
    f.write("ALL PRIMS (full hierarchy)\n")
    f.write("=" * 80 + "\n")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        f.write(f"  {path}  [{prim.GetTypeName()}]\n")

    return len(joints), len(wheels)


def inspect_via_core_api(f):
    """Inspect articulation via Isaac Sim Core API after physics init."""
    f.write("\n" + "=" * 80 + "\n")
    f.write("ISAAC SIM CORE API INSPECTION\n")
    f.write("=" * 80 + "\n")

    try:
        world = World(stage_units_in_meters=1.0)
        prim_path = add_reference_to_stage(USD_PATH, "/World/Emma")
        f.write(f"Added reference, prim_path returned: '{prim_path}'\n")

        world.reset()

        # Find the correct articulation root path
        # add_reference_to_stage returns the path to the first prim in the referenced USD
        art = Articulation(prim_path)
        art.initialize()
        f.write(f"Articulation initialized at: {prim_path}\n")
        f.write(f"  num_joints: {art.num_joints}\n")
        f.write(f"  joint_names: {art.joint_names}\n")
        f.write(f"  num_dof: {art.num_dof}\n")
        if art.num_joints > 0:
            f.write(f"  dof_names: {art.dof_names}\n")
            f.write(f"  dof_types: {art.dof_types}\n")
            f.write(f"  dof_paths: {art.dof_paths}\n")

        f.write("\nAll articulation info:\n")
        f.write(f"  joint_limits:\n")
        for name, limit in zip(art.joint_names, art.get_joint_limits()):
            f.write(f"    {name}: {limit}\n")

        # Print velocities to check which joints can be driven
        f.write(f"\n  dof_properties:\n")
        if hasattr(art, 'dof_properties'):
            f.write(f"    {art.dof_properties}\n")

    except Exception as e:
        f.write(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc(file=f)


def main():
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Emma_no_ground.usd Inspection Results\n")
        f.write("=" * 80 + "\n\n")

        n_joints, n_wheels = inspect_via_pxr(f)
        f.write(f"\nTotal joints found: {n_joints}\n")
        f.write(f"Total wheel candidates: {n_wheels}\n")

        inspect_via_core_api(f)

    print(f"\n\nResults written to: {OUTPUT_FILE}")
    simulation_app.close()


if __name__ == "__main__":
    main()
