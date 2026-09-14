"""Author optional navigation contact proxies before cloning/physics creation.

The source USD, visible cylinders, rigid-body masses/inertias, and joints stay
intact. These proxies approximate a crowned drive tire and bias support toward
the drive wheels; they are not a calibrated tire or caster suspension model.
"""

from __future__ import annotations

import math
from pxr import PhysxSchema, UsdGeom, UsdPhysics


def configure_drive_contacts(stage, robot_root: str, cfg) -> None:
    """Configure the env_0 source, then let the normal scene cloner replicate it."""
    if cfg.drive_wheel_collision_shape not in ("cylinder", "sphere"):
        raise ValueError("drive_wheel_collision_shape must be cylinder or sphere")
    radius = cfg.caster_collision_radius
    if radius is not None and (not math.isfinite(radius) or radius <= 0):
        raise ValueError("caster_collision_radius must be positive or None")

    replacements = []
    for name in ("dwheel_left", "dwheel_right", "caster_fl", "caster_fr", "caster_rl", "caster_rr"):
        parent = stage.GetPrimAtPath(f"{robot_root}/{name}")
        if not parent.IsValid() or not parent.IsA(UsdGeom.Cylinder):
            raise ValueError(f"Expected a cylinder rigid body at {robot_root}/{name}")
        is_drive = name.startswith("dwheel_")
        if (is_drive and cfg.drive_wheel_collision_shape == "cylinder") or (not is_drive and radius is None):
            continue
        replacements.append((parent, is_drive))

    for parent, is_drive in replacements:
        source = UsdGeom.Cylinder(parent)
        proxy_path = parent.GetPath().AppendChild("navigation_contact")
        if is_drive:
            proxy = UsdGeom.Sphere.Define(stage, proxy_path)
            proxy.CreateRadiusAttr().Set(source.GetRadiusAttr().Get())
        else:
            proxy = UsdGeom.Cylinder.Define(stage, proxy_path)
            proxy.CreateRadiusAttr().Set(radius)
            proxy.CreateHeightAttr().Set(source.GetHeightAttr().Get())
            proxy.CreateAxisAttr().Set(source.GetAxisAttr().Get())
        UsdGeom.Imageable(proxy.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        # Preserve explicitly authored contact offsets when present. The link
        # remains the only rigid body; the proxy must not acquire RigidBodyAPI.
        source_physics = PhysxSchema.PhysxCollisionAPI(parent)
        proxy_physics = PhysxSchema.PhysxCollisionAPI.Apply(proxy.GetPrim())
        for getter, creator in (("GetContactOffsetAttr", "CreateContactOffsetAttr"),
                                ("GetRestOffsetAttr", "CreateRestOffsetAttr")):
            attr = getattr(source_physics, getter)()
            if attr and attr.HasAuthoredValueOpinion():
                getattr(proxy_physics, creator)().Set(attr.Get())
        UsdPhysics.CollisionAPI.Apply(parent).CreateCollisionEnabledAttr().Set(False)

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateEnableExternalForcesEveryIterationAttr().Set(
                cfg.drive_enable_external_forces_every_iteration
            )
            print(
                f"[Emma drive] shape={cfg.drive_wheel_collision_shape} "
                f"caster_radius={radius} external_every_iteration="
                f"{api.GetEnableExternalForcesEveryIterationAttr().Get()}", flush=True
            )
