"""Measure Emma AMR wheel radius via simulation.

Output file: /tmp/emma_measure_result.txt
Usage:
    EMMA_USD_PATH=/path/to/Emma_no_ground.usd $ISAAC_PYTHON tools/measure_wheel_radius.py
"""

import os
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import traceback
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage

DEFAULT_USD_PATH = Path(__file__).resolve().parents[1] / "assets" / "Emma_no_ground.usd"
USD_PATH = os.environ.get("EMMA_USD_PATH", str(DEFAULT_USD_PATH))
OUTPUT = "/tmp/emma_measure_result.txt"


def write_log(fp, msg):
    fp.write(msg + "\n")
    fp.flush()


def run_measure(fp, world, art):
    """Step simulation at fixed wheel velocities, measure body velocity."""
    results = []
    test_omegas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    for omega in test_omegas:
        try:
            # Build action: velocity targets
            n_dof = art.num_dof
            vels = np.zeros(n_dof, dtype=np.float64)
            # wheel_left=DOF0, wheel_right=DOF1 (from inspection)
            vels[0] = omega
            vels[1] = omega

            # Use set_joint_velocity_targets directly (avoid apply_action API mismatch)
            art.set_joint_velocity_targets(velocities=vels)

            # Step to settle
            for _ in range(120):
                world.step(render=False)

            # Read velocities
            lin_vel = art.get_linear_velocities()
            joint_vel = art.get_joint_velocities()

            # Handle tensor or ndarray return
            if hasattr(lin_vel, 'cpu'):
                lin_vel = lin_vel.cpu().numpy()
                joint_vel = joint_vel.cpu().numpy()

            # Flatten in case of batch dim
            lv = lin_vel.reshape(-1)
            jv = joint_vel.reshape(-1)

            v_x = float(lv[0])
            w_L = float(jv[0])
            w_R = float(jv[1])
            w_avg = (w_L + w_R) / 2.0

            r_eff = v_x / w_avg if abs(w_avg) > 1e-4 else 0.0

            write_log(fp, "  w_set=%.1f  w_L=%.3f w_R=%.3f w_avg=%.3f -> v_x=%.4f -> r_eff=%.4f m" %
                      (omega, w_L, w_R, w_avg, v_x, r_eff))
            results.append({"omega": omega, "v_x": v_x, "r_eff": r_eff, "w_avg": w_avg})

        except Exception as e:
            write_log(fp, "  ERROR during omega=%.1f: %s" % (omega, e))
            write_log(fp, traceback.format_exc())

        # Stop
        try:
            art.set_joint_velocity_targets(velocities=np.zeros(art.num_dof, dtype=np.float64))
            for _ in range(30):
                world.step(render=False)
        except Exception:
            pass

    return results


def main():
    with open(OUTPUT, 'w') as fp:
        write_log(fp, "=" * 60)
        write_log(fp, "Emma Wheel Radius Measurement")
        write_log(fp, "=" * 60)

        # Setup world
        write_log(fp, "\n[1] Creating world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        write_log(fp, "[2] Loading Emma asset...")
        try:
            add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/Emma")
            prim_path = "/World/Emma"
            write_log(fp, "    Prim path: %s" % prim_path)
        except Exception as e:
            write_log(fp, "    ERROR loading asset: %s" % e)
            write_log(fp, traceback.format_exc())
            simulation_app.close()
            return

        write_log(fp, "[3] Creating articulation...")
        try:
            art = Articulation(prim_paths_expr=prim_path, name="emma_robot")
            world.scene.add(art)
            write_log(fp, "    DOF names: %s" % str(art.dof_names))
            write_log(fp, "    num_dof: %s" % str(art.num_dof))
        except Exception as e:
            write_log(fp, "    ERROR creating articulation: %s" % e)
            write_log(fp, traceback.format_exc())
            simulation_app.close()
            return

        write_log(fp, "[4] Resetting world...")
        try:
            world.reset()
            write_log(fp, "    DOF names after reset: %s" % str(art.dof_names))
            write_log(fp, "    num_dof after reset: %s" % str(art.num_dof))
        except Exception as e:
            write_log(fp, "    ERROR resetting world: %s" % e)
            write_log(fp, traceback.format_exc())
            simulation_app.close()
            return

        # Identify wheel DOFs
        dof_names = art.dof_names
        if dof_names is None:
            write_log(fp, "ERROR: dof_names is None after reset!")
            simulation_app.close()
            return

        wheel_indices = []
        for i, name in enumerate(dof_names):
            if "wheel" in name.lower():
                wheel_indices.append(i)
                write_log(fp, "    Drive wheel DOF[%d]: %s" % (i, name))
        write_log(fp, "    Found %d drive wheels" % len(wheel_indices))

        if len(wheel_indices) < 2:
            write_log(fp, "ERROR: Need at least 2 drive wheels!")
            simulation_app.close()
            return

        # Measure
        write_log(fp, "\n[5] Measuring wheel radius (straight line, both wheels same velocity)...")
        results = run_measure(fp, world, art)

        valid = [r for r in results if abs(r["r_eff"]) > 0.001 and abs(r["r_eff"]) < 0.5]
        if valid:
            radii = [r["r_eff"] for r in valid]
            avg_r = np.mean(radii)
            std_r = np.std(radii)
            write_log(fp, "\n  >>> Effective radius: %.4f +/- %.4f m" % (avg_r, std_r))
            write_log(fp, "  >>> Geometric radius (USD): 0.09 m")
            write_log(fp, "  >>> Ratio effective/geometric: %.4f" % (avg_r / 0.09))
        else:
            write_log(fp, "\n  No valid measurements!")

        # ================================================================
        # 6. Measure wheel base (in-place rotation: w_L = -w_R)
        # ================================================================
        write_log(fp, "\n[6] Measuring wheel base (in-place rotation: w_L = -w_R)...")

        omega_test = 5.0  # rad/s
        base_results = []

        for direction in [+1, -1]:
            try:
                vels = np.zeros(art.num_dof, dtype=np.float64)
                vels[0] = -omega_test * direction  # left wheel
                vels[1] = +omega_test * direction  # right wheel (opposite)
                art.set_joint_velocity_targets(velocities=vels)

                for _ in range(120):
                    world.step(render=False)

                ang_vel = art.get_angular_velocities()
                joint_vel = art.get_joint_velocities()

                if hasattr(ang_vel, 'cpu'):
                    ang_vel = ang_vel.cpu().numpy()
                    joint_vel = joint_vel.cpu().numpy()

                av = ang_vel.reshape(-1)
                jv = joint_vel.reshape(-1)

                omega_z = float(av[2])  # body angular velocity about Z
                w_L = float(jv[0])
                w_R = float(jv[1])
                delta_w = w_R - w_L  # should equal 2*omega_test

                # L = (w_R - w_L) * r / omega_z
                # Use avg radius from previous step
                L_eff = delta_w * avg_r / omega_z if abs(omega_z) > 1e-4 else 0.0

                write_log(fp, "  dir=%+d: w_L=%.3f w_R=%.3f delta_w=%.3f -> omega_z=%.4f -> L=%.4f m" %
                          (direction, w_L, w_R, delta_w, omega_z, L_eff))
                base_results.append(L_eff)

            except Exception as e:
                write_log(fp, "  ERROR during wheel base measurement: %s" % e)
                write_log(fp, traceback.format_exc())

            # Stop
            try:
                art.set_joint_velocity_targets(velocities=np.zeros(art.num_dof, dtype=np.float64))
                for _ in range(30):
                    world.step(render=False)
            except Exception:
                pass

        valid_bases = [b for b in base_results if abs(b) > 0.01 and abs(b) < 2.0]
        if valid_bases:
            avg_base = np.mean(valid_bases)
            write_log(fp, "\n  >>> Wheel base: %.4f m" % avg_base)

        # ================================================================
        # 7. Summary
        # ================================================================
        write_log(fp, "\n" + "=" * 60)
        write_log(fp, "SUMMARY FOR RL ENV CONFIG")
        write_log(fp, "=" * 60)
        write_log(fp, "  wheel_radius = %.4f  # m (effective, sim-verified)" % avg_r)
        if valid_bases:
            write_log(fp, "  wheel_base   = %.4f  # m (distance between L/R wheels)" % avg_base)
        write_log(fp, "")
        write_log(fp, "  Differential drive kinematics:")
        write_log(fp, "    v_lin  = (w_L + w_R) / 2 * wheel_radius")
        write_log(fp, "    v_ang  = (w_R - w_L) / wheel_base * wheel_radius")
        write_log(fp, "    w_L = (v_lin - v_ang * wheel_base/2) / wheel_radius")
        write_log(fp, "    w_R = (v_lin + v_ang * wheel_base/2) / wheel_radius")

        write_log(fp, "\nDone.")

    print("\n\nResults written to: %s" % OUTPUT)
    simulation_app.close()


if __name__ == "__main__":
    main()
