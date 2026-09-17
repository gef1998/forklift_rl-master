# Random obstacles: connected procedural layouts

`world_mode="random_obstacles"` now samples one joint distribution of walls and
mixed cylinder/cuboid clutter. It does not select a named S/ring/straight scene.

- A randomized spanning tree opens a 3×3 cell graph. Additional independently
  opened edges create loops, including perimeter routes around central walls.
  The same graph distribution includes alternating turns and straight runs.
- Wall positions receive independent normal offsets. Each environment samples
  its own loop probability, clutter density, clutter positions and cuboid yaws.
- The existing 15 clutter templates become capacity slots: sampling may leave
  slots inactive when spacing or density limits require it. Twelve additional
  wall slots support all possible internal cell boundaries. Inactive slots are
  below ground and absent from the LiDAR mesh.
- Clutter-to-clutter edge spacing remains 1.5 m. Wall clearance is separately
  controlled by `obstacle_wall_clearance`; clutter can obstruct straight routes.
- Start/goal banks are generated only after obstacles have been placed. A
  four-neighbour grid inflated by `navigation_clearance` plus half a grid-cell
  diagonal certifies continuous paths for the robot's circumscribed footprint.
  Fragmented maps with less than 90% of free grid nodes in one component are
  rejected. Both endpoints belong to that component, respect spawn clearances,
  and remain more than 5.2 m apart.

Configuration lives in `EmmaObsNavEnvCfg.random_obstacle_world`:

- `loop_probability_range=(0.0, 0.65)`: independently sampled per map; lower values
  retain more detours, higher values open more alternative routes.
- `clutter_density_range=(0.35, 1.0)`: per-map activation probability range;
  obstacle capacity and safe placement further bound the actual count.
- `wall_jitter=0.20`: normal wall-position perturbation in metres.
- `navigation_clearance=0.60`: robot-center clearance, including turning room.
- `connectivity_grid_resolution=0.15`: conservative reachability grid spacing.
- `spawn_bank_size=128`: independently sampled start/goal pairs per map.
- `structured_layout=False`: restores the previous pure scattered-obstacle mode.

As before, map geometry is generated once per environment at initialization.
Episode resets sample a new bank entry and a fresh heading; they do not rebuild
PhysX bodies or the global Warp mesh. Torch's training seed also seeds the CPU
geometry generator. Thus 1024 environments provide 1024 independent maps, with
128 candidate tasks per map. This is geometric reachability, not a guarantee
that the current policy or local VFH guidance can finish every detour in time.

Run the geometry checks with the project's Isaac Lab Python:

```bash
python tests/test_structured_random_layout.py
python tests/test_random_obstacle_layout.py
```

The structured tests check reproducibility, topology diversity, S-like shortest
graph paths, perimeter loops, clutter blocking direct routes, boundary/obstacle
clearance, and independently flood-fill all sampled start/goal endpoints.

## Playback seeds

`play.py` uses a fresh random seed on each launch by default and prints it.
Pass `--seed 123` to reproduce a map and random stream. The environment, skrl wrapper and Runner receive the same seed. Seeding must
happen before wrapper construction: the wrapper caches a key (otherwise 0)
and passes it to the first reset. Runner also reseeds from its YAML config
(otherwise 42), so both sources must be aligned.
Episode resets still reuse that launch's map and resample start/goal pairs.
