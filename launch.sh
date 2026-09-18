#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAACLAB_ROOT}/_isaac_sim/python.sh}"
TASK="${TASK:-Emma-RL-ObsNav-Direct-v0}"
MPPI_TASK="${MPPI_TASK:-Emma-RL-MPPINav-Direct-v0}"

usage() {
    cat <<EOF
Forklift RL launcher

Usage: bash launch.sh <command> [arguments]

Commands:
  install       Install the local Isaac Lab package
  list-envs     List registered project environments
  train         Train PPO on TASK
  play          Play a checkpoint on TASK
  evaluate      Run the legacy continuous play/evaluate loop
  test          Scan three documented test scenarios and save metrics
  export        Export a checkpoint to a deployable ONNX policy (--checkpoint, TASK)
  random-agent  Run TASK with random actions
  zero-agent    Run TASK with zero actions
  wheel-response  Run one-environment constant wheel velocity diagnostics
  collect-mppi  Collect MPPI expert demonstrations for the new navigation task
  train-bc      Train the Beta actor from an MPPI dataset
  train-mppi-ppo  Initialize PPO from BC_CHECKPOINT and fine-tune
  test-action-control  Run pure Torch action mapping tests
  test-random-obstacle-layout  Run pure Torch random-obstacle geometry tests
  test-ring-layout     Run pure Torch local-maze geometry tests
  test-nav-reward      Run pure Torch navigation reward scaling tests
  test-safety-control  Run pure Torch runtime safety tests
  test-beta-policy  Run pure Torch signed Beta policy tests
  test-async-lidar  Run pure Torch async LiDAR tests
  test-costmap      Run pure Torch Costmap tests
  test-mppi         Run pure Torch MPPI geometry, action, and planner tests
  check         Check the configured Isaac Lab Python

Environment variables:
  ISAACLAB_ROOT  Isaac Lab checkout (default: ${ISAACLAB_ROOT})
  ISAAC_PYTHON   Isaac Sim/Isaac Lab Python launcher (default: ${ISAAC_PYTHON})
  EMMA_USD_PATH  Path to Emma_no_ground.usd
  TASK           Gym task id (default: ${TASK})
  MPPI_TASK      MPPI task id (default: ${MPPI_TASK})
  BC_CHECKPOINT  BC actor used by train-mppi-ppo
EOF
}

require_python() {
    if [[ ! -x "${ISAAC_PYTHON}" ]]; then
        echo "Isaac Lab Python is not executable: ${ISAAC_PYTHON}" >&2
        echo "Set ISAAC_PYTHON or ISAACLAB_ROOT before running this command." >&2
        exit 1
    fi
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${command}" in
    install)
        require_python
        "${ISAAC_PYTHON}" -m pip install -e "${ROOT_DIR}/source/emma_rl"
        ;;
    list-envs)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/list_envs.py" "$@"
        ;;
    train)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/train.py" --task "${TASK}" --algorithm PPO "$@"
        ;;
    play|evaluate)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/play.py" --task "${TASK}" --algorithm PPO "$@"
        ;;
    test)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/test.py" --task "${TASK}" --algorithm PPO "$@"
        ;;
    export)
        # TASK selects the network, so a GRU run needs
        # TASK=Emma-RL-ObsNav-GRU-Direct-v0. A checkpoint trained under the other
        # task is reported rather than exported with the wrong network.
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/export.py" --task "${TASK}" "$@"
        ;;
    random-agent)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/random_agent.py" --task "${TASK}" "$@"
        ;;
    zero-agent)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/zero_agent.py" --task "${TASK}" "$@"
        ;;
    wheel-response)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/play.py" \
            --task "${TASK}" \
            --algorithm PPO \
            --num_envs 1 \
            --fixed-wheel-cmd "${WHEEL_CMD_LEFT:-3.0}" "${WHEEL_CMD_RIGHT:-3.0}" \
            --fixed-wheel-steps "${WHEEL_CMD_STEPS:-150}" \
            --action-plot-episodes 1 \
            "$@"
        ;;
    collect-mppi)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/mppi_bc/collect_expert.py" \
            --task "${MPPI_TASK}" \
            "$@"
        ;;
    train-bc)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/mppi_bc/train_bc.py" "$@"
        ;;
    train-mppi-ppo)
        require_python
        if [[ -z "${BC_CHECKPOINT:-}" ]]; then
            echo "Set BC_CHECKPOINT to the MPPI behavior-cloning actor checkpoint" >&2
            exit 2
        fi
        "${ISAAC_PYTHON}" "${ROOT_DIR}/scripts/skrl/train.py" \
            --task "${MPPI_TASK}" \
            --algorithm PPO \
            --bc-checkpoint "${BC_CHECKPOINT}" \
            "$@"
        ;;
    test-action-control)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_action_control.py" "$@"
        ;;
    test-random-obstacle-layout)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_random_obstacle_layout.py" "$@"
        ;;
    test-ring-layout)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_ring_layout.py" "$@"
        ;;
    test-nav-reward)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_navigation_reward_scaling.py" "$@"
        ;;
    test-safety-control)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_safety_control.py" "$@"
        ;;
    test-beta-policy)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_beta_policy.py" "$@"
        ;;
    test-async-lidar)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_async_lidar.py" "$@"
        ;;
    test-costmap)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_lidar_costmap.py" "$@"
        ;;
    test-mppi)
        require_python
        "${ISAAC_PYTHON}" "${ROOT_DIR}/tests/test_mppi_pipeline.py" "$@"
        ;;
    check)
        require_python
        "${ISAAC_PYTHON}" -c "import isaaclab; print('Isaac Lab import: OK')"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: ${command}" >&2
        usage >&2
        exit 2
        ;;
esac
