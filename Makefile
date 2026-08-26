SHELL := /bin/bash

TASK ?= Emma-RL-ObsNav-Direct-v0
NUM_ENVS ?= 128

.PHONY: help install list-envs train play random-agent zero-agent check test-action-control test-ring-layout test-nav-reward test-safety-control test-beta-policy test-costmap test-async-lidar

help:
	@echo "Forklift RL (Emma Nav / Emma Costmap)"
	@echo "make train TASK=<gym-id> NUM_ENVS=128"
	@echo "make play TASK=<gym-id> CHECKPOINT=<path>"

install:
	TASK="$(TASK)" bash ./launch.sh install

list-envs:
	TASK="$(TASK)" bash ./launch.sh list-envs

train:
	TASK="$(TASK)" bash ./launch.sh train --num_envs $(NUM_ENVS) $(EXTRA_ARGS)

play:
	TASK="$(TASK)" bash ./launch.sh play --num_envs $(NUM_ENVS) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) $(EXTRA_ARGS)

random-agent:
	TASK="$(TASK)" bash ./launch.sh random-agent --num_envs $(NUM_ENVS) $(EXTRA_ARGS)

zero-agent:
	TASK="$(TASK)" bash ./launch.sh zero-agent --num_envs $(NUM_ENVS) $(EXTRA_ARGS)

check:
	bash ./launch.sh check

test-action-control:
	bash ./launch.sh test-action-control

test-ring-layout:
	bash ./launch.sh test-ring-layout

test-nav-reward:
	bash ./launch.sh test-nav-reward

test-safety-control:
	bash ./launch.sh test-safety-control

test-beta-policy:
	bash ./launch.sh test-beta-policy

test-costmap:
	bash ./launch.sh test-costmap

test-async-lidar:
	bash ./launch.sh test-async-lidar
