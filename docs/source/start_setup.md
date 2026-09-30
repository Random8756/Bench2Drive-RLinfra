# Setup Guide

This page describes the minimum setup needed before running a training or evaluation job.

## Clone the repository

```bash
git clone https://github.com/Random8756/Bench2Drive-RLinfra.git
cd Bench2Drive-RLinfra
```

## Download and setup CARLA 0.9.15

Download the Linux CARLA 0.9.15 package into a sibling directory and import the additional maps:

```bash
mkdir -p ../carla
cd ../carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd ..
bash ImportAssets.sh
cd ../Bench2Drive-RLinfra
```

Use the resulting `carla` directory as `CARLA_ROOT` in the launch scripts.

## Create the Python environment

Python 3.10 is the supported runtime. Python 3.7 is retained only as a legacy,
best-effort environment. The examples below use `venv`, but you can also create an
equivalent Conda environment. The PyTorch versions are references; use the installation
command that matches your CUDA runtime and models.

### Python 3.10

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt
```

### Python 3.7 (legacy)

```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install torch==1.13.1 torchvision==0.14.1
pip install -r tools/requirements/legacy-py37.txt
```

!!! note
    End-to-end RL finetuning recipes (MindDrive, DrivePi0) may have additional dependencies from their upstream model repositories. Read [Finetune Workflow](rl_finetune.md), then check model-specific pages such as [MindDrive Finetune Recipe](rl_finetune_minddrive.md) and [DrivePi0 Finetune Recipe](rl_finetune_drivepi0.md) before launching those workflows.

## Configure CARLA paths

The launch scripts use `CARLA_ROOT` and extend `PYTHONPATH` before running Python entry points.

Edit the relevant script for your machine:

```bash
# RL Baseline Training
tools/launch/baseline/train.sh

# End-to-End RL Finetune
tools/launch/finetune/train.sh

# Distributed RL Finetune
tools/slurm/rl_finetune_sbatch.sh
tools/slurm/rl_finetune_resume_sbatch.sh

# Leaderboard Evaluation
tools/launch/evaluation/run_leaderboard_eval_parallel.sh
```

Training-side launchers select `vendor/carla/training-runtime/`; the evaluation launcher selects `vendor/carla/evaluation-runtime/`. Each launcher also activates the matching ScenarioRunner tree and CARLA PythonAPI path.

## Build the fake bind helper

The parallel CARLA launcher can use a small `LD_PRELOAD` helper to bind CARLA servers to different loopback hosts.

```bash
gcc -Wall -Wextra -O2 -shared -fPIC -o tools/fake_bind.so tools/fake_bind.c -ldl
```

Keep the generated `fake_bind.so` beside `tools/fake_bind.c`.

## Configure the first PPO run

Before launching the default PPO BEV example, edit `configs/ppo_bev_example.yaml` for the machine you are using. The main fields to check are under `env.carla`:

- `num_envs`
- `host`
- `port`
- `traffic_manager_port`
- `gpu_id`

The list-valued fields should match the number of CARLA workers. For a first local smoke test, reduce `num_envs` and the corresponding lists together.
