#!/usr/bin/env python3
"""RL finetune training entry point.

Run with:

    python -m b2d_rlinfra.finetuning.train --config configs/rl_finetune_rgb_example.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from b2d_rlinfra.finetuning.collector import rl_ppo_config
from b2d_rlinfra.finetuning.config_schema import distributed_enabled, normalize_rl_finetune_config
from b2d_rlinfra.finetuning.coordinator import Coordinator
from b2d_rlinfra.learning.utils.config import load_config

logger = logging.getLogger("RLFinetune.Train")


def _create_run_dir(base_log_dir: str, prefix: str = "") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}_{ts}" if prefix else ts
    run_dir = Path(base_log_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def _save_run_config(config: Any, run_dir: Path) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def _configure_logging(verbose: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(int(verbose), logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train(config_path: str, max_updates: int | None = None, init_from_checkpoint: str | None = None) -> Path:
    config = load_config(config_path)
    raw_config = normalize_rl_finetune_config(config.to_dict())
    config.raw_config = raw_config
    config.env_config = raw_config.get("env")
    if distributed_enabled(raw_config):
        raise ValueError(
            "distributed rl_finetune configs must be launched with "
            "python -m b2d_rlinfra.finetuning.node_agent or tools/slurm/rl_finetune_sbatch.sh"
        )
    if init_from_checkpoint is not None and not Path(init_from_checkpoint).is_file():
        raise ValueError(f"--init-from-checkpoint must point to a checkpoint file: {init_from_checkpoint}")
    rl_cfg = raw_config.get("rl_finetune", {}) or {}
    training_cfg = raw_config.get("training", {}) or {}
    verbose = int(rl_cfg.get("verbose", training_cfg.get("verbose", config.training.verbose)))
    log_dir = str(rl_cfg.get("log_dir", training_cfg.get("log_dir", config.training.log_dir)))
    device = str(rl_cfg.get("device", training_cfg.get("device", config.training.device)))
    _configure_logging(verbose)
    algo_config = rl_ppo_config(raw_config)
    if str(algo_config.name).lower() != "ppo":
        raise ValueError("rl_finetune v1 only supports algorithm.name=ppo")

    run_dir = _create_run_dir(log_dir, prefix="rl_finetune")
    _save_run_config(config, run_dir)

    device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
    if device == "auto":
        device = "cpu"
    device_obj = torch.device(device)

    logger.info("=" * 70)
    logger.info("RL Finetune PPO")
    logger.info("Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Config: %s", config_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Device: %s", device_obj)
    logger.info("Collectors: %s", rl_cfg.get("num_collectors", 1))
    logger.info("=" * 70)

    coordinator = Coordinator(
        config=config,
        config_path=config_path,
        run_dir=str(run_dir),
        device=device_obj,
        init_from_checkpoint=init_from_checkpoint,
    )
    coordinator.run(max_updates=max_updates)
    logger.info("RL finetune complete. Run dir: %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RL finetune PPO runner")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max-updates", type=int, default=None, help="Override rl_finetune.max_updates")
    parser.add_argument("--init-from-checkpoint", type=str, default=None, help="Initialize a new run from an RL checkpoint file")
    args = parser.parse_args()
    train(args.config, max_updates=args.max_updates, init_from_checkpoint=args.init_from_checkpoint)


if __name__ == "__main__":
    main()
