#!/usr/bin/env python3
"""On-policy (PPO / A2C) resume entry point.

Loads a PPO/A2C checkpoint and continues training via ``model.learn()``. The
algorithm is selected through ``config.algorithm.name``. The default resume log
dir is ``<config.training.log_dir>/resume_<timestamp>``.

See ``tools/launch/baseline/resume_on_policy.sh`` for the matching shell entry point.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

__layer__ = (4, "Algorithm")

from b2d_rlinfra.learning.training.runner_utils import (
    build_action_space,
    build_observation_space,
    CleanupManager,
    create_env_pool_and_adapter,
    create_visualizer,
    get_algorithm_components,
    get_carla_hosts_ports,
    get_feature_extractor,
    install_signal_handlers,
    save_run_config,
    wait_for_carla_servers_ready,
)
from b2d_rlinfra.learning.utils.config import load_config
from b2d_rlinfra.learning.utils.callbacks import CheckpointCallback, CallbackList

logger = logging.getLogger("Training Loop")


def _configure_logging(verbose=1):
    """Configure global logging level. Compatible with Python 3.7 (no force param)."""
    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.INFO)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def resume_train(
    config_path: str,
    checkpoint: str,
    resume_log_dir: str = None,
):
    config = load_config(config_path)
    _configure_logging(config.training.verbose)
    algo = config.algorithm
    training = config.training
    env_config = config.env_config
    if env_config is None:
        raise ValueError("env config not loaded - check YAML")

    algo_name = algo.name.upper()
    if algo.name.lower() not in ("ppo", "a2c"):
        raise ValueError(
            "resume_on_policy.py only supports PPO/A2C, "
            f"got config.algorithm.name={algo.name!r}"
        )

    # Validate checkpoint
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {ckpt_path}")
    for required in ("policy.pth", "metadata.json"):
        if not (ckpt_path / required).exists():
            raise ValueError(f"Checkpoint missing {required}: {ckpt_path}")

    logger.info("=" * 70)
    logger.info("  %s On-policy Resume Training", algo_name)
    logger.info("  Adapter: STANDARD")
    logger.info("  Checkpoint: %s", checkpoint)
    logger.info("=" * 70)
    logger.info("Started at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    cleanup = CleanupManager()
    install_signal_handlers(cleanup)

    try:
        # -- determine log dir --
        if resume_log_dir:
            log_dir = Path(resume_log_dir).expanduser().resolve()
        else:
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir = Path(training.log_dir) / f"resume_{ts}"

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        cleanup.register(log_dir=str(log_dir))

        # save config
        cfg_file = log_dir / "config.yaml"
        if not cfg_file.exists():
            save_run_config(config, log_dir)
        logger.info("[1] Config loaded: %s", config_path)
        logger.info("    Algorithm: %s  total_timesteps: %s", algo.name, f"{algo.total_timesteps:,}")
        logger.info("    LR: %s  gamma: %s", algo.learning_rate, algo.gamma)
        logger.info("    Checkpoint: %s", checkpoint)

        # -- read metadata --
        with open(ckpt_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        initial_timesteps = metadata.get('num_timesteps', 0)
        initial_episodes = metadata.get('episode_num', 0)
        logger.info("    Checkpoint timesteps: %s  episodes: %s", f"{initial_timesteps:,}", f"{initial_episodes:,}")
        logger.info("[2] Run dir: %s", log_dir)

        # -- device --
        device = 'cuda' if (training.device == 'auto' and torch.cuda.is_available()) else training.device
        if device == 'auto':
            device = 'cpu'
        logger.info("[3] Device: %s", device)

        # Build observation / action spaces from YAML.
        l2_obs_space = build_observation_space(env_config)
        l2_act_space = build_action_space(env_config)
        logger.info("[4] obs_space=%s  act_space=%s", l2_obs_space, l2_act_space)

        _fe_cls, _fe_kw, obs_key, policy_kwargs = get_feature_extractor(config)
        algorithm_config = config.to_dict().get('algorithm', {})

        # CARLA host/port pool.
        l3_hosts, l3_ports, server_wait_timeout = get_carla_hosts_ports(env_config)
        cleanup.register(carla_hosts=l3_hosts)

        # Spawn parallel CARLA worker pool and adapter.
        logger.info("[5] Creating CARLAEnvPool ...")
        l3_pool, l4_adapter = create_env_pool_and_adapter(
            env_config=env_config,
            observation_space=l2_obs_space,
            action_space=l2_act_space,
            adapter_timeout=config.adapter.timeout,
            min_ready=config.adapter.min_ready,
            obs_key=obs_key,
            algorithm_config=algorithm_config,
        )
        cleanup.register(pool=l3_pool, carla_hosts=getattr(l3_pool, 'carla_hosts', l3_hosts))

        l3_runtime_hosts = list(getattr(l3_pool, 'carla_hosts', l3_hosts))
        l3_runtime_ports = list(getattr(l3_pool, 'carla_ports', l3_ports))
        if l3_runtime_ports:
            if not wait_for_carla_servers_ready(l3_runtime_hosts, l3_runtime_ports, timeout=server_wait_timeout):
                raise RuntimeError("CARLA servers not ready")

        # Training visualizer.
        l5_visualizer = create_visualizer(
            config, log_dir,
            initial_step=initial_timesteps,
            initial_episode=initial_episodes,
        )
        cleanup.register(visualizer=l5_visualizer)

        # Load checkpoint as the learner.
        logger.info("[6] Loading %s model ...", algo_name)
        AlgoClass, algo_kwargs = get_algorithm_components(
            config, l4_adapter, obs_key, policy_kwargs,
            visualizer=l5_visualizer, log_dir=log_dir,
            config_path=config_path, checkpoint_path=checkpoint,
        )
        l4_learner = AlgoClass.load(path=checkpoint, device=device, **algo_kwargs)
        cleanup.register(model=l4_learner)
        logger.info("    Loaded  timesteps=%s", f"{l4_learner.num_timesteps:,}")

        param_count = sum(p.numel() for p in l4_learner.policy.parameters())
        logger.info("    Params: %s  FE: %s", f"{param_count:,}", _fe_cls.__name__)

        callbacks = [
            CheckpointCallback(
                save_freq=training.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{algo.name}_model",
                verbose=1,
                save_on='update_end',
            ),
        ]
        callback_list = CallbackList(callbacks)

        logger.info("[7] Resuming training  total_timesteps=%s", f"{algo.total_timesteps:,}")
        logger.info("=" * 70)

        l4_learner.learn(
            total_timesteps=algo.total_timesteps,
            callback=callback_list,
            log_interval=training.log_interval,
            reset_num_timesteps=False,
        )

        final_path = checkpoint_dir / "final_model"
        l4_learner.save(str(final_path))
        logger.info("  Final model saved: %s", final_path)

        logger.info("=" * 70)
        logger.info("  Resume training complete!  timesteps=%s", f"{l4_learner.num_timesteps:,}")
        logger.info("  Finished at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("=" * 70)

    except Exception as e:
        logger.error("ERROR: %s", e, exc_info=True)
        cleanup.cleanup(force_kill=True)
        raise

    finally:
        cleanup.cleanup(force_kill=True)


def main():
    parser = argparse.ArgumentParser(description="On-policy (PPO/A2C) resume training")
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to YAML config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory (must contain policy.pth + metadata.json)')
    parser.add_argument('--resume-log-dir', type=str, default=None,
                        help='Override log dir (default: <config.log_dir>/resume_<timestamp>)')
    args = parser.parse_args()

    resume_train(
        config_path=args.config,
        checkpoint=args.checkpoint,
        resume_log_dir=args.resume_log_dir,
    )


if __name__ == '__main__':
    main()
