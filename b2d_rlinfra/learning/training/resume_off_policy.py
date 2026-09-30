#!/usr/bin/env python3
"""Resume SAC / TD3 training from a checkpoint.

Loads actor and critic branches selectively while the new YAML controls the
fresh run configuration.
"""

import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

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


def _configure_logging(verbose: int = 1) -> None:
    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.INFO)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if v in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def _filter_state_dict(state: Dict[str, torch.Tensor], prefixes) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if any(k.startswith(p) for p in prefixes)}


def _load_selective_weights(
    policy: torch.nn.Module,
    ckpt_state: Dict[str, torch.Tensor],
    load_actor: bool,
    load_critic: bool,
) -> None:
    """Selectively load checkpoint weights into a freshly built policy.

    Conventions:
      * ``actor.*`` / ``critic.*`` / ``critic_target.*`` belong to their
        respective branches.
      * ``features_extractor.*`` is loaded whenever actor or critic is
        loaded - it is the shared CNN backbone, so leaving it randomly
        initialised while loading one branch would shift the input
        distribution of that branch out of its trained regime.
    """
    to_load: Dict[str, torch.Tensor] = {}
    if load_actor:
        to_load.update(_filter_state_dict(ckpt_state, ('actor.',)))
    if load_critic:
        to_load.update(_filter_state_dict(
            ckpt_state, ('critic.', 'critic_target.')
        ))
    if load_actor or load_critic:
        to_load.update(_filter_state_dict(ckpt_state, ('features_extractor.',)))

    if not to_load:
        logger.warning("No checkpoint weights loaded (load_actor=%s, load_critic=%s)",
                       load_actor, load_critic)
        return

    missing_keys, unexpected_keys = policy.load_state_dict(to_load, strict=False)
    loaded_keys = set(to_load.keys()) - set(unexpected_keys)
    logger.info(
        "[Resume] Loaded %d / %d checkpoint tensors (missing=%d, unexpected=%d)",
        len(loaded_keys), len(to_load), len(missing_keys), len(unexpected_keys),
    )
    fe_hits = sum(1 for k in loaded_keys if k.startswith('features_extractor.'))
    actor_hits = sum(1 for k in loaded_keys if k.startswith('actor.'))
    critic_hits = sum(1 for k in loaded_keys if k.startswith(('critic.', 'critic_target.')))
    logger.info("[Resume]  - features_extractor loaded: %d", fe_hits)
    logger.info("[Resume]  - actor branch loaded:       %d", actor_hits)
    logger.info("[Resume]  - critic branch loaded:      %d", critic_hits)
    if unexpected_keys:
        logger.warning("[Resume] Unmatched keys (first 5): %s", list(unexpected_keys)[:5])


def _build_warmup_policy_from_ckpt(
    base_policy: torch.nn.Module,
    ckpt_state: Dict[str, torch.Tensor],
) -> torch.nn.Module:
    """Deep-copy ``base_policy`` and overwrite it with checkpoint weights.

    The result is a frozen, eval-only network used during warmup when
    ``warmup_source='actor'`` to collect initial data with the checkpoint
    actor regardless of whether ``model.policy.actor`` is being re-trained.
    """
    warmup_policy = copy.deepcopy(base_policy)
    missing, unexpected = warmup_policy.load_state_dict(ckpt_state, strict=False)
    fe_n = sum(1 for k in ckpt_state if k.startswith('features_extractor.'))
    ac_n = sum(1 for k in ckpt_state if k.startswith('actor.'))
    logger.info(
        "[Warmup-Actor] built standalone warmup policy: features_extractor=%d, actor=%d "
        "(missing=%d, unexpected=%d)",
        fe_n, ac_n, len(missing), len(unexpected),
    )
    warmup_policy.eval()
    for p in warmup_policy.parameters():
        p.requires_grad = False
    return warmup_policy


def resume_train(
    config_path: str,
    checkpoint: str,
    load_actor: bool = True,
    load_critic: bool = True,
    reset_timesteps: bool = True,
    resume_log_dir: str = None,
) -> None:
    config = load_config(config_path)
    _configure_logging(config.training.verbose)

    algo = config.algorithm
    training = config.training
    env_config = config.env_config
    if env_config is None:
        raise ValueError("env config not loaded - check YAML")
    algo_name_lower = algo.name.lower()
    if algo_name_lower not in ('sac', 'td3'):
        raise ValueError(
            "resume_off_policy.py only supports SAC / TD3, got "
            f"config.algorithm.name={algo.name!r}"
        )
    is_sac = algo_name_lower == 'sac'

    sample_mode = (env_config.get('routes') or {}).get('sample_mode', 'sequential')
    adaptive_cfg = (env_config.get('routes') or {}).get('adaptive') or {}

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {ckpt_path}")
    for required in ('policy.pth', 'metadata.json'):
        if not (ckpt_path / required).exists():
            raise ValueError(f"Checkpoint missing {required}: {ckpt_path}")

    logger.info("=" * 70)
    logger.info("  %s Resume Training (selective load)", algo.name.upper())
    logger.info("  Checkpoint     : %s", checkpoint)
    logger.info("  load_actor     : %s", load_actor)
    logger.info("  load_critic    : %s", load_critic)
    logger.info("  reset_timesteps: %s", reset_timesteps)
    logger.info("  warmup_source  : %s", getattr(algo, 'warmup_source', 'random'))
    logger.info("  sample_mode    : %s", sample_mode)
    if sample_mode == 'adaptive':
        logger.info(
            "  adaptive       : window=%s success_threshold=%s "
            "low_sr=%s high_sr=%s low_w=%s high_w=%s warmup_min=%s log_interval=%s",
            adaptive_cfg.get('window_size', 8),
            adaptive_cfg.get('success_threshold', 100.0),
            adaptive_cfg.get('low_success_rate', 0.5),
            adaptive_cfg.get('high_success_rate', 0.8),
            adaptive_cfg.get('low_success_weight', 2.0),
            adaptive_cfg.get('high_success_weight', 0.5),
            adaptive_cfg.get('warmup_min_samples', 4),
            adaptive_cfg.get('log_interval_updates', 100),
        )
    logger.info("=" * 70)
    logger.info("Started at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    cleanup = CleanupManager()
    install_signal_handlers(cleanup)

    try:
        # Resolve run directory.
        if resume_log_dir:
            log_dir = Path(resume_log_dir).expanduser().resolve()
        else:
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir = Path(training.log_dir) / f"resume_{ts}"
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        cleanup.register(log_dir=str(log_dir))
        save_run_config(config, log_dir)
        logger.info("[1] Config loaded: %s", config_path)
        logger.info("    total_timesteps: %s  LR: %s  gamma: %s",
                    f"{algo.total_timesteps:,}", algo.learning_rate, algo.gamma)
        if is_sac:
            logger.info("    ent_coef=%s  target_entropy=%s  buffer_size=%s",
                        algo.ent_coef, algo.target_entropy, algo.buffer_size)
        else:
            logger.info("    buffer_size=%s  policy_delay=%s  target_policy_noise=%s",
                        algo.buffer_size,
                        getattr(algo, 'policy_delay', None),
                        getattr(algo, 'target_policy_noise', None))

        # Read checkpoint metadata (used for logging and optional timestep resume).
        with open(ckpt_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        ckpt_timesteps = int(metadata.get('num_timesteps', 0))
        ckpt_episodes = int(metadata.get('episode_num', 0))
        logger.info("    Checkpoint timesteps: %s  episodes: %s",
                    f"{ckpt_timesteps:,}", f"{ckpt_episodes:,}")
        logger.info("[2] Run dir: %s", log_dir)

        device = 'cuda' if (training.device == 'auto' and torch.cuda.is_available()) else training.device
        if device == 'auto':
            device = 'cpu'
        logger.info("[3] Device: %s", device)

        l2_obs_space = build_observation_space(env_config)
        l2_act_space = build_action_space(env_config)
        logger.info("[4] obs_space=%s  act_space=%s", l2_obs_space, l2_act_space)

        _fe_cls, _fe_kw, obs_key, policy_kwargs = get_feature_extractor(config)
        algorithm_config = config.to_dict().get('algorithm', {})

        l3_hosts, l3_ports, server_wait_timeout = get_carla_hosts_ports(env_config)
        cleanup.register(carla_hosts=l3_hosts)

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

        initial_step = 0 if reset_timesteps else ckpt_timesteps
        initial_episode = 0 if reset_timesteps else ckpt_episodes
        l5_visualizer = create_visualizer(
            config, log_dir,
            initial_step=initial_step,
            initial_episode=initial_episode,
        )
        cleanup.register(visualizer=l5_visualizer)

        # Build a fresh off-policy learner driven by the new YAML.
        logger.info("[6] Building fresh %s with new config ...", algo.name.upper())
        AlgoClass, algo_kwargs = get_algorithm_components(
            config, l4_adapter, obs_key, policy_kwargs,
            visualizer=l5_visualizer, log_dir=log_dir,
        )
        l4_learner = AlgoClass(device=device, seed=training.seed, **algo_kwargs)
        cleanup.register(model=l4_learner)
        param_count = sum(p.numel() for p in l4_learner.policy.parameters())
        logger.info("    Params: %s  FE: %s", f"{param_count:,}", _fe_cls.__name__)

        # Selectively load checkpoint weights.
        logger.info("[7] Loading checkpoint weights from %s", ckpt_path)
        ckpt_state = torch.load(
            ckpt_path / 'policy.pth',
            map_location=device,
            weights_only=False,
        )
        _load_selective_weights(
            l4_learner.policy, ckpt_state,
            load_actor=load_actor, load_critic=load_critic,
        )

        # If critic was loaded but the checkpoint has no critic_target,
        # synchronise the target network from the current critic.
        ckpt_keys = set(ckpt_state.keys())
        has_critic_target = any(k.startswith('critic_target.') for k in ckpt_keys)
        if load_critic and not has_critic_target:
            logger.info("    checkpoint has no critic_target.*; syncing target from critic")
            l4_learner.policy.critic_target.load_state_dict(l4_learner.policy.critic.state_dict())

        # When warmup_source='actor', mount a frozen checkpoint-based policy
        # for the warmup phase, regardless of whether the actor is being
        # re-trained in model.policy.
        warmup_source = str(getattr(algo, 'warmup_source', 'random')).lower()
        if warmup_source == 'actor':
            warmup_policy = _build_warmup_policy_from_ckpt(l4_learner.policy, ckpt_state)
            warmup_policy.to(device)
            l4_learner._warmup_policy = warmup_policy
            logger.info(
                "    warmup_source='actor' -> mounted standalone warmup policy "
                "(load_actor=%s, model.policy is unaffected)", load_actor,
            )

        # SAC only: entropy coefficient is reset from the new config.
        if is_sac:
            logger.info(
                "[8] entropy coefficient reset (ent_coef=%s, target_entropy=%s)",
                l4_learner.ent_coef, l4_learner.target_entropy,
            )
        else:
            logger.info(
                "[8] %s has no entropy term, skipping entropy reset",
                algo.name.upper(),
            )

        if reset_timesteps:
            l4_learner.num_timesteps = 0
            l4_learner._episode_num = 0
            l4_learner._reset_episode_metric_buffers()
            logger.info("    num_timesteps / episodes reset to 0 (fresh warmup)")
        else:
            l4_learner.num_timesteps = ckpt_timesteps
            l4_learner._episode_num = ckpt_episodes
            l4_learner._scenario_episode_counts = {
                str(name): int(count)
                for name, count in (metadata.get('scenario_episode_counts') or {}).items()
            }
            logger.info("    continuing from checkpoint counters: timesteps=%s episodes=%s",
                        f"{ckpt_timesteps:,}", f"{ckpt_episodes:,}")

        callbacks = [
            CheckpointCallback(
                save_freq=training.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{algo.name}_model",
                verbose=1,
            ),
        ]
        callback_list = CallbackList(callbacks)

        logger.info("[9] Resuming training  total_timesteps=%s", f"{algo.total_timesteps:,}")
        logger.info("=" * 70)

        l4_learner.learn(
            total_timesteps=algo.total_timesteps,
            callback=callback_list,
            log_interval=training.log_interval,
            reset_num_timesteps=reset_timesteps,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Off-policy (SAC/TD3) resume training with selective actor/critic loading")
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the YAML config that drives the resume run')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint directory (must contain policy.pth + metadata.json)')
    parser.add_argument('--load-actor', type=_str2bool, default=True,
                        help='Load actor weights (and shared features_extractor) from the checkpoint')
    parser.add_argument('--load-critic', type=_str2bool, default=True,
                        help='Load critic / critic_target weights from the checkpoint')
    parser.add_argument('--reset-timesteps', type=_str2bool, default=True,
                        help='Reset num_timesteps / episode counters (defaults to true, triggers a fresh warmup)')
    parser.add_argument('--resume-log-dir', type=str, default=None,
                        help='Override the output dir (default: <config.log_dir>/resume_<timestamp>)')
    # Adaptive route sampling parameters live in the YAML
    # (config.env.routes.adaptive); they are updated online from episode
    # results and do not need CLI overrides.

    args = parser.parse_args()

    resume_train(
        config_path=args.config,
        checkpoint=args.checkpoint,
        load_actor=args.load_actor,
        load_critic=args.load_critic,
        reset_timesteps=args.reset_timesteps,
        resume_log_dir=args.resume_log_dir,
    )


if __name__ == '__main__':
    main()
