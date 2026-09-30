"""On-policy algorithm base class.

Drives async rollout collection via ``StandardEnvAdapter`` and feeds the
results into ``PerWorkerRolloutBuffer``. Subclasses (PPO, A2C) override
``train()``.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING

import numpy as np
import torch
from gymnasium import spaces

from .base_algorithm import BaseAlgorithm, EPISODE_LOG_WINDOW
from ..policies.actor_critic_policy_v2 import ActorCriticPolicyV2
from ..buffers.per_worker_buffer import PerWorkerRolloutBuffer
from ..utils.callbacks import BaseCallback
from ..utils.obs_utils import ObsUtils
from ..utils.success_trajectory_recorder import SuccessTrajectoryRecorder

if TYPE_CHECKING:
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer

# Use a dedicated logger name to avoid clashing with the envs layer.
logger = logging.getLogger("Training Loop")

__layer__ = (4, "Algorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """Base class for on-policy algorithms (PPO, A2C, etc.).

    Uses ``PerWorkerRolloutBuffer`` to collect trajectories per worker through
    the standard async adapter interface.

    Key methods:
        - ``learn()``: main training loop
        - ``collect_rollouts()``: collect trajectories using the current policy
        - ``train()``: update policy using collected data (override in subclass)

    Truncated vs terminated transitions:
        - ``terminated``: real episode end; the obs is the terminal observation
          and the worker resets.
        - ``truncated``: reached ``max_episode_steps``; the worker also resets,
          but the reward is bootstrapped with ``gamma * V(terminal_obs)``.

    Algorithm-internal slicing (``truncation_steps``):
        - When set, the buffer inserts artificial episode boundaries every
          ``truncation_steps`` steps before GAE, bootstrapping reward at each
          boundary. This is purely algorithm-side and does not affect the env.
    """
    
    rollout_buffer: PerWorkerRolloutBuffer
    policy: ActorCriticPolicyV2
    
    def __init__(
        self,
        policy: Type[ActorCriticPolicyV2],
        env,
        learning_rate: Union[float, callable],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        min_ready: int = 1,
        truncation_steps: Optional[int] = None,
        visualizer: Optional["TrainingVisualizer"] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = 'auto',
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        success_trajectory_config: Optional[Dict[str, Any]] = None,
        success_trajectory_log_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        _init_setup_model: bool = True,
    ):
        """
        Initialize on-policy algorithm.
        
        Args:
            policy: Policy class.
            env: Environment adapter (standard async interface).
            learning_rate: Learning rate or schedule.
            n_steps: Number of steps per rollout (total across all workers).
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            ent_coef: Entropy coefficient.
            vf_coef: Value function coefficient.
            max_grad_norm: Maximum gradient norm.
            min_ready: For Standard mode, minimum workers to wait for per step.
            truncation_steps: Algorithm-internal episode slicing step count (None = disabled).
            visualizer: TrainingVisualizer for recording training episodes (optional).
            policy_kwargs: Additional policy arguments.
            tensorboard_log: TensorBoard log directory.
            verbose: Verbosity level.
            device: Device for training.
            seed: Random seed.
            config: Full configuration for checkpoint saving.
            success_trajectory_config: Optional successful trajectory export config.
            success_trajectory_log_dir: Current run directory for default exports.
            config_path: Source config path to write into trajectory index metadata.
            checkpoint_path: Source checkpoint path to write into trajectory index metadata.
            _init_setup_model: Whether to setup model.
        """
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            config=config,
            _init_setup_model=False,
        )
        
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.min_ready = min_ready
        self.truncation_steps = truncation_steps
        self.l5_visualizer = visualizer
        
        # Rollout buffer
        self.rollout_buffer: Optional[PerWorkerRolloutBuffer] = None
        
        # For tracking episodes
        self._last_obs: Union[Dict[int, np.ndarray], List[np.ndarray], None] = None
        self._last_episode_starts: Union[Dict[int, bool], List[bool], None] = None
        
        # For standard mode: track pending actions (sent but not yet returned)
        # These persist across rollouts to handle async results
        self._pending_obs: Dict[int, np.ndarray] = {}
        self._pending_actions: Dict[int, Any] = {}
        self._pending_values: Dict[int, float] = {}
        self._pending_log_probs: Dict[int, float] = {}
        self._pending_action_dist_params: Dict[int, Optional[np.ndarray]] = {}
        self._pending_episode_starts: Dict[int, bool] = {}
        
        # Episode statistics tracking (for smoothed logging)
        self._ep_reward_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._ep_length_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        # Per-worker episode counters (for visualizer continuity)
        self._worker_episode_counts: Dict[int, int] = {}
        self._success_trajectory_recorder = self._build_success_trajectory_recorder(
            success_trajectory_config or {},
            success_trajectory_log_dir,
            config_path,
            checkpoint_path,
        )
        
        if _init_setup_model:
            self._setup_model()

    def _build_success_trajectory_recorder(
        self,
        recorder_config: Dict[str, Any],
        log_dir: Optional[Union[str, Path]],
        config_path: Optional[str],
        checkpoint_path: Optional[str],
    ) -> Optional[SuccessTrajectoryRecorder]:
        if not bool(recorder_config.get("enabled", False)):
            return None

        raw_output_dir = recorder_config.get("output_dir")
        if raw_output_dir:
            output_dir = Path(raw_output_dir).expanduser()
            if not output_dir.is_absolute() and log_dir is not None:
                output_dir = Path(log_dir).expanduser() / output_dir
        elif log_dir is not None:
            output_dir = Path(log_dir).expanduser() / "success_trajectories"
        else:
            output_dir = Path("success_trajectories")

        return SuccessTrajectoryRecorder(
            output_dir=output_dir,
            gamma=self.gamma,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            compress=bool(recorder_config.get("compress", True)),
            strict_success_threshold=float(
                recorder_config.get("strict_success_threshold", 99.9)
            ),
            max_success_trajectories=int(
                recorder_config.get("max_success_trajectories", 100)
            ),
        )

    def _is_episode_start(self, info: Dict[str, Any]) -> bool:
        """Whether next step should be marked as episode start for GAE.
        
        Now both terminated and truncated trigger async reset, so episode starts
        are detected solely from info flags set during reset (from_reset/episode_start).
        """
        return info.get('episode_start', False) or info.get('from_reset', False)

    def _is_visualizer_start(self, info: Dict[str, Any]) -> bool:
        """Whether to trigger visualizer episode start (reset only)."""
        return self._is_episode_start(info)

    def _apply_truncated_bootstrap(
        self,
        reward: float,
        truncated: bool,
        terminal_obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
    ) -> float:
        """Apply value bootstrap for truncated transitions.

        For truncations the env still resets, but the reward must be
        bootstrapped with the terminal-observation value before being added
        to the rollout buffer.
        """
        if not truncated or terminal_obs is None:
            return reward
        with torch.no_grad():
            terminal_obs_tensor = ObsUtils.obs_to_tensor(
                terminal_obs,
                device=self.device,
                observation_space=self.observation_space,
            )
            if isinstance(terminal_obs_tensor, dict):
                terminal_obs_tensor = {
                    key: value.unsqueeze(0) for key, value in terminal_obs_tensor.items()
                }
            else:
                terminal_obs_tensor = terminal_obs_tensor.unsqueeze(0)
            terminal_value = self.policy.predict_values(terminal_obs_tensor)
            return reward + self.gamma * terminal_value.cpu().numpy().flatten()[0]
        
    def _record_episode_end(
        self,
        worker_id: int,
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record per-episode statistics for terminated/truncated transitions."""
        if not (terminated or truncated):
            return

        self._episode_num += 1
        ep_reward = info.get('episode_reward', info.get('total_reward', 0))
        ep_length = info.get('episode_length', info.get('total_steps', 0))
        self._record_episode_metrics(info, ep_reward, ep_length)
        self._worker_episode_counts[worker_id] = self._worker_episode_counts.get(worker_id, 0) + 1
    
    def _setup_model(self) -> None:
        """Setup policy and rollout buffer."""
        # Create learning rate schedule
        lr_schedule = self.get_lr_schedule_fn()
        
        # Create policy
        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lr_schedule,
            **self.policy_kwargs,
        )
        self.policy.to(self.device)

        # Create per-worker rollout buffer
        self.rollout_buffer = PerWorkerRolloutBuffer(
            max_steps=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_workers=self.n_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            truncation_steps=self.truncation_steps,
        )
    
    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: PerWorkerRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy.
        
        Uses the standard async adapter interface.
        
        Args:
            env: Environment adapter.
            callback: Callback.
            rollout_buffer: Per-worker buffer to store rollouts.
            n_rollout_steps: Total steps to collect across all workers.
            
        Returns:
            True if training should continue, False if stopped by callback.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        
        # Switch to eval mode
        self.policy.set_training_mode(False)
        
        rollout_buffer.reset()
        callback.on_rollout_start()
        
        return self._collect_rollouts_standard(env, callback, rollout_buffer, n_rollout_steps)
    
    def _collect_rollouts_standard(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: PerWorkerRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect rollouts using Standard adapter (dict-based interface).
        
        IMPORTANT: Due to async nature, we need to track pending actions for each worker.
        When results return, we match them with their corresponding actions/values.
        The pending state is stored in instance variables to persist across rollouts.
        """
        obs_dict = self._last_obs  # Dict of observations
        episode_starts = self._last_episode_starts  # Dict of bools
        obs_keys = ObsUtils.get_obs_keys(self.observation_space)
        
        # Use instance variables for pending state tracking
        # (persists across function calls to handle async results that span rollouts)
        def _process_standard_step_outputs(
            new_obs_dict: Dict[int, Any],
            rewards: Dict[int, float],
            terminateds: Dict[int, bool],
            truncateds: Dict[int, bool],
            infos: Dict[int, Dict[str, Any]],
        ) -> Tuple[Dict[int, Any], Dict[int, bool]]:
            if self.l5_visualizer is not None:
                vis_actions = {wid: self._pending_actions.get(wid) for wid in rewards.keys()}
                vis_obs = {}
                for wid in rewards.keys():
                    info = infos.get(wid, {})
                    obs_item = new_obs_dict.get(wid)
                    bev_image = info.get("bev_image")
                    if bev_image is not None:
                        if isinstance(obs_item, dict):
                            vis_obs[wid] = dict(obs_item)
                            vis_obs[wid]["bev_image"] = bev_image
                        else:
                            vis_obs[wid] = {"bev_image": bev_image}
                    else:
                        vis_obs[wid] = obs_item

                self.l5_visualizer.process_step_result(
                    actions=vis_actions,
                    obs=vis_obs,
                    rewards=rewards,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    infos=infos,
                )

            for wid in rewards.keys():
                info = infos.get(wid, {})

                if info.get("crashed", False):
                    discarded = rollout_buffer.discard_episode(wid)
                    if self._success_trajectory_recorder is not None:
                        self._success_trajectory_recorder.discard(wid)
                    has_pending = wid in self._pending_actions
                    context = self._format_episode_context(info)
                    context_suffix = f" | {context}" if context else ""
                    logger.warning(
                        f"Worker {wid}{context_suffix} crashed ({info.get('crash_type', 'unknown')}), "
                        f"discarded {discarded} steps (last episode), had_pending_action={has_pending}"
                    )
                    self._episode_num += 1
                    self._worker_episode_counts[wid] = 0
                    self._pending_obs.pop(wid, None)
                    self._pending_actions.pop(wid, None)
                    self._pending_values.pop(wid, None)
                    self._pending_log_probs.pop(wid, None)
                    self._pending_action_dist_params.pop(wid, None)
                    self._pending_episode_starts.pop(wid, None)
                    continue

                if wid not in self._pending_actions:
                    logger.warning(f"Worker {wid} returned result but has no pending action, skipping")
                    continue

                reward_to_use = self._apply_truncated_bootstrap(
                    rewards[wid],
                    truncateds.get(wid, False),
                    new_obs_dict.get(wid),
                )

                rollout_buffer.add(
                    worker_id=wid,
                    obs=self._pending_obs[wid],
                    action=self._pending_actions[wid],
                    reward=reward_to_use,
                    done=self._pending_episode_starts[wid],
                    value=self._pending_values[wid],
                    log_prob=self._pending_log_probs[wid],
                    action_dist_params=self._pending_action_dist_params.get(wid),
                )

                self.num_timesteps += 1

                is_terminated = bool(terminateds.get(wid, False))
                is_truncated = bool(truncateds.get(wid, False))
                if self._success_trajectory_recorder is not None:
                    self._success_trajectory_recorder.record(
                        worker_id=wid,
                        obs=self._pending_obs[wid],
                        action=self._pending_actions[wid],
                        reward=rewards[wid],
                        terminated=is_terminated,
                        truncated=is_truncated,
                        info=info,
                    )
                if is_terminated or is_truncated:
                    self._record_episode_end(
                        worker_id=wid,
                        info=info,
                        terminated=is_terminated,
                        truncated=is_truncated,
                    )
                    if self._success_trajectory_recorder is not None:
                        self._success_trajectory_recorder.flush_if_success(
                            worker_id=wid,
                            terminal_info=info,
                            episode_num=self._episode_num,
                        )

                del self._pending_obs[wid]
                del self._pending_actions[wid]
                del self._pending_values[wid]
                del self._pending_log_probs[wid]
                self._pending_action_dist_params.pop(wid, None)
                del self._pending_episode_starts[wid]

            next_obs_dict = dict(new_obs_dict)
            for wid in list(next_obs_dict.keys()):
                if terminateds.get(wid, False) or truncateds.get(wid, False):
                    del next_obs_dict[wid]

            next_episode_starts = {}
            vis_episode_starts = {}
            for wid in next_obs_dict:
                info = infos.get(wid, {})
                next_episode_starts[wid] = self._is_episode_start(info)
                is_vis_start = self._is_visualizer_start(info)
                vis_episode_starts[wid] = is_vis_start
                if is_vis_start:
                    logger.debug(
                        f"(standard) Worker {wid}: vis_episode_start detected! "
                        f"terminated={terminateds.get(wid, False)}, truncated={truncateds.get(wid, False)}, "
                        f"from_reset={info.get('from_reset', False)}, episode_start={info.get('episode_start', False)}, "
                        f"episode_count={self._worker_episode_counts.get(wid, 0)}"
                    )

            if self.l5_visualizer is not None:
                for wid, is_new_episode in vis_episode_starts.items():
                    if is_new_episode:
                        self.l5_visualizer.on_episode_start(
                            wid,
                            episode_id=self._worker_episode_counts.get(wid, 0)
                        )

            return next_obs_dict, next_episode_starts
        
        while rollout_buffer.total_steps < n_rollout_steps:
            if not obs_dict:
                new_obs_dict, rewards, terminateds, truncateds, infos = env.step(
                    {}, min_ready=self.min_ready
                )
                if rewards:
                    next_obs_dict, next_episode_starts = _process_standard_step_outputs(
                        new_obs_dict, rewards, terminateds, truncateds, infos
                    )
                    callback.update_locals(locals())
                    if not callback.on_step():
                        self._last_obs = next_obs_dict
                        self._last_episode_starts = next_episode_starts
                        return False
                    obs_dict = next_obs_dict
                    episode_starts = next_episode_starts
                    if obs_dict:
                        continue
                else:
                    obs_dict = new_obs_dict
                    episode_starts = {}
                    if self.l5_visualizer is not None:
                        for wid in obs_dict:
                            info = infos.get(wid, {})
                            episode_starts[wid] = self._is_episode_start(info)
                            is_vis_start = self._is_visualizer_start(info)
                            if is_vis_start:
                                self.l5_visualizer.on_episode_start(
                                    wid,
                                    episode_id=self._worker_episode_counts.get(wid, 0)
                                )
                            else:
                                logger.debug(
                                    f"standard_step_wait got obs without reset flags: "
                                    f"worker={wid}, info_keys={list(info.keys())}"
                                )
                    else:
                        for wid in obs_dict:
                            info = infos.get(wid, {})
                            episode_starts[wid] = self._is_episode_start(info)
                continue
            
            # Stack observations for ready workers (those with new obs to process)
            ready_workers = list(obs_dict.keys())
            stacked_obs = ObsUtils.stack_obs(
                [obs_dict[wid] for wid in ready_workers],
                keys=obs_keys,
                observation_space=self.observation_space,
            )
            
            with torch.no_grad():
                obs_tensor = ObsUtils.obs_to_tensor(
                    stacked_obs,
                    device=self.device,
                    keys=obs_keys,
                    observation_space=self.observation_space,
                )
                forward_with_dist_params = getattr(
                    self.policy,
                    'forward_with_action_dist_params',
                    None,
                )
                if callable(forward_with_dist_params):
                    actions_t, values_t, log_probs_t, action_dist_params_t = (
                        forward_with_dist_params(obs_tensor)
                    )
                else:
                    actions_t, values_t, log_probs_t = self.policy(obs_tensor)
                    action_dist_params_t = None

            actions_np = actions_t.cpu().numpy()
            values_np = values_t.cpu().numpy().flatten()
            log_probs_np = log_probs_t.cpu().numpy().flatten()
            action_dist_params_np = (
                action_dist_params_t.cpu().numpy()
                if action_dist_params_t is not None
                else None
            )
            
            # Build actions dict and record pending state
            actions = {}
            for idx, wid in enumerate(ready_workers):
                if isinstance(self.action_space, spaces.Discrete):
                    action = int(actions_np[idx])
                else:
                    action = actions_np[idx]
                
                actions[wid] = action

                # Record pending state for this worker
                self._pending_obs[wid] = obs_dict[wid]
                self._pending_actions[wid] = action
                self._pending_values[wid] = values_np[idx]
                self._pending_log_probs[wid] = log_probs_np[idx]
                self._pending_action_dist_params[wid] = (
                    action_dist_params_np[idx]
                    if action_dist_params_np is not None
                    else None
                )
                self._pending_episode_starts[wid] = episode_starts.get(wid, False)
            
            # Step environment - results may come from ANY worker with pending action
            new_obs_dict, rewards, terminateds, truncateds, infos = env.step(
                actions, min_ready=self.min_ready
            )

            next_obs_dict, next_episode_starts = _process_standard_step_outputs(
                new_obs_dict, rewards, terminateds, truncateds, infos
            )

            callback.update_locals(locals())
            if not callback.on_step():
                self._last_obs = next_obs_dict
                self._last_episode_starts = next_episode_starts
                return False

            obs_dict = next_obs_dict
            episode_starts = next_episode_starts
        
        # Compute last values for GAE
        last_values = {}
        last_dones = {}
        
        if obs_dict:
            ready_workers = list(obs_dict.keys())
            stacked_obs = ObsUtils.stack_obs(
                [obs_dict[wid] for wid in ready_workers],
                keys=obs_keys,
                observation_space=self.observation_space,
            )
            
            with torch.no_grad():
                obs_tensor = ObsUtils.obs_to_tensor(
                    stacked_obs,
                    device=self.device,
                    keys=obs_keys,
                    observation_space=self.observation_space,
                )
                values_t = self.policy.predict_values(obs_tensor)
                values_np = values_t.cpu().numpy().flatten()
            
            for idx, wid in enumerate(ready_workers):
                last_values[wid] = values_np[idx]
                last_dones[wid] = episode_starts.get(wid, False)
        
        # Use pending action state to bootstrap async workers that have not
        # returned by the rollout boundary.
        for wid in range(self.n_envs):
            if wid not in last_values:
                if wid in self._pending_actions:
                    last_values[wid] = float(self._pending_values[wid])
                    last_dones[wid] = bool(self._pending_episode_starts[wid])
                else:
                    last_values[wid] = 0.0
                    last_dones[wid] = True
        
        # Compute returns and advantages
        rollout_buffer.compute_returns_and_advantage(last_values, last_dones)
        
        # Update state
        self._last_obs = obs_dict
        self._last_episode_starts = episode_starts
        
        callback.on_rollout_end()
        
        return True
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
    ) -> 'OnPolicyAlgorithm':
        """
        Train the algorithm.

        Args:
            total_timesteps: Total timesteps to train.
            callback: Callbacks.
            log_interval: Logging frequency (in iterations).
            reset_num_timesteps: Whether to reset timestep counter.

        Returns:
            self
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
        )
        
        callback.on_training_start(locals(), globals())
        
        assert self.env is not None, "Environment must be set before training"
        
        # Initialize observations with the standard async adapter.
        obs_dict, reset_infos = self.env.reset(min_ready=self.n_envs)
        self._last_obs = obs_dict
        self._last_episode_starts = {wid: True for wid in obs_dict}
        self._worker_episode_counts = {i: 0 for i in range(self.n_envs)}
        self._pending_obs.clear()
        self._pending_actions.clear()
        self._pending_values.clear()
        self._pending_log_probs.clear()
        self._pending_action_dist_params.clear()
        self._pending_episode_starts.clear()
        if self._success_trajectory_recorder is not None:
            self._success_trajectory_recorder.reset()
        if self.l5_visualizer is not None:
            for wid in obs_dict:
                self.l5_visualizer.on_episode_start(wid, episode_id=self._worker_episode_counts.get(wid, 0))
        self._reset_rollout_training_timer()
        
        self._iteration = 0
        
        try:
            while self.num_timesteps < total_timesteps:
                # Collect rollouts
                continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    self.rollout_buffer,
                    n_rollout_steps=self.n_steps,
                )
                
                if not continue_training:
                    break
                
                self._iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                
                # Train on collected data
                self.train()
                callback.update_locals(locals())
                callback.on_update_end()
                
                # Logging (after train to include training stats)
                if log_interval is not None and self._iteration % log_interval == 0:
                    self._dump_logs(total_timesteps)
        
        finally:
            pass
        
        callback.on_training_end()
        
        return self
    
    def _dump_logs(self, total_timesteps: int) -> None:
        """
        Write training logs to console and logger.
        
        Consolidates all logging in one place for cleaner output.
        
        Args:
            total_timesteps: Total timesteps target for progress calculation.
        """
        time_elapsed = self._get_elapsed_sec()
        sim_real_ratio = self._get_sim_real_ratio(time_elapsed)
        # Calculate episode statistics
        ep_rew_mean = float(np.mean(self._ep_reward_buffer)) if self._ep_reward_buffer else None
        ep_len_mean = float(np.mean(self._ep_length_buffer)) if self._ep_length_buffer else None

        logger.info(
            self._format_progress_line(
                total_timesteps=total_timesteps,
                sim_real_ratio=sim_real_ratio,
                reward_mean=ep_rew_mean,
                length_mean=ep_len_mean,
                extra={"rollout_steps": self.rollout_buffer.total_steps},
            )
        )
        
        # Record to TensorBoard logger
        if self.logger is not None:
            # Time metrics
            self.logger.record('time/iterations', self._iteration)
            self.logger.record('time/episodes', self._episode_num)
            self.logger.record('time/elapsed_sec', float(time_elapsed))
            self.logger.record('time/sim_real_ratio', float(sim_real_ratio))
            
            # Rollout metrics
            self.logger.record('rollout/collected_steps', self.rollout_buffer.total_steps)
            self._record_episode_tensorboard_metrics()
            
            # Dump all recorded values
            self.logger.dump(step=self.num_timesteps)
    
    def train(self) -> None:
        """
        Update policy using collected rollouts.
        
        Must be implemented by subclasses (PPO, A2C, etc.).
        """
        raise NotImplementedError
