"""Configuration loading utilities.

Parses ``configs/*.yaml`` into typed dataclasses (``AlgorithmConfig``,
``PolicyConfig``, ``TrainingConfig``, ``AdapterConfig``,
``VisualizationConfig``) consumed by the learners and runner scripts.
"""

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

__layer__ = (4, "Algorithm")


@dataclass
class AlgorithmConfig:
    """Algorithm configuration (supports PPO/A2C/TD3/SAC)."""
    name: str = 'ppo'
    total_timesteps: int = 100000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 3e-4
    learning_rate_final: Optional[float] = None
    
    # PPO specific
    n_steps: int = 1024
    n_epochs: int = 3
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    ent_coef_final: Optional[float] = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    truncation: Optional[int] = None

    # Off-policy (TD3/SAC) specific
    buffer_size: int = 100000
    learning_starts: int = 1000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    
    # TD3 specific
    policy_delay: int = 2
    target_policy_noise: Union[float, List[float]] = 0.2  # Can be float or list per dimension
    target_noise_clip: Union[float, List[float]] = 0.5     # Can be float or list per dimension
    exploration_noise: Union[float, List[float]] = 0.1     # Can be float or list per dimension
    
    # TD3+BC: Behavior Cloning auxiliary loss
    bc_enabled: bool = False                              # Enable BC auxiliary loss
    bc_lambda_initial: float = 1.0                        # Initial BC loss weight (lambda)
    bc_lambda_decay_steps: int = 200000                   # Steps to linearly decay lambda to final
    bc_lambda_final: float = 0.0                          # Final BC loss weight (residual regularization)
    bc_prior_action: List[float] = field(default_factory=lambda: [0.6, 0.0])  # Prior action [throttle, steering]

    # Exploration-noise annealing (Gaussian only). Decays the raw sigma by
    # a multiplicative factor from 1.0 to exploration_noise_final_scale over
    # exploration_noise_decay_steps post-warmup steps. 0 disables annealing.
    exploration_noise_decay_steps: int = 0
    exploration_noise_final_scale: float = 1.0
    
    # Prioritized Experience Replay (PER) — always enabled in the off-policy
    # stack (the train sub-process requires shared-memory PER); only the
    # hyper-parameters below are user-configurable.
    per_alpha: float = 0.6                                # Priority exponent
    per_beta: float = 0.4                                 # Initial IS weight exponent
    per_beta_annealing_steps: int = 100000                # Steps to anneal beta to 1.0
    per_min_priority: float = 1e-6                        # Minimum priority

    # Static (scenario-keyed) replay buffer for successful / high-completion
    # episodes.  Full layout is documented in
    # :meth:`OffPolicyAlgorithm._init_static_buffer_state`.  An empty dict
    # (or ``enabled: false``) keeps the legacy single-PER-buffer behaviour.
    static_buffer: Dict[str, Any] = field(default_factory=dict)

    # On-policy successful trajectory export for offline BC/value warmup.
    # ``enabled: false`` preserves the legacy PPO/A2C behaviour.
    success_trajectory: Dict[str, Any] = field(default_factory=dict)

    # SAC specific
    target_update_interval: int = 1  # Target network update frequency
    target_entropy: Union[str, float] = "auto"  # Target entropy for auto temperature

    # BC source (TD3/SAC)
    bc_source: str = 'fixed'  # 'fixed' for constant prior, 'lqr' for LQR rule-based expert
    lqr_config: Dict[str, Any] = field(default_factory=dict)

    # Warmup source for off-policy algorithms.
    # 'random' = heuristic random sampling
    # 'lqr'    = LQR rule-based expert (requires LQRExpertWrapper on env)
    # 'actor'  = checkpoint policy attached as model._warmup_policy (resume_sac)
    warmup_source: str = 'random'

    # Exploration mode for off-policy algorithms.
    # 'gaussian'       = algorithm-native exploration (TD3 noise / SAC sampling)
    # 'epsilon_greedy' = deterministic policy output mixed with predefined
    #                    exploration actions (see EpsilonGreedyExploration)
    exploration_mode: str = 'gaussian'

    # Epsilon-greedy exploration parameters
    exploit_prob_initial: float = 0.7       # Initial probability of using model output
    exploit_prob_final: float = 0.7         # Final probability (after linear decay)
    exploit_prob_decay_steps: int = 1000000  # Steps over which exploit_prob decays
    explore_actions: List[List[float]] = field(default_factory=lambda: [
        [0.1, 0.0],    # slow forward
        [1.0, 0.0],    # full throttle forward
        [-1.0, 0.0],   # brake
        [0.7, -0.5],   # left turn 0.5
        [0.7, 0.5],    # right turn 0.5
        [0.7, -1.0],   # hard left turn
        [0.7, 1.0],    # hard right turn
    ])

    # Gaussian repeats a sample N times. Epsilon-greedy repeats a sampled
    # exploration action N times, then forces N policy steps.
    explore_action_repeat: int = 1


@dataclass
class PolicyConfig:
    """Policy network configuration (supports both PPO and TD3)."""
    features_extractor_type: str = 'cnn'
    features_dim: int = 256
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 128])
    net_arch_vf: List[int] = field(default_factory=lambda: [256, 128])  # PPO value net
    net_arch_qf: List[int] = field(default_factory=lambda: [256, 256])  # TD3 critic net
    value_head_type: str = 'scalar'
    value_num_bins: int = 129
    value_support_min: float = -7.0
    value_support_max: float = 7.0
    value_transform: str = 'identity'
    action_distribution: str = 'auto'
    beta_min_a_b_value: float = 1.0
    beta_epsilon: float = 1e-6
    beta_deterministic_action: str = 'mean'

    # V2-specific fields (used when features_extractor_type == 'combined_v2')
    use_layer_norm: bool = True
    use_layer_norm_policy_head: bool = False
    cnn_channels: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    state_neurons: List[int] = field(default_factory=lambda: [256, 256])
    fusion_dims: List[int] = field(default_factory=lambda: [512])
    rgb_camera_feature_dim: int = 128

    # Distributional Q-value (two-hot encoding) — off-policy only (TD3/SAC)
    use_distributional: bool = False    # Enable reward bucketing + two-hot Q-value
    num_bins: int = 255                 # Number of bins for distributional Q
    v_min: float = -300.0              # Minimum Q-value (before symlog)
    v_max: float = 800.0               # Maximum Q-value (before symlog)
    use_symlog: bool = True             # Apply symlog transform before binning

    # SAC-specific
    log_std_init: float = -3.0          # Initial log std for SAC actor

    # SAC actor/critic MoE trunk; disabled preserves the shared MLP.
    moe_actor_enabled: bool = False
    moe_actor_num_experts: int = 4
    moe_actor_top_k: int = 2
    moe_actor_noisy_gating: bool = True
    moe_critic_enabled: bool = False
    moe_critic_num_experts: int = 4
    moe_critic_top_k: int = 2
    moe_critic_noisy_gating: bool = True
    moe_aux_loss_weight: float = 0.0     # Weight on the load-balancing aux loss (0 disables)


@dataclass
class TrainingConfig:
    """Training configuration."""
    seed: int = 3407
    device: str = 'cuda'
    log_dir: str = './logs'
    save_freq: int = 10000
    verbose: int = 1
    log_interval: int = 1  # Log every N iterations


@dataclass
class AdapterConfig:
    """Environment adapter configuration."""
    mode: str = 'standard'
    min_ready: int = 1
    timeout: float = 120.0  # Timeout for step/reset operations


@dataclass
class VisualizationConfig:
    """TrainingVisualizer configuration."""
    enabled: bool = True  # Enable episode video recording
    save_interval: int = 10  # Save video every N episodes
    fps: int = 10  # Video frame rate
    lazy_capture: bool = True  # Memory-friendly mode
    overlay_info: bool = True  # Show reward/action overlay
    max_videos: int = 50  # Maximum videos to keep


@dataclass
class Config:
    """Full training configuration."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    env_config_path: Optional[str] = None
    env_config: Optional[Dict] = None
    raw_config: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Return the original YAML structure when available."""
        if self.raw_config is not None:
            return deepcopy(self.raw_config)

        algo = self.algorithm
        algo_dict = {
            'name': algo.name,
            'total_timesteps': algo.total_timesteps,
            'batch_size': algo.batch_size,
            'gamma': algo.gamma,
            'learning_rate': algo.learning_rate,
            'learning_rate_final': algo.learning_rate_final,
        }
        # Add algorithm-specific params
        if algo.name in ('ppo', 'a2c'):
            on_policy_common = {
                'n_steps': algo.n_steps,
                'gae_lambda': algo.gae_lambda,
                'ent_coef': algo.ent_coef,
                'ent_coef_final': algo.ent_coef_final,
                'vf_coef': algo.vf_coef,
                'max_grad_norm': algo.max_grad_norm,
                'normalize_advantage': algo.normalize_advantage,
                'truncation': algo.truncation,
            }
            algo_dict.update(on_policy_common)

        if algo.name == 'ppo':
            algo_dict.update({
                'n_epochs': algo.n_epochs,
                'clip_range': algo.clip_range,
                'target_kl': algo.target_kl,
                'success_trajectory': algo.success_trajectory,
            })
        elif algo.name == 'a2c':
            algo_dict.update({
                'success_trajectory': algo.success_trajectory,
            })
        elif algo.name in ('td3', 'sac'):
            algo_dict.update({
                'buffer_size': algo.buffer_size,
                'learning_starts': algo.learning_starts,
                'tau': algo.tau,
                'train_freq': algo.train_freq,
                'gradient_steps': algo.gradient_steps,
                'bc_source': algo.bc_source,
                'lqr_config': algo.lqr_config,
                'warmup_source': algo.warmup_source,
                # Exploration
                'exploration_mode': algo.exploration_mode,
                'exploit_prob_initial': algo.exploit_prob_initial,
                'exploit_prob_final': algo.exploit_prob_final,
                'exploit_prob_decay_steps': algo.exploit_prob_decay_steps,
                'explore_actions': algo.explore_actions,
                'explore_action_repeat': algo.explore_action_repeat,
                # PER (always enabled)
                'per_alpha': algo.per_alpha,
                'per_beta': algo.per_beta,
                'per_beta_annealing_steps': algo.per_beta_annealing_steps,
                'per_min_priority': algo.per_min_priority,
                # Static replay buffer
                'static_buffer': algo.static_buffer,
                # BC
                'bc_enabled': algo.bc_enabled,
                'bc_lambda_initial': algo.bc_lambda_initial,
                'bc_lambda_decay_steps': algo.bc_lambda_decay_steps,
                'bc_lambda_final': algo.bc_lambda_final,
                'bc_prior_action': algo.bc_prior_action,
            })
            if algo.name == 'td3':
                algo_dict.update({
                    'policy_delay': algo.policy_delay,
                    'target_policy_noise': algo.target_policy_noise,
                    'target_noise_clip': algo.target_noise_clip,
                    'exploration_noise': algo.exploration_noise,
                    'exploration_noise_decay_steps': algo.exploration_noise_decay_steps,
                    'exploration_noise_final_scale': algo.exploration_noise_final_scale,
                })
            elif algo.name == 'sac':
                algo_dict.update({
                    'ent_coef': algo.ent_coef,
                    'target_update_interval': algo.target_update_interval,
                    'target_entropy': algo.target_entropy,
                })
        
        policy_dict = {
            'feature_extractor': {
                'type': self.policy.features_extractor_type,
                'features_dim': self.policy.features_dim,
                'use_layer_norm': self.policy.use_layer_norm,
                'cnn_channels': self.policy.cnn_channels,
                'state_neurons': self.policy.state_neurons,
                'fusion_dims': self.policy.fusion_dims,
                'rgb_camera_feature_dim': self.policy.rgb_camera_feature_dim,
            },
            'net_arch': {
                'pi': self.policy.net_arch_pi,
            },
        }
        if algo.name in ('ppo', 'a2c'):
            policy_dict['net_arch']['vf'] = self.policy.net_arch_vf
            policy_dict.update({
                'use_layer_norm_policy_head': self.policy.use_layer_norm_policy_head,
                'value_head_type': self.policy.value_head_type,
                'value_num_bins': self.policy.value_num_bins,
                'value_support_min': self.policy.value_support_min,
                'value_support_max': self.policy.value_support_max,
                'value_transform': self.policy.value_transform,
                'action_distribution': self.policy.action_distribution,
                'beta_min_a_b_value': self.policy.beta_min_a_b_value,
                'beta_epsilon': self.policy.beta_epsilon,
                'beta_deterministic_action': self.policy.beta_deterministic_action,
            })
        elif algo.name in ('td3', 'sac'):
            policy_dict['net_arch']['qf'] = self.policy.net_arch_qf
            policy_dict.update({
                'log_std_init': self.policy.log_std_init,
                'distributional': {
                    'enabled': self.policy.use_distributional,
                    'num_bins': self.policy.num_bins,
                    'v_min': self.policy.v_min,
                    'v_max': self.policy.v_max,
                    'use_symlog': self.policy.use_symlog,
                },
                'moe': {
                    'aux_loss_weight': self.policy.moe_aux_loss_weight,
                    'actor': {
                        'enabled': self.policy.moe_actor_enabled,
                        'num_experts': self.policy.moe_actor_num_experts,
                        'top_k': self.policy.moe_actor_top_k,
                        'noisy_gating': self.policy.moe_actor_noisy_gating,
                    },
                    'critic': {
                        'enabled': self.policy.moe_critic_enabled,
                        'num_experts': self.policy.moe_critic_num_experts,
                        'top_k': self.policy.moe_critic_top_k,
                        'noisy_gating': self.policy.moe_critic_noisy_gating,
                    },
                },
            })

        return {
            'algorithm': algo_dict,
            'policy': policy_dict,
            'training': {
                'seed': self.training.seed,
                'device': self.training.device,
                'log_dir': self.training.log_dir,
                'save_freq': self.training.save_freq,
                'verbose': self.training.verbose,
                'log_interval': self.training.log_interval,
                'adapter': {
                    'mode': self.adapter.mode,
                    'min_ready': self.adapter.min_ready,
                    'timeout': self.adapter.timeout,
                },
                'visualization': {
                    'enabled': self.visualization.enabled,
                    'save_interval': self.visualization.save_interval,
                    'fps': self.visualization.fps,
                    'lazy_capture': self.visualization.lazy_capture,
                    'overlay_info': self.visualization.overlay_info,
                    'max_videos': self.visualization.max_videos,
                },
            },
            'env_config_path': self.env_config_path,
            'env': self.env_config,
        }


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to config file (for example, ppo_bev_example.yaml).
        
    Returns:
        Loaded Config object.
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    # Parse algorithm config
    algo_dict = raw_config.get('algorithm', {})
    algorithm = AlgorithmConfig(
        name=algo_dict.get('name', 'ppo'),
        total_timesteps=algo_dict.get('total_timesteps', 100000),
        batch_size=algo_dict.get('batch_size', 128),
        gamma=algo_dict.get('gamma', 0.99),
        learning_rate=algo_dict.get('learning_rate', 3e-4),
        learning_rate_final=algo_dict.get('learning_rate_final'),
        # PPO specific
        n_steps=algo_dict.get('n_steps', 1024),
        n_epochs=algo_dict.get('n_epochs', 3),
        gae_lambda=algo_dict.get('gae_lambda', 0.95),
        clip_range=algo_dict.get('clip_range', 0.2),
        ent_coef=algo_dict.get('ent_coef', 0.01),
        ent_coef_final=algo_dict.get('ent_coef_final'),
        vf_coef=algo_dict.get('vf_coef', 0.5),
        max_grad_norm=algo_dict.get('max_grad_norm', 0.5),
        target_kl=algo_dict.get('target_kl'),
        normalize_advantage=algo_dict.get('normalize_advantage', True),
        truncation=algo_dict.get('truncation'),
        # Off-policy specific
        buffer_size=algo_dict.get('buffer_size', 100000),
        learning_starts=algo_dict.get('learning_starts', 1000),
        tau=algo_dict.get('tau', 0.005),
        train_freq=algo_dict.get('train_freq', 1),
        gradient_steps=algo_dict.get('gradient_steps', 1),
        # TD3 specific
        policy_delay=algo_dict.get('policy_delay', 2),
        # Support both float and list for per-dimension noise
        target_policy_noise=algo_dict.get('target_policy_noise', 0.2),
        target_noise_clip=algo_dict.get('target_noise_clip', 0.5),
        exploration_noise=algo_dict.get('exploration_noise', 0.1),
        exploration_noise_decay_steps=algo_dict.get('exploration_noise_decay_steps', 0),
        exploration_noise_final_scale=algo_dict.get('exploration_noise_final_scale', 1.0),
        # BC
        bc_enabled=algo_dict.get('bc_enabled', False),
        bc_lambda_initial=algo_dict.get('bc_lambda_initial', 1.0),
        bc_lambda_decay_steps=algo_dict.get('bc_lambda_decay_steps', 200000),
        bc_lambda_final=algo_dict.get('bc_lambda_final', 0.0),
        bc_prior_action=algo_dict.get('bc_prior_action', [0.6, 0.0]),
        # PER (always enabled in the off-policy stack)
        per_alpha=algo_dict.get('per_alpha', 0.6),
        per_beta=algo_dict.get('per_beta', 0.4),
        per_beta_annealing_steps=algo_dict.get('per_beta_annealing_steps', 100000),
        per_min_priority=algo_dict.get('per_min_priority', 1e-6),
        # Static replay buffer
        static_buffer=algo_dict.get('static_buffer', {}) or {},
        # On-policy successful trajectory export
        success_trajectory=algo_dict.get('success_trajectory', {}) or {},
        # SAC specific
        target_update_interval=algo_dict.get('target_update_interval', 1),
        target_entropy=algo_dict.get('target_entropy', 'auto'),
        # BC source
        bc_source=algo_dict.get('bc_source', 'fixed'),
        lqr_config=algo_dict.get('lqr_config') or {},
        # Warmup and exploration
        warmup_source=algo_dict.get('warmup_source', 'random'),
        exploration_mode=algo_dict.get('exploration_mode', 'gaussian'),
        exploit_prob_initial=algo_dict.get('exploit_prob_initial', 0.7),
        exploit_prob_final=algo_dict.get('exploit_prob_final', 0.7),
        exploit_prob_decay_steps=algo_dict.get('exploit_prob_decay_steps', 1000000),
        explore_actions=algo_dict.get('explore_actions', [
            [0.1, 0.0], [1.0, 0.0], [-1.0, 0.0],
            [0.7, -0.5], [0.7, 0.5], [0.7, -1.0], [0.7, 1.0],
        ]),
        explore_action_repeat=int(algo_dict.get('explore_action_repeat', 1)),
    )
    
    # Parse policy config
    policy_dict = raw_config.get('policy', {})
    fe_dict = policy_dict.get('feature_extractor', {})
    net_arch = policy_dict.get('net_arch', {})
    
    # Distributional options accept either nested or legacy flat keys.
    dist_dict = policy_dict.get('distributional', {})
    moe_dict = policy_dict.get('moe', {}) or {}
    moe_actor_dict = moe_dict.get('actor', {}) or {}
    moe_critic_dict = moe_dict.get('critic', {}) or {}

    policy = PolicyConfig(
        features_extractor_type=fe_dict.get('type', 'cnn'),
        features_dim=fe_dict.get('features_dim', 256),
        net_arch_pi=net_arch.get('pi', [256, 128]),
        net_arch_vf=net_arch.get('vf', [256, 128]),
        net_arch_qf=net_arch.get('qf', [256, 256]),
        value_head_type=policy_dict.get('value_head_type', 'scalar'),
        value_num_bins=policy_dict.get('value_num_bins', 129),
        value_support_min=policy_dict.get('value_support_min', -7.0),
        value_support_max=policy_dict.get('value_support_max', 7.0),
        value_transform=policy_dict.get('value_transform', 'identity'),
        action_distribution=policy_dict.get('action_distribution', 'auto'),
        beta_min_a_b_value=float(policy_dict.get('beta_min_a_b_value', 1.0)),
        beta_epsilon=float(policy_dict.get('beta_epsilon', 1e-6)),
        beta_deterministic_action=policy_dict.get('beta_deterministic_action', 'mean'),
        # V2-specific fields
        use_layer_norm=fe_dict.get(
            'use_layer_norm',
            policy_dict.get('use_layer_norm', True),
        ),
        use_layer_norm_policy_head=policy_dict.get('use_layer_norm_policy_head', False),
        cnn_channels=fe_dict.get('cnn_channels', [8, 16, 32, 64, 128, 256]),
        state_neurons=fe_dict.get('state_neurons', [256, 256]),
        fusion_dims=fe_dict.get('fusion_dims', [512]),
        rgb_camera_feature_dim=int(fe_dict.get('rgb_camera_feature_dim', 128)),
        # Distributional Q-value (two-hot)
        use_distributional=dist_dict.get('enabled', policy_dict.get('use_distributional', False)),
        num_bins=dist_dict.get('num_bins', policy_dict.get('num_bins', 255)),
        v_min=dist_dict.get('v_min', policy_dict.get('v_min', -300.0)),
        v_max=dist_dict.get('v_max', policy_dict.get('v_max', 800.0)),
        use_symlog=dist_dict.get('use_symlog', policy_dict.get('use_symlog', True)),
        log_std_init=policy_dict.get('log_std_init', -3.0),
        # TD3 and SAC consume the MoE fields.
        moe_actor_enabled=bool(moe_actor_dict.get('enabled', False)),
        moe_actor_num_experts=int(moe_actor_dict.get('num_experts', 4)),
        moe_actor_top_k=int(moe_actor_dict.get('top_k', 2)),
        moe_actor_noisy_gating=bool(moe_actor_dict.get('noisy_gating', True)),
        moe_critic_enabled=bool(moe_critic_dict.get('enabled', False)),
        moe_critic_num_experts=int(moe_critic_dict.get('num_experts', 4)),
        moe_critic_top_k=int(moe_critic_dict.get('top_k', 2)),
        moe_critic_noisy_gating=bool(moe_critic_dict.get('noisy_gating', True)),
        moe_aux_loss_weight=float(moe_dict.get('aux_loss_weight', 0.0)),
    )
    
    training_dict = raw_config.get('training', {})
    training = TrainingConfig(
        seed=training_dict.get('seed', 3407),
        device=training_dict.get('device', 'cuda'),
        log_dir=training_dict.get('log_dir', './logs'),
        save_freq=training_dict.get('save_freq', 10000),
        verbose=training_dict.get('verbose', 1),
        log_interval=training_dict.get('log_interval', 1),
    )
    
    adapter_dict = training_dict.get('adapter', {})
    adapter_mode = str(adapter_dict.get('mode', 'standard')).strip().lower()
    if adapter_mode != 'standard':
        raise ValueError(
            f"adapter.mode only supports 'standard'; got '{adapter_mode}'"
        )
    adapter = AdapterConfig(
        mode='standard',
        min_ready=adapter_dict.get('min_ready', 1),
        timeout=adapter_dict.get('timeout', 120.0),
    )
    
    vis_dict = training_dict.get('visualization', {})
    visualization = VisualizationConfig(
        enabled=vis_dict.get('enabled', True),
        save_interval=vis_dict.get('save_interval', 10),
        fps=vis_dict.get('fps', 10),
        lazy_capture=vis_dict.get('lazy_capture', True),
        overlay_info=vis_dict.get('overlay_info', True),
        max_videos=vis_dict.get('max_videos', 50),
    )
    
    config = Config(
        algorithm=algorithm,
        policy=policy,
        training=training,
        adapter=adapter,
        visualization=visualization,
        raw_config=deepcopy(raw_config),
    )
    
    # Inline environment config takes precedence over an external file.
    env_section = raw_config.get('env')
    if env_section:
        config.env_config = env_section
        config.env_config_path = str(config_path)
    elif raw_config.get('env_config'):
        config.env_config_path = raw_config['env_config']
        env_config_path = Path(config.env_config_path)
        if not env_config_path.is_absolute():
            env_config_path = config_path.parent / env_config_path
        config.env_config = load_env_config(env_config_path)
    
    return config


def load_env_config(config_path: Union[str, Path]) -> Dict:
    """Load an environment YAML file."""
    config_path = Path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Union[Config, Dict], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
