from dataclasses import dataclass, field


@dataclass
class Config:
    # --- 世界 ---
    world_size: float = 10.0
    n_agvs: int = 4
    max_agents: int = 8
    agv_radius: float = 0.25
    step_size: float = 0.3
    max_steps: int = 200

    # --- 动态障碍物 ---
    n_dynamic_obs: int = 3
    obs_radius: float = 0.2
    obs_speed: float = 0.15

    # --- 优先级 ---
    enable_priority: bool = True
    priority_ratio: float = 0.25  # 高优先级 AGV 比例
    reward_yield: float = 2.0     # 低优先级主动避让奖励
    reward_prio_collision: float = -30.0  # 撞高优先级额外惩罚

    # --- 奖励 ---
    reward_goal: float = 30.0
    reward_collision: float = -20.0
    reward_step: float = -0.05
    reward_shaping_scale: float = 0.5

    # --- DQN ---
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    hidden: int = 256
    buffer_capacity: int = 200_000
    learn_start: int = 2000

    # --- NoisyNet ---
    use_noisy: bool = True

    # --- PER ---
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-4

    # --- 训练 ---
    episodes: int = 3000
    epsilon: float = 1.0
    eps_min: float = 0.02
    eps_decay: float = 0.998
    target_update_freq: int = 100  # 每隔多少步硬更新 target（0=软更新）
    grad_clip: float = 10.0
    device: str = "cuda"
    render_interval: int = 200
    print_interval: int = 50
    save_path: str = "e:/carAI/rl_advanced/best_model.pth"

    # --- 课程学习 ---
    curriculum: bool = True
    curriculum_stages: list = field(default_factory=lambda: [
        {"episodes": 800,  "n_agvs": 2, "n_dynamic_obs": 0, "enable_priority": False},
        {"episodes": 800,  "n_agvs": 3, "n_dynamic_obs": 2, "enable_priority": False},
        {"episodes": 800,  "n_agvs": 4, "n_dynamic_obs": 3, "enable_priority": True},
        {"episodes": 600,  "n_agvs": 5, "n_dynamic_obs": 4, "enable_priority": True},
    ])
