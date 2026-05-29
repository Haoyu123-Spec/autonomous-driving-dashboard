from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """多AGV冲突消解DRL训练配置。

    既可直接构造（代码兼容），也可通过 from_hydra() 从 Hydra DictConfig 创建。
    """

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

    # --- 紧急订单插入 ---
    emergency_enabled: bool = True
    emergency_prob: float = 0.3
    emergency_trigger_step: int = 10
    emergency_max_per_episode: int = 1
    reward_emergency_bonus: float = 25.0

    # --- 奖励（复合奖励函数） ---
    direction_angle_coef: float = 1.0
    reward_collision_static: float = -5.0
    reward_collision_dynamic: float = -8.0
    reward_step: float = -0.05
    coop_danger_threshold: float = 2.0
    coop_safe_threshold: float = 4.0
    coop_danger_penalty: float = -3.0
    coop_zone_reward: float = 1.0
    reward_complete: float = 20.0

    # --- DQN ---
    batch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    hidden: int = 256
    buffer_capacity: int = 200_000
    learn_start: int = 2000

    # --- NoisyNet ---
    use_noisy: bool = False

    # --- Self-Attention ---
    use_attention: bool = True
    attention_heads: int = 4

    # --- PER ---
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 1e-4

    # --- 训练 ---
    episodes: int = 3000
    epsilon: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.999
    target_update_freq: int = 100
    grad_clip: float = 10.0
    device: str = "cuda"
    render_interval: int = 200
    print_interval: int = 50
    save_path: str = "e:/carAI/rl_advanced/best_model.pth"
    resume_path: str = ""

    # --- 课程学习 ---
    curriculum: bool = True
    curriculum_stages: list = field(default_factory=lambda: [
        {"episodes": 400, "n_agvs": 2, "n_dynamic_obs": 0, "emergency_enabled": False},
        {"episodes": 400, "n_agvs": 3, "n_dynamic_obs": 1, "emergency_enabled": False},
        {"episodes": 500, "n_agvs": 3, "n_dynamic_obs": 2, "emergency_enabled": False},
        {"episodes": 500, "n_agvs": 4, "n_dynamic_obs": 2, "emergency_enabled": False},
        {"episodes": 600, "n_agvs": 4, "n_dynamic_obs": 3, "emergency_enabled": True},
        {"episodes": 600, "n_agvs": 5, "n_dynamic_obs": 4, "emergency_enabled": True},
    ])

    @classmethod
    def from_hydra(cls, hydra_cfg: Any) -> Config:
        """从 Hydra OmegaConf DictConfig 构造 Config 实例。

        Args:
            hydra_cfg: Hydra 提供的配置对象，结构与 conf/config.yaml 一致。
        """
        c = cls()
        c.world_size = float(hydra_cfg.world.size)
        c.n_agvs = int(hydra_cfg.world.n_agvs)
        c.max_agents = int(hydra_cfg.world.max_agents)
        c.agv_radius = float(hydra_cfg.world.agv_radius)
        c.step_size = float(hydra_cfg.world.step_size)
        c.max_steps = int(hydra_cfg.world.max_steps)

        c.n_dynamic_obs = int(hydra_cfg.obstacle.n_dynamic_obs)
        c.obs_radius = float(hydra_cfg.obstacle.obs_radius)
        c.obs_speed = float(hydra_cfg.obstacle.obs_speed)

        c.emergency_enabled = bool(hydra_cfg.emergency.enabled)
        c.emergency_prob = float(hydra_cfg.emergency.prob)
        c.emergency_trigger_step = int(hydra_cfg.emergency.trigger_step)
        c.emergency_max_per_episode = int(hydra_cfg.emergency.max_per_episode)
        c.reward_emergency_bonus = float(hydra_cfg.emergency.bonus)

        r = hydra_cfg.reward
        c.direction_angle_coef = float(r.direction_angle_coef)
        c.reward_collision_static = float(r.collision_static)
        c.reward_collision_dynamic = float(r.collision_dynamic)
        c.reward_step = float(r.step)
        c.coop_danger_threshold = float(r.coop_danger_threshold)
        c.coop_safe_threshold = float(r.coop_safe_threshold)
        c.coop_danger_penalty = float(r.coop_danger_penalty)
        c.coop_zone_reward = float(r.coop_zone_reward)
        c.reward_complete = float(r.complete)

        d = hydra_cfg.dqn
        c.batch_size = int(d.batch_size)
        c.lr = float(d.lr)
        c.gamma = float(d.gamma)
        c.tau = float(d.tau)
        c.hidden = int(d.hidden)
        c.buffer_capacity = int(d.buffer_capacity)
        c.learn_start = int(d.learn_start)
        c.use_noisy = bool(d.use_noisy)
        c.use_attention = bool(d.use_attention)
        c.attention_heads = int(d.attention_heads)
        c.use_per = bool(d.use_per)
        c.per_alpha = float(d.per_alpha)
        c.per_beta = float(d.per_beta)
        c.per_beta_increment = float(d.per_beta_increment)
        c.target_update_freq = int(d.target_update_freq)
        c.grad_clip = float(d.grad_clip)

        t = hydra_cfg.training
        c.episodes = int(t.episodes)
        c.epsilon = float(t.epsilon)
        c.eps_min = float(t.eps_min)
        c.eps_decay = float(t.eps_decay)
        c.render_interval = int(t.render_interval)
        c.print_interval = int(t.print_interval)
        c.save_path = str(t.save_path)
        c.resume_path = str(t.resume_path)
        c.device = "cuda"

        cur = hydra_cfg.curriculum
        c.curriculum = bool(cur.enabled)
        c.curriculum_stages = []
        for s in cur.stages:
            c.curriculum_stages.append({
                "episodes": int(s.episodes),
                "n_agvs": int(s.n_agvs),
                "n_dynamic_obs": int(s.n_dynamic_obs),
                "emergency_enabled": bool(s.emergency_enabled),
            })

        return c
