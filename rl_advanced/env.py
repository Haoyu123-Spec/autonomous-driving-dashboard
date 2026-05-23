import numpy as np
from config import Config


def compute_max_state_dim(cfg: Config) -> int:
    """计算所有可能场景下的最大观测维度"""
    if cfg.curriculum:
        max_obs = max(s["n_dynamic_obs"] for s in cfg.curriculum_stages)
        max_prio = any(s.get("enable_priority", False) for s in cfg.curriculum_stages)
    else:
        max_obs = cfg.n_dynamic_obs
        max_prio = cfg.enable_priority
    return (4 + (cfg.max_agents - 1) * 2 + max_obs * 2 + int(max_prio))


class MultiAGVEnv:
    """多 AGV 环境：动态障碍物 + 优先通行"""

    def __init__(self, cfg: Config, fixed_state_dim: int = None):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.fixed_state_dim = fixed_state_dim or compute_max_state_dim(cfg)

    def reset(self):
        cfg = self.cfg
        self.positions = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.goals = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.done_agents = np.zeros(cfg.n_agvs, dtype=bool)
        self.priority = np.zeros(cfg.n_agvs, dtype=bool)
        self.step_count = 0

        for i in range(cfg.n_agvs):
            self.positions[i] = self.rng.uniform(1, cfg.world_size - 1, 2)
            while True:
                g = self.rng.uniform(1, cfg.world_size - 1, 2)
                if np.linalg.norm(g - self.positions[i]) > cfg.world_size * 0.4:
                    self.goals[i] = g
                    break

        if cfg.enable_priority:
            n_high = max(1, int(cfg.n_agvs * cfg.priority_ratio))
            high_idx = self.rng.choice(cfg.n_agvs, n_high, replace=False)
            self.priority[high_idx] = True

        self.obs_positions = np.zeros((cfg.n_dynamic_obs, 2), dtype=np.float32)
        self.obs_directions = np.zeros((cfg.n_dynamic_obs, 2), dtype=np.float32)
        for k in range(cfg.n_dynamic_obs):
            self.obs_positions[k] = self.rng.uniform(1, cfg.world_size - 1, 2)
            angle = self.rng.uniform(0, 2 * np.pi)
            self.obs_directions[k] = [np.cos(angle), np.sin(angle)]

        return self._get_states()

    def _get_states(self):
        cfg = self.cfg
        D = self.fixed_state_dim  # 固定输出维度，零填充
        states = np.zeros((cfg.n_agvs, D), dtype=np.float32)

        for i in range(cfg.n_agvs):
            s = states[i]
            idx = 0
            s[idx:idx+2] = self.positions[i] / cfg.world_size
            idx += 2
            s[idx:idx+2] = self.goals[i] / cfg.world_size
            idx += 2

            others = []
            for j in range(cfg.n_agvs):
                if j != i:
                    rel = (self.positions[j] - self.positions[i]) / cfg.world_size
                    others.append((np.linalg.norm(rel), rel))
            others.sort(key=lambda x: x[0])
            for k in range(min(len(others), cfg.max_agents - 1)):
                s[idx:idx+2] = others[k][1]
                idx += 2

            idx = 4 + (cfg.max_agents - 1) * 2

            for k in range(cfg.n_dynamic_obs):
                s[idx:idx+2] = (self.obs_positions[k] - self.positions[i]) / cfg.world_size
                idx += 2

            if cfg.enable_priority:
                s[idx] = 1.0 if self.priority[i] else 0.0

        return states

    def _move_obstacles(self):
        cfg = self.cfg
        for k in range(cfg.n_dynamic_obs):
            self.obs_positions[k] += self.obs_directions[k] * cfg.obs_speed
            for d in range(2):
                if self.obs_positions[k][d] <= 0 or self.obs_positions[k][d] >= cfg.world_size:
                    self.obs_directions[k][d] *= -1
                    self.obs_positions[k][d] = np.clip(self.obs_positions[k][d], 0, cfg.world_size)

    def step(self, actions):
        cfg = self.cfg
        dirs = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
                        dtype=np.float32)

        prev_dist = np.linalg.norm(self.positions - self.goals, axis=1)

        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            self.positions[i] += dirs[actions[i]] * cfg.step_size
            self.positions[i] = np.clip(self.positions[i], 0, cfg.world_size)

        self._move_obstacles()
        self.step_count += 1

        collision = np.zeros(cfg.n_agvs, dtype=bool)
        collision_with_prio = np.zeros(cfg.n_agvs, dtype=bool)
        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            for j in range(i + 1, cfg.n_agvs):
                if self.done_agents[j]:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < 2 * cfg.agv_radius:
                    collision[i] = True
                    collision[j] = True
                    if self.priority[i] or self.priority[j]:
                        collision_with_prio[i] = True
                        collision_with_prio[j] = True

        obs_collision = np.zeros(cfg.n_agvs, dtype=bool)
        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            for k in range(cfg.n_dynamic_obs):
                d = np.linalg.norm(self.positions[i] - self.obs_positions[k])
                if d < cfg.agv_radius + cfg.obs_radius:
                    obs_collision[i] = True
                    break

        cur_dist = np.linalg.norm(self.positions - self.goals, axis=1)
        rewards = np.full(cfg.n_agvs, cfg.reward_step, dtype=np.float32)
        rewards += (prev_dist - cur_dist) * cfg.reward_shaping_scale

        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                rewards[i] = 0
                continue

            if cur_dist[i] < cfg.step_size:
                rewards[i] += cfg.reward_goal
                self.done_agents[i] = True

            if collision[i]:
                rewards[i] += cfg.reward_collision
                if collision_with_prio[i]:
                    rewards[i] += cfg.reward_prio_collision
                self.done_agents[i] = True

            if obs_collision[i]:
                rewards[i] += cfg.reward_collision
                self.done_agents[i] = True

        if cfg.enable_priority:
            for i in range(cfg.n_agvs):
                if self.done_agents[i] or self.priority[i]:
                    continue
                for j in range(cfg.n_agvs):
                    if i == j or not self.priority[j]:
                        continue
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < 1.0:
                        rewards[i] += cfg.reward_yield * (d - 2 * cfg.agv_radius) / 1.0

        done = self.done_agents.all() or self.step_count >= cfg.max_steps
        info = {
            "collision": collision.sum(),
            "obs_collision": obs_collision.sum(),
            "goal_reached": (cur_dist < cfg.step_size).sum(),
            "priority_yield": collision_with_prio.sum() == 0,
        }
        return self._get_states(), rewards, done, info
