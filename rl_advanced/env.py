import numpy as np

from config import Config


def compute_max_state_dim(cfg: Config) -> int:
    """计算所有可能场景下的最大观测维度"""
    if cfg.curriculum:
        max_obs = max(s["n_dynamic_obs"] for s in cfg.curriculum_stages)
    else:
        max_obs = cfg.n_dynamic_obs
    return 4 + (cfg.max_agents - 1) * 2 + max_obs * 2


class MultiAGVEnv:
    """多 AGV 环境：动态障碍物 + 三区协同 + 方向引导奖励 + 紧急订单插入"""

    # 动作 → 方向向量映射
    ACTION_DIRS = np.array([
        [0., 0.],   # 0: 停留
        [1., 0.],   # 1: 右
        [-1., 0.],  # 2: 左
        [0., 1.],   # 3: 上
        [0., -1.],  # 4: 下
    ], dtype=np.float32)

    def __init__(self, cfg: Config, fixed_state_dim: int = None):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.fixed_state_dim = fixed_state_dim or compute_max_state_dim(cfg)

    def reset(self):
        cfg = self.cfg
        self.positions = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.goals = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.done_agents = np.zeros(cfg.n_agvs, dtype=bool)
        self.headings = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.emergency_triggered = np.zeros(cfg.n_agvs, dtype=bool)
        self.step_count = 0
        self._emergency_count = 0

        # 预定紧急订单触发步（episode 级别决策）
        if cfg.emergency_enabled and self.rng.random() < cfg.emergency_prob:
            lo = cfg.emergency_trigger_step
            hi = max(lo + 1, int(cfg.max_steps * 0.4))
            self._emergency_trigger_step = self.rng.integers(lo, hi)
        else:
            self._emergency_trigger_step = None

        for i in range(cfg.n_agvs):
            self.positions[i] = self.rng.uniform(1, cfg.world_size - 1, 2)
            while True:
                g = self.rng.uniform(1, cfg.world_size - 1, 2)
                if np.linalg.norm(g - self.positions[i]) > cfg.world_size * 0.4:
                    self.goals[i] = g
                    break
            to_goal = self.goals[i] - self.positions[i]
            self.headings[i] = to_goal / (np.linalg.norm(to_goal) + 1e-8)

        self.obs_positions = np.zeros((cfg.n_dynamic_obs, 2), dtype=np.float32)
        self.obs_directions = np.zeros((cfg.n_dynamic_obs, 2), dtype=np.float32)
        for k in range(cfg.n_dynamic_obs):
            self.obs_positions[k] = self.rng.uniform(1, cfg.world_size - 1, 2)
            angle = self.rng.uniform(0, 2 * np.pi)
            self.obs_directions[k] = [np.cos(angle), np.sin(angle)]

        return self._get_states()

    def _get_states(self):
        cfg = self.cfg
        D = self.fixed_state_dim
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

        return states

    def _move_obstacles(self):
        cfg = self.cfg
        for k in range(cfg.n_dynamic_obs):
            self.obs_positions[k] += self.obs_directions[k] * cfg.obs_speed
            for d in range(2):
                if self.obs_positions[k][d] <= 0 or self.obs_positions[k][d] >= cfg.world_size:
                    self.obs_directions[k][d] *= -1
                    self.obs_positions[k][d] = np.clip(self.obs_positions[k][d], 0, cfg.world_size)

    def _compute_direction_reward(self, i):
        """R_direction = log(max(d,1)) * exp(-a * theta)"""
        cfg = self.cfg
        to_goal = self.goals[i] - self.positions[i]
        d = np.linalg.norm(to_goal)
        d_clipped = max(d, 1.0)

        heading = self.headings[i]
        heading_norm = np.linalg.norm(heading)
        if heading_norm < 1e-8:
            return 0.0

        cos_theta = np.dot(heading, to_goal) / (heading_norm * d + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.log(d_clipped) * np.exp(-cfg.direction_angle_coef * theta)

    def _compute_coop_rewards(self):
        """R_coop: 三区制协同奖励"""
        cfg = self.cfg
        rewards = np.zeros(cfg.n_agvs, dtype=np.float32)

        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            for j in range(i + 1, cfg.n_agvs):
                if self.done_agents[j]:
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < cfg.coop_danger_threshold:
                    rewards[i] += cfg.coop_danger_penalty
                    rewards[j] += cfg.coop_danger_penalty
                elif dist <= cfg.coop_safe_threshold:
                    rewards[i] += cfg.coop_zone_reward
                    rewards[j] += cfg.coop_zone_reward

        return rewards

    def _maybe_trigger_emergency(self):
        """随机触发紧急订单：episode 级别概率，在指定步触发"""
        cfg = self.cfg
        if not cfg.emergency_enabled:
            return 0
        if self._emergency_count >= cfg.emergency_max_per_episode:
            return 0
        # episode 级别：在 reset 时预定的触发步
        if self._emergency_trigger_step is None:
            return 0
        if self.step_count == self._emergency_trigger_step:
            # 选一个未完成 AGV
            candidates = [i for i in range(cfg.n_agvs) if not self.done_agents[i]]
            if not candidates:
                return 0
            victim = self.rng.choice(candidates)
            while True:
                new_goal = self.rng.uniform(1, cfg.world_size - 1, 2)
                d = np.linalg.norm(new_goal - self.positions[victim])
                if cfg.world_size * 0.15 < d < cfg.world_size * 0.35:
                    break
            self.goals[victim] = new_goal
            to_new = new_goal - self.positions[victim]
            self.headings[victim] = to_new / (np.linalg.norm(to_new) + 1e-8)
            self.emergency_triggered[victim] = True
            self._emergency_count += 1
            return 1
        return 0

    def step(self, actions):
        cfg = self.cfg

        # 0. 紧急订单触发（移动前）
        emergency_now = self._maybe_trigger_emergency()

        # 1. 移动 AGV + 墙壁碰撞检测
        wall_collision = np.zeros(cfg.n_agvs, dtype=bool)
        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            a = actions[i]
            new_pos = self.positions[i] + self.ACTION_DIRS[a] * cfg.step_size
            if new_pos[0] < 0 or new_pos[0] > cfg.world_size or \
               new_pos[1] < 0 or new_pos[1] > cfg.world_size:
                wall_collision[i] = True
                new_pos = np.clip(new_pos, 0, cfg.world_size)
            self.positions[i] = new_pos
            if a != 0:
                self.headings[i] = self.ACTION_DIRS[a].copy()

        self._move_obstacles()
        self.step_count += 1

        # 2. AGV 间碰撞检测
        collision = np.zeros(cfg.n_agvs, dtype=bool)
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

        # 3. 障碍物碰撞检测
        obs_collision = np.zeros(cfg.n_agvs, dtype=bool)
        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue
            for k in range(cfg.n_dynamic_obs):
                d = np.linalg.norm(self.positions[i] - self.obs_positions[k])
                if d < cfg.agv_radius + cfg.obs_radius:
                    obs_collision[i] = True
                    break

        # 4. 计算奖励
        rewards = np.zeros(cfg.n_agvs, dtype=np.float32)
        emergency_completed = 0
        goals_this_step = 0

        for i in range(cfg.n_agvs):
            if self.done_agents[i]:
                continue

            # R_step
            rewards[i] += cfg.reward_step

            # R_direction
            rewards[i] += self._compute_direction_reward(i)

            # R_collision
            if wall_collision[i] or obs_collision[i]:
                rewards[i] += cfg.reward_collision_static
                self.done_agents[i] = True
            if collision[i]:
                rewards[i] += cfg.reward_collision_dynamic
                self.done_agents[i] = True

            # R_complete: 到达目标
            cur_dist = np.linalg.norm(self.positions[i] - self.goals[i])
            if cur_dist < cfg.step_size:
                rewards[i] += cfg.reward_complete
                if self.emergency_triggered[i]:
                    rewards[i] += cfg.reward_emergency_bonus
                    emergency_completed += 1
                self.done_agents[i] = True
                goals_this_step += 1

        # R_coop
        coop_rewards = self._compute_coop_rewards()
        for i in range(cfg.n_agvs):
            if not self.done_agents[i]:
                rewards[i] += coop_rewards[i]

        done = self.done_agents.all() or self.step_count >= cfg.max_steps
        info = {
            "collision": collision.sum(),
            "obs_collision": obs_collision.sum(),
            "wall_collision": wall_collision.sum(),
            "goal_reached": goals_this_step,
            "emergency_triggered": emergency_now,
            "emergency_completed": emergency_completed,
        }
        return self._get_states(), rewards, done, info
