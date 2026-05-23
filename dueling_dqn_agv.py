#!/usr/bin/env python3
"""
Multi-AGV Collision Avoidance with Dueling DQN
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class CONFIG:
    world_size: float = 10.0
    n_agvs: int = 4
    max_agents: int = 8
    agv_radius: float = 0.25
    step_size: float = 0.3
    max_steps: int = 200
    episodes: int = 2000
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 1.0
    eps_min: float = 0.02
    eps_decay: float = 0.998
    tau: float = 0.005
    hidden: int = 256
    buffer_capacity: int = 200_000
    learn_start: int = 2000
    reward_goal: float = 30.0
    reward_collision: float = -20.0
    reward_step: float = -0.05
    reward_shaping_scale: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    render_interval: int = 200
    print_interval: int = 50


class MultiAGVEnv:
    def __init__(self, cfg: CONFIG):
        self.cfg = cfg
        self.rng = np.random.default_rng()

    def reset(self):
        cfg = self.cfg
        self.positions = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.goals = np.zeros((cfg.n_agvs, 2), dtype=np.float32)
        self.done_agents = np.zeros(cfg.n_agvs, dtype=bool)
        self.step_count = 0
        for i in range(cfg.n_agvs):
            self.positions[i] = self.rng.uniform(1, cfg.world_size - 1, 2)
            while True:
                g = self.rng.uniform(1, cfg.world_size - 1, 2)
                if np.linalg.norm(g - self.positions[i]) > cfg.world_size * 0.4:
                    self.goals[i] = g
                    break
        return self._get_states()

    def _get_states(self):
        cfg = self.cfg
        obs_dim = 4 + (cfg.max_agents - 1) * 2
        states = np.zeros((cfg.n_agvs, obs_dim), dtype=np.float32)
        for i in range(cfg.n_agvs):
            s = states[i]
            s[0:2] = self.positions[i] / cfg.world_size
            s[2:4] = self.goals[i] / cfg.world_size
            others = []
            for j in range(cfg.n_agvs):
                if j != i:
                    rel = (self.positions[j] - self.positions[i]) / cfg.world_size
                    others.append((np.linalg.norm(rel), rel))
            others.sort(key=lambda x: x[0])
            for k, (_, rel) in enumerate(others[: cfg.max_agents - 1]):
                s[4 + k * 2 : 6 + k * 2] = rel
        return states

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
        self.step_count += 1
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
                self.done_agents[i] = True
        done = self.done_agents.all() or self.step_count >= cfg.max_steps
        return self._get_states(), rewards, done


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        s, a, r, ns, d = zip(*(self.buf[i] for i in idx))
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, cfg: CONFIG):
        self.action_dim = action_dim
        self.cfg = cfg
        self.online = DuelingDQN(state_dim, action_dim, cfg.hidden).to(cfg.device)
        self.target = DuelingDQN(state_dim, action_dim, cfg.hidden).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.epsilon = cfg.epsilon

    def act(self, states, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim, len(states))
        with torch.no_grad():
            t = torch.FloatTensor(np.array(states)).to(self.cfg.device)
            return self.online(t).argmax(dim=1).cpu().numpy()

    def update(self):
        cfg = self.cfg
        if len(self.buffer) < cfg.learn_start or len(self.buffer) < cfg.batch_size:
            return None
        s, a, r, ns, d = [x.to(cfg.device) for x in self.buffer.sample(cfg.batch_size)]
        with torch.no_grad():
            next_a = self.online(ns).argmax(dim=1)
            target_q = self.target(ns).gather(1, next_a.unsqueeze(1)).squeeze()
            y = r + cfg.gamma * (1 - d) * target_q
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.copy_(cfg.tau * op.data + (1 - cfg.tau) * tp.data)
        self.epsilon = max(cfg.eps_min, self.epsilon * cfg.eps_decay)
        return loss.item()


def render(env, ax, scores=None):
    import matplotlib.pyplot as plt
    ax.clear()
    cfg = env.cfg
    ws = cfg.world_size
    ax.set_xlim(0, ws)
    ax.set_ylim(0, ws)
    ax.set_aspect("equal")
    ax.set_title(f"Step {env.step_count}")
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.n_agvs))
    for i in range(cfg.n_agvs):
        x, y = env.positions[i]
        gx, gy = env.goals[i]
        c = colors[i]
        circle = plt.Circle((x, y), cfg.agv_radius, color=c, alpha=0.6)
        ax.add_patch(circle)
        ax.plot(gx, gy, marker="*", color=c, markersize=12)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8, fontweight="bold")
    if scores is not None:
        ax.text(0.02, 0.98,
                "scores: " + " ".join(f"{s:.0f}" for s in scores),
                transform=ax.transAxes, va="top", fontsize=7)
    plt.pause(0.01)


def train(cfg=None):
    if cfg is None:
        cfg = CONFIG()
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    env = MultiAGVEnv(cfg)
    state_dim = env.reset().shape[1]
    action_dim = 5
    agent = DuelingDQNAgent(state_dim, action_dim, cfg)
    all_scores = []
    best_avg = -float("inf")
    for ep in range(1, cfg.episodes + 1):
        states = env.reset()
        ep_scores = np.zeros(cfg.n_agvs)
        step = 0
        loss = None
        while True:
            actions = agent.act(states)
            next_states, rewards, done = env.step(actions)
            ep_scores += rewards
            step += 1
            for i in range(cfg.n_agvs):
                agent.buffer.push(states[i], actions[i], rewards[i],
                                  next_states[i], float(env.done_agents[i]))
            loss = agent.update()
            states = next_states
            if done:
                break
        avg_score = ep_scores.mean()
        all_scores.append(avg_score)
        running_avg = np.mean(all_scores[-100:])
        if ep % cfg.print_interval == 0:
            loss_str = f"loss: {loss:.4f}" if loss is not None else "loss: -"
            print(f"ep {ep:5d} | avg_score: {avg_score:7.1f} | "
                  f"avg100: {running_avg:7.1f} | eps: {agent.epsilon:.3f} | "
                  f"steps: {step:3d} | {loss_str}")
        if ep % cfg.render_interval == 0:
            render(env, ax, ep_scores)
        if len(all_scores) >= 100 and running_avg > best_avg:
            best_avg = running_avg
            torch.save(agent.online.state_dict(), r"D:\dueling_dqn_agv_best.pth")
            print(f"  -> saved (avg100={best_avg:.1f})")
    plt.ioff()
    plt.show()
    print(f"\nDone. best avg100: {best_avg:.1f}")
    return agent, env


def demo(model_path=r"D:\dueling_dqn_agv_best.pth", n_episodes=3):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    cfg = CONFIG()
    env = MultiAGVEnv(cfg)
    state_dim = env.reset().shape[1]
    agent = DuelingDQNAgent(state_dim, 5, cfg)
    agent.online.load_state_dict(torch.load(model_path, map_location=cfg.device))
    agent.online.eval()
    for _ in range(n_episodes):
        states = env.reset()
        scores = np.zeros(cfg.n_agvs)
        while True:
            actions = agent.act(states, eval_mode=True)
            render(env, ax, scores)
            states, rewards, done = env.step(actions)
            scores += rewards
            if done:
                break
        print(f"demo scores: {scores}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    agent, env = train()
    demo()
