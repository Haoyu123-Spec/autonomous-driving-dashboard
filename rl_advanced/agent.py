import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import DuelingDQN, PrioritizedReplayBuffer


class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, cfg: Config):
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = (cfg.device if torch.cuda.is_available() and cfg.device == "cuda"
                       else "cpu")

        self.online = DuelingDQN(state_dim, action_dim, cfg.hidden, cfg.use_noisy).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim, cfg.hidden, cfg.use_noisy).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr)

        if cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(cfg.buffer_capacity, cfg.per_alpha)
        else:
            from collections import deque
            self._simple_buf = deque(maxlen=cfg.buffer_capacity)
            self.buffer = None  # 标志位

        self.use_noisy = cfg.use_noisy
        self.use_per = cfg.use_per
        self.update_count = 0  # 用于硬更新计数

        # ε-greedy 仅在不用 NoisyNet 时生效
        self.epsilon = cfg.epsilon if not cfg.use_noisy else 0.0
        self.per_beta = cfg.per_beta

    def act(self, states, eval_mode=False):
        """选择动作。NoisyNet 时用网络噪声探索，否则 ε-greedy。"""
        if not eval_mode and not self.use_noisy and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim, len(states))

        if self.use_noisy and not eval_mode:
            self.online.sample_noise()

        with torch.no_grad():
            t = torch.FloatTensor(np.array(states)).to(self.device)
            return self.online(t).argmax(dim=1).cpu().numpy()

    def push(self, s, a, r, ns, d):
        if self.use_per:
            self.buffer.push(s, a, r, ns, d)
        else:
            self._simple_buf.append((s, a, r, ns, d))

    def update(self):
        cfg = self.cfg

        if self.use_per:
            if len(self.buffer) < cfg.learn_start:
                return None
            result = self.buffer.sample(cfg.batch_size, self.per_beta)
            if result is None:
                return None
            s, a, r, ns, d, indices, weights = result
            s, a, r, ns, d, weights = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device), weights.to(self.device)
            self.per_beta = min(1.0, self.per_beta + cfg.per_beta_increment)
        else:
            if len(self._simple_buf) < cfg.learn_start or len(self._simple_buf) < cfg.batch_size:
                return None
            idx = np.random.choice(len(self._simple_buf), cfg.batch_size, replace=False)
            s_list, a_list, r_list, ns_list, d_list = zip(
                *(self._simple_buf[i] for i in idx))
            s = torch.FloatTensor(np.array(s_list)).to(self.device)
            a = torch.LongTensor(a_list).to(self.device)
            r = torch.FloatTensor(r_list).to(self.device)
            ns = torch.FloatTensor(np.array(ns_list)).to(self.device)
            d = torch.FloatTensor(d_list).to(self.device)
            weights = torch.ones(cfg.batch_size).to(self.device)
            indices = None

        # Double DQN
        if self.use_noisy:
            self.online.sample_noise()
            self.target.sample_noise()

        with torch.no_grad():
            next_a = self.online(ns).argmax(dim=1)
            target_q = self.target(ns).gather(1, next_a.unsqueeze(1)).squeeze()
            y = r + cfg.gamma * (1 - d) * target_q

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        td_errors = (y - q).detach().abs().cpu().numpy()

        loss = (weights * nn.SmoothL1Loss(reduction='none')(q, y)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.update_count += 1
        # 硬更新：每 N 步完全同步 target 网络（更稳定）
        if cfg.target_update_freq > 0:
            if self.update_count % cfg.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        else:
            # 软更新（Polyak averaging）
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.data.copy_(cfg.tau * op.data + (1 - cfg.tau) * tp.data)

        # 更新 PER 优先级
        if self.use_per and indices is not None:
            self.buffer.update_priorities(indices, td_errors + 1e-6)

        # 更新 epsilon
        if not self.use_noisy:
            self.epsilon = max(cfg.eps_min, self.epsilon * cfg.eps_decay)

        return loss.item()
