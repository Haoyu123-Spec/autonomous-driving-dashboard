import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """NoisyNet 线性层：权重和偏置带可学习高斯噪声"""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init * bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init * bound)

    def _sample_noise(self):
        weight_noise = self._f(self.weight_epsilon)
        bias_noise = self._f(self.bias_epsilon)
        return weight_noise, bias_noise

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight_noise, bias_noise = self._sample_noise()
            weight = self.weight_mu + self.weight_sigma * weight_noise
            bias = self.bias_mu + self.bias_sigma * bias_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def sample(self):
        """重新采样噪声"""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, use_noisy=True):
        super().__init__()
        Linear = NoisyLinear if use_noisy else nn.Linear
        linear_kwargs = {} if use_noisy else {}

        self.use_noisy = use_noisy
        self.feature = nn.Sequential(
            Linear(state_dim, hidden, **linear_kwargs), nn.ReLU(),
            Linear(hidden, hidden, **linear_kwargs), nn.ReLU(),
        )
        self.value = nn.Sequential(
            Linear(hidden, hidden // 2, **linear_kwargs), nn.ReLU(),
            Linear(hidden // 2, 1, **linear_kwargs),
        )
        self.advantage = nn.Sequential(
            Linear(hidden, hidden // 2, **linear_kwargs), nn.ReLU(),
            Linear(hidden // 2, action_dim, **linear_kwargs),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)

    def sample_noise(self):
        if self.use_noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample()


class PrioritizedReplayBuffer:
    """优先经验回放，按 TD-error 加权采样"""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((s, a, r, ns, d))
            self.size += 1
        else:
            self.buffer[self.pos] = (s, a, r, ns, d)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n, beta=0.4):
        if self.size < n:
            return None

        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, n, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        s, a, r, ns, d = zip(*samples)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d),
                indices,
                torch.FloatTensor(weights))

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.size
