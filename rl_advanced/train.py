import json
from collections import defaultdict

import numpy as np
import torch

from config import Config
from env import MultiAGVEnv
from agent import DuelingDQNAgent


class MetricsTracker:
    """训练指标收集与统计"""

    def __init__(self):
        self.history = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def avg(self, key, n=100):
        vals = self.history[key][-n:]
        return np.mean(vals) if vals else 0.0

    def save(self, path):
        with open(path, "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f)


def render(env, ax, scores=None, priorities=None):
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
        marker = "D" if (priorities is not None and priorities[i]) else "o"
        ax.plot(x, y, marker=marker, color=c, markersize=10)
        ax.plot(gx, gy, marker="*", color=c, markersize=14)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, fontweight="bold")

    # 动态障碍物
    if cfg.n_dynamic_obs > 0:
        for k in range(cfg.n_dynamic_obs):
            ox, oy = env.obs_positions[k]
            circle = plt.Circle((ox, oy), cfg.obs_radius, color="red", alpha=0.5)
            ax.add_patch(circle)

    if scores is not None:
        ax.text(0.02, 0.98,
                "scores: " + " ".join(f"{s:.0f}" for s in scores),
                transform=ax.transAxes, va="top", fontsize=7)
    plt.pause(0.01)


def apply_curriculum(cfg: Config, episode: int):
    """根据当前 episode 应用课程学习阶段配置"""
    if not cfg.curriculum:
        return

    cumulative = 0
    for stage in cfg.curriculum_stages:
        cumulative += stage["episodes"]
        if episode <= cumulative:
            for key in ["n_agvs", "n_dynamic_obs", "enable_priority"]:
                setattr(cfg, key, stage[key])
            return

    # 超出课程表，用最后阶段
    last = cfg.curriculum_stages[-1]
    for key in ["n_agvs", "n_dynamic_obs", "enable_priority"]:
        setattr(cfg, key, last[key])


def train(cfg=None, headless=True):
    if cfg is None:
        cfg = Config()

    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not headless:
        plt.ion()
        fig, (ax_env, ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax_env = ax_metrics = None

    metrics = MetricsTracker()
    best_avg = -float("inf")

    # 用初始配置建环境以获取 state_dim
    apply_curriculum(cfg, 1)
    env = MultiAGVEnv(cfg)
    state_dim = env._get_obs_dim()
    action_dim = 5
    agent = DuelingDQNAgent(state_dim, action_dim, cfg)

    # 首次 reset 后可能 state_dim 变了（课程前期参数不同）
    # 训练中如果课程切换导致 state_dim 变化，重新创建 agent
    current_state_dim = state_dim

    global_ep = 0
    for stage_idx, stage in enumerate(cfg.curriculum_stages if cfg.curriculum else []):
        stage_eps = stage["episodes"]
        # 应用阶段配置
        for key in ["n_agvs", "n_dynamic_obs", "enable_priority"]:
            setattr(cfg, key, stage[key])

        print(f"\n{'='*50}")
        print(f"课程阶段 {stage_idx+1}: AGV={cfg.n_agvs}, "
              f"动态障碍={cfg.n_dynamic_obs}, 优先级={'开' if cfg.enable_priority else '关'}")
        print(f"{'='*50}")

        for ep_in_stage in range(1, stage_eps + 1):
            global_ep += 1
            env = MultiAGVEnv(cfg)
            new_dim = env._get_obs_dim()

            # state_dim 变化时重建 agent（保留 buffer 和模型权重）
            if new_dim != current_state_dim:
                print(f"  -> state_dim 变化: {current_state_dim} -> {new_dim}，重建网络")
                old_online = agent.online
                old_target = agent.target
                old_buffer = agent.buffer
                old_opt = agent.optimizer

                agent = DuelingDQNAgent(new_dim, action_dim, cfg)
                # 尝试迁移可复用的特征层权重（前几层维度不变的部分跳过）
                agent.buffer = old_buffer  # 但 buffer 存的是旧维度...直接清空
                if agent.use_per:
                    from network import PrioritizedReplayBuffer
                    agent.buffer = PrioritizedReplayBuffer(cfg.buffer_capacity, cfg.per_alpha)
                current_state_dim = new_dim

            states = env.reset()
            ep_rewards = np.zeros(cfg.n_agvs)
            ep_collisions = 0
            ep_goals = 0
            ep_steps = 0
            loss_val = None

            while True:
                actions = agent.act(states)
                next_states, rewards, done, info = env.step(actions)
                ep_rewards += rewards
                ep_collisions += info["collision"] + info["obs_collision"]
                ep_goals += info["goal_reached"]
                ep_steps += 1

                for i in range(cfg.n_agvs):
                    agent.push(states[i], actions[i], rewards[i],
                               next_states[i], float(env.done_agents[i]))
                loss_val = agent.update()
                states = next_states
                if done:
                    break

            avg_reward = ep_rewards.mean()
            metrics.add(
                avg_reward=avg_reward,
                collisions=ep_collisions,
                goals=ep_goals,
                steps=ep_steps,
                epsilon=agent.epsilon,
                loss=loss_val or 0,
            )

            running_avg = metrics.avg("avg_reward", 100)

            if global_ep % cfg.print_interval == 0:
                ls = f"loss: {loss_val:.4f}" if loss_val else "loss: -"
                col_rate = np.mean(metrics.history["collisions"][-50:])
                goal_rate = np.mean(metrics.history["goals"][-50:])
                print(f"ep {global_ep:5d} | avg_r: {avg_reward:7.1f} | "
                      f"avg100: {running_avg:7.1f} | "
                      f"碰撞: {col_rate:.1f} | 到达: {goal_rate:.1f} | "
                      f"steps: {ep_steps:3d} | {ls}")

            if global_ep % cfg.render_interval == 0 and not headless:
                render(env, ax_env, ep_rewards, env.priority)
                ax_metrics.clear()
                h = metrics.history
                if len(h["avg_reward"]) > 1:
                    ax_metrics.plot(h["avg_reward"], alpha=0.3, color="blue")
                    if len(h["avg_reward"]) >= 10:
                        smooth = np.convolve(h["avg_reward"], np.ones(50)/50, mode="valid")
                        ax_metrics.plot(range(49, len(h["avg_reward"])), smooth, color="blue", label="Reward")
                ax_metrics.set_title("Training Progress")
                ax_metrics.set_xlabel("Episode")
                ax_metrics.legend()
                ax_metrics.grid(True, alpha=0.3)
                plt.pause(0.01)

            if len(metrics.history["avg_reward"]) >= 100 and running_avg > best_avg:
                best_avg = running_avg
                torch.save({
                    "model_state": agent.online.state_dict(),
                    "state_dim": current_state_dim,
                    "action_dim": action_dim,
                    "config": {k: v for k, v in cfg.__dict__.items()
                               if not k.startswith("_") and not callable(v)},
                }, cfg.save_path)
                print(f"  -> saved (avg100={best_avg:.1f})")

    if not headless:
        plt.ioff()
        plt.show()
    metrics.save(cfg.save_path.replace(".pth", "_metrics.json"))
    print(f"\n训练完成。最佳 avg100: {best_avg:.1f}")
    print(f"指标已保存至: {cfg.save_path}")
    return agent, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="显示可视化窗口（需桌面环境）")
    args = parser.parse_args()

    cfg = Config()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA 不可用，使用 CPU 训练")

    agent, metrics = train(cfg, headless=not args.show)
