import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import trackio as wandb

from agent import DuelingDQNAgent
from config import Config
from env import MultiAGVEnv, compute_max_state_dim


class MetricsTracker:
    """训练指标收集与统计，同时写入 Trackio 和内存缓存。"""

    def __init__(self, use_trackio: bool = True):
        self.history = defaultdict(list)
        self._recent = {}
        self.use_trackio = use_trackio

    def add(self, step: int, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)
        self._recent = kwargs
        if self.use_trackio:
            wandb.log(kwargs, step=step)

    def avg(self, key, n=100):
        vals = self.history[key][-n:]
        return np.mean(vals) if vals else 0.0

    def save(self, path):
        with open(path, "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f)


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
        ax.plot(x, y, marker="o", color=c, markersize=10)
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
            for key in ["n_agvs", "n_dynamic_obs", "emergency_enabled"]:
                if key in stage:
                    setattr(cfg, key, stage[key])
            return

    # 超出课程表，用最后阶段
    last = cfg.curriculum_stages[-1]
    for key in ["n_agvs", "n_dynamic_obs", "emergency_enabled"]:
        if key in last:
            setattr(cfg, key, last[key])


def train(cfg=None, headless=True, run_name: str = None):
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

    # 初始化 Trackio 实验追踪
    run_name = run_name or f"rl-advanced-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="rl-advanced", name=run_name)
    wandb.config.update({k: v for k, v in cfg.__dict__.items()
                         if not k.startswith("_") and not callable(v)})

    metrics = MetricsTracker(use_trackio=True)
    best_avg = -float("inf")
    best_ep = 0
    early_stop_patience = 600  # 连续无改善则早停

    # 用全局最大 obs_dim 统一网络输入维度
    state_dim = compute_max_state_dim(cfg)
    action_dim = 5
    agent = DuelingDQNAgent(state_dim, action_dim, cfg)
    start_stage_idx = 0
    start_ep_in_stage = 1

    # 断点续训
    if cfg.resume_path and cfg.resume_path.strip():
        import os as _os
        if _os.path.exists(cfg.resume_path):
            ckpt = torch.load(cfg.resume_path, map_location=agent.device)
            agent.online.load_state_dict(ckpt["model_state"])
            agent.target.load_state_dict(ckpt["model_state"])
            # 从已保存的指标恢复
            metrics_path = cfg.resume_path.replace(".pth", "_metrics.json")
            if _os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    saved_metrics = json.load(f)
                for k, v in saved_metrics.items():
                    metrics.history[k] = v
                best_avg = max(saved_metrics.get("avg_reward", [-float("inf")]))
                print(f"从 {cfg.resume_path} 恢复，已有 {len(metrics.history.get('avg_reward', []))} 条记录，best_avg={best_avg:.1f}")
            # 计算从哪个阶段开始
            total_done = len(metrics.history.get("avg_reward", []))
            cumulative = 0
            for si, stage in enumerate(cfg.curriculum_stages if cfg.curriculum else []):
                if total_done < cumulative + stage["episodes"]:
                    start_stage_idx = si
                    start_ep_in_stage = total_done - cumulative + 1
                    break
                cumulative += stage["episodes"]
            else:
                print("已有训练已完成所有课程阶段")
                start_stage_idx = len(cfg.curriculum_stages)
                start_ep_in_stage = 1
        else:
            print(f"resume_path 不存在: {cfg.resume_path}，从头训练")

    print(f"Fixed state_dim: {state_dim}")

    global_ep = len(metrics.history.get("avg_reward", []))
    early_stopped = False
    for stage_idx, stage in enumerate(cfg.curriculum_stages if cfg.curriculum else []):
        if stage_idx < start_stage_idx:
            continue  # 跳过已完成阶段
        stage_eps = stage["episodes"]
        # 应用阶段配置
        for key in ["n_agvs", "n_dynamic_obs", "emergency_enabled"]:
            if key in stage:
                setattr(cfg, key, stage[key])

        print(f"\n{'='*50}")
        print(f"课程阶段 {stage_idx+1}/{len(cfg.curriculum_stages)}: AGV={cfg.n_agvs}, "
              f"动态障碍={cfg.n_dynamic_obs}, 紧急订单={'开' if cfg.emergency_enabled else '关'}")
        print(f"{'='*50}")

        # 每阶段重置 epsilon，确保后期阶段仍有探索能力
        if not agent.use_noisy and (stage_idx == 0 or stage_idx >= start_stage_idx):
            agent.epsilon = cfg.epsilon
            print(f"  epsilon 重置为 {agent.epsilon:.2f}")

        for _ep_in_stage in range(start_ep_in_stage if stage_idx == start_stage_idx else 1,
                                   stage_eps + 1):
            global_ep += 1
            env = MultiAGVEnv(cfg, fixed_state_dim=state_dim)

            states = env.reset()
            ep_rewards = np.zeros(cfg.n_agvs)
            ep_collisions = 0
            ep_goals = 0
            ep_emergency = 0
            ep_steps = 0
            loss_val = None

            while True:
                actions = agent.act(states)
                next_states, rewards, done, info = env.step(actions)
                ep_rewards += rewards
                ep_collisions += info["collision"] + info["obs_collision"] + info.get("wall_collision", 0)
                ep_goals += info["goal_reached"]
                ep_emergency += info.get("emergency_completed", 0)
                ep_steps += 1

                for i in range(cfg.n_agvs):
                    agent.push(states[i], actions[i], rewards[i],
                               next_states[i], float(env.done_agents[i]))
                loss_val = agent.update()
                states = next_states
                if done:
                    break

            # 按 episode 衰减 epsilon（而不是每步衰减）
            if not agent.use_noisy:
                agent.epsilon = max(cfg.eps_min, agent.epsilon * cfg.eps_decay)

            avg_reward = ep_rewards.mean()
            metrics.add(
                step=global_ep,
                avg_reward=avg_reward,
                collisions=ep_collisions,
                goals=ep_goals,
                emergency=ep_emergency,
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
                render(env, ax_env, ep_rewards)
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
                best_ep = global_ep
                torch.save({
                    "model_state": agent.online.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "config": {k: v for k, v in cfg.__dict__.items()
                               if not k.startswith("_") and not callable(v)},
                }, cfg.save_path)
                print(f"  -> saved (avg100={best_avg:.1f}, ep={global_ep})")

            # 早停
            if (len(metrics.history["avg_reward"]) >= 100
                    and global_ep - best_ep > early_stop_patience
                    and best_ep > 0):
                print(f"\n早停：{early_stop_patience} 集无改善 (best_ep={best_ep}, best_avg={best_avg:.1f})")
                early_stopped = True
                break

        if early_stopped:
            break

    if not headless:
        plt.ioff()
        plt.show()
    metrics.save(cfg.save_path.replace(".pth", "_metrics.json"))
    wandb.finish()
    print(f"\n训练完成。最佳 avg100: {best_avg:.1f} at ep {best_ep}")
    print(f"指标已保存至: {cfg.save_path}")
    return agent, metrics


def main():
    """Hydra 入口点 —— 支持命令行覆盖配置。

    示例:
        python train.py world.n_agvs=5 dqn.lr=1e-4 training.episodes=5000
        python train.py --show dqn.use_attention=false
    """
    import hydra
    from omegaconf import DictConfig

    # 解析 --show 参数（Hydra 不处理自定义 CLI 参数）
    show_idx = -1
    for i, a in enumerate(sys.argv):
        if a == "--show":
            show_idx = i
            break
    headless = show_idx < 0

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def _hydra_entry(hydra_cfg: DictConfig):
        cfg = Config.from_hydra(hydra_cfg)
        if not torch.cuda.is_available():
            cfg.device = "cpu"
            print("CUDA 不可用，使用 CPU 训练")

        print(f"\n训练配置: AGV={cfg.n_agvs}, 障碍物={cfg.n_dynamic_obs}, "
              f"课程学习={'开' if cfg.curriculum else '关'}")
        return train(cfg, headless=headless)

    return _hydra_entry()


if __name__ == "__main__":
    main()
