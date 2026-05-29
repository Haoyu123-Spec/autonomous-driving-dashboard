#!/usr/bin/env python3
"""分析训练结果：曲线、Q值、策略可视化"""
import re
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
from collections import defaultdict

from dueling_dqn_agv import CONFIG, MultiAGVEnv, DuelingDQNAgent

# ─── 1. 解析训练日志 ───────────────────────────────────────────
def parse_log(path):
    """从训练输出中提取每10集的指标"""
    data = defaultdict(list)
    pattern = re.compile(
        r"ep\s+(\d+)\s+\|\s+avg_score:\s+([-\d.]+)\s+\|\s+"
        r"avg100:\s+([-\d.]+)\s+\|\s+eps:\s+([-\d.]+)\s+\|\s+"
        r"steps:\s+(\d+)\s+\|\s+loss:\s+([-\d.]+)"
    )
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data["ep"].append(int(m.group(1)))
                data["score"].append(float(m.group(2)))
                data["avg100"].append(float(m.group(3)))
                data["eps"].append(float(m.group(4)))
                data["steps"].append(int(m.group(5)))
                data["loss"].append(float(m.group(6)))
    return data


def plot_curves(data, save_path):
    """4合1训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dueling DQN — 训练曲线", fontsize=14, fontweight="bold")

    eps_arr = np.array(data["ep"])

    # 左上：avg_score + avg100
    ax = axes[0, 0]
    ax.plot(eps_arr, data["score"], alpha=0.25, color="steelblue", linewidth=0.8)
    ax.plot(eps_arr, data["avg100"], color="darkorange", linewidth=2, label="avg100")
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(data["score"], 30)
    ax.plot(eps_arr, smooth, color="steelblue", linewidth=1.5, label="score (smooth 30)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.set_title("奖励曲线")
    ax.grid(True, alpha=0.3)

    # 右上：loss
    ax = axes[0, 1]
    loss_arr = np.array(data["loss"])
    ax.plot(eps_arr, loss_arr, alpha=0.3, color="crimson", linewidth=0.6)
    loss_smooth = uniform_filter1d(loss_arr, 20)
    ax.plot(eps_arr, loss_smooth, color="crimson", linewidth=1.5, label="loss (smooth 20)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.set_title("损失曲线 (SmoothL1Loss)")
    ax.grid(True, alpha=0.3)

    # 左下：steps
    ax = axes[1, 0]
    ax.plot(eps_arr, data["steps"], alpha=0.25, color="seagreen", linewidth=0.8)
    steps_smooth = uniform_filter1d(data["steps"], 30)
    ax.plot(eps_arr, steps_smooth, color="seagreen", linewidth=1.5, label="steps (smooth 30)")
    ax.axhline(y=200, color="gray", linestyle="--", alpha=0.5, label="max=200")
    ax.set_ylabel("Steps")
    ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.set_title("每集步数")
    ax.grid(True, alpha=0.3)

    # 右下：epsilon
    ax = axes[1, 1]
    ax.plot(eps_arr, data["eps"], color="purple", linewidth=1.5)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="eps_min=0.05")
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.set_title("探索率衰减")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"曲线已保存: {save_path}")


# ─── 2. Q 值分析 ───────────────────────────────────────────────
def analyze_q_values(model_path="e:/carAI/dueling_dqn_agv_best.pth"):
    """分析最优模型的 Q 值分布和价值流"""
    cfg = CONFIG()
    env = MultiAGVEnv(cfg)
    state_dim = env.reset().shape[1]
    agent = DuelingDQNAgent(state_dim, 5, cfg)
    agent.online.load_state_dict(torch.load(model_path, map_location=cfg.device))
    agent.online.eval()

    # 收集多集的 Q 值
    all_q = []
    all_v = []
    all_a = []
    for _ in range(20):
        states = env.reset()
        for _ in range(200):
            t = torch.FloatTensor(states).to(cfg.device)
            with torch.no_grad():
                f = agent.online.feature(t)
                v = agent.online.value(f)
                a = agent.online.advantage(f)
                q = v + a - a.mean(dim=1, keepdim=True)
            all_q.append(q.cpu().numpy())
            all_v.append(v.cpu().numpy())
            all_a.append(a.cpu().numpy())

            actions = agent.act(states, eval_mode=True)
            states, _, done = env.step(actions)
            if done:
                break

    all_q = np.concatenate(all_q, axis=0)  # (N, 5)
    all_v = np.concatenate(all_v, axis=0)  # (N, 1)
    all_a = np.concatenate(all_a, axis=0)  # (N, 5)

    return all_q, all_v, all_a


def plot_q_analysis(all_q, all_v, all_a, save_path):
    """Q 值分布 + Value/Advantage 分解"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Q 值分布与价值流分解", fontsize=14, fontweight="bold")

    # Q 值分布
    ax = axes[0]
    actions = ["Stay", "Right", "Left", "Up", "Down"]
    colors = ["gray", "steelblue", "coral", "seagreen", "darkorange"]
    for i, (a_name, c) in enumerate(zip(actions, colors)):
        ax.hist(all_q[:, i], bins=40, alpha=0.4, color=c, label=a_name, density=True)
    ax.set_xlabel("Q value")
    ax.set_ylabel("Density")
    ax.set_title("各动作 Q 值分布")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Value 分布
    ax = axes[1]
    ax.hist(all_v.flatten(), bins=40, color="steelblue", alpha=0.7, density=True)
    ax.set_xlabel("V(s)")
    ax.set_ylabel("Density")
    ax.set_title("状态价值 V(s) 分布")
    ax.grid(True, alpha=0.3)

    # Advantage 分布
    ax = axes[2]
    for i, (a_name, c) in enumerate(zip(actions, colors)):
        ax.hist(all_a[:, i], bins=40, alpha=0.3, color=c, label=a_name, density=True)
    ax.set_xlabel("A(s,a)")
    ax.set_ylabel("Density")
    ax.set_title("优势函数 A(s,a) 分布")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Q 值分析已保存: {save_path}")

    # 打印统计
    print("\n─── Q 值统计 ───")
    for i, a_name in enumerate(actions):
        print(f"  {a_name:6s}: mean={all_q[:, i].mean():.2f}, std={all_q[:, i].std():.2f}, "
              f"min={all_q[:, i].min():.2f}, max={all_q[:, i].max():.2f}")
    print(f"  V(s): mean={all_v.mean():.2f}, std={all_v.std():.2f}")
    best_action = np.argmax(all_q.mean(axis=0))
    print(f"  最优动作: {actions[best_action]} (avg Q={all_q[:, best_action].mean():.2f})")


# ─── 3. 轨迹可视化 ──────────────────────────────────────────────
def render_trajectory(model_path="e:/carAI/dueling_dqn_agv_best.pth",
                      save_path="e:/carAI/trajectory_analysis.png"):
    """渲染一个完整 episode 的轨迹"""
    cfg = CONFIG()
    env = MultiAGVEnv(cfg)
    state_dim = env.reset().shape[1]
    agent = DuelingDQNAgent(state_dim, 5, cfg)
    agent.online.load_state_dict(torch.load(model_path, map_location=cfg.device))
    agent.online.eval()

    # 随机找一集
    for attempt in range(20):
        states = env.reset()
        positions_traj = [[] for _ in range(cfg.n_agvs)]
        goals = env.goals.copy()
        actions_seq = []
        episode_rewards = np.zeros(cfg.n_agvs)
        ep_steps = 0
        collisions = 0

        while True:
            for i in range(cfg.n_agvs):
                positions_traj[i].append(env.positions[i].copy())

            actions = agent.act(states, eval_mode=True)
            actions_seq.append(actions.copy())
            states, rewards, done = env.step(actions)
            episode_rewards += rewards
            ep_steps += 1

            # 检测碰撞
            for i in range(cfg.n_agvs):
                for j in range(i + 1, cfg.n_agvs):
                    if not env.done_agents[i] and not env.done_agents[j]:
                        d = np.linalg.norm(env.positions[i] - env.positions[j])
                        if d < 2 * cfg.agv_radius:
                            collisions += 1

            if done:
                break

        # 只保留有到达目标的轨迹
        goals_reached = sum(1 for i in range(cfg.n_agvs)
                          if np.linalg.norm(env.positions[i] - goals[i]) < cfg.step_size)
        if goals_reached >= 2 and ep_steps > 20:
            break
    else:
        print("警告：20次尝试未找到有足够目标到达的轨迹，使用最后一集")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"策略轨迹可视化 (steps={ep_steps}, goals={goals_reached}/{cfg.n_agvs})",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, cfg.n_agvs))
    action_names = ["Stay", "→", "←", "↑", "↓"]

    # 左图：完整轨迹
    ax = axes[0]
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_title("AGV 路径轨迹")
    ax.grid(True, alpha=0.3)

    for i in range(cfg.n_agvs):
        traj = np.array(positions_traj[i])
        c = colors[i]
        # 轨迹线
        ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.4, linewidth=1, linestyle="-")
        # 起点
        ax.plot(traj[0, 0], traj[0, 1], marker="o", color=c, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
        # 终点
        ax.plot(traj[-1, 0], traj[-1, 1], marker="s", color=c, markersize=10, markeredgecolor="black", markeredgewidth=0.5)
        # 目标
        gx, gy = goals[i]
        ax.plot(gx, gy, marker="*", color=c, markersize=16, markeredgecolor="black", markeredgewidth=0.5)
        # 标注
        ax.annotate(f"AGV{i}", (traj[0, 0], traj[0, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color=c, fontweight="bold")

    # 右图：动作分布
    ax = axes[1]
    all_actions = np.concatenate(actions_seq)
    action_counts = np.bincount(all_actions, minlength=5)
    bars = ax.bar(action_names, action_counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, action_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("动作分布")
    ax.set_ylabel("次数")
    ax.grid(True, alpha=0.3, axis="y")

    # 额外信息
    info_text = (f"Rewards: {' '.join(f'{r:.1f}' for r in episode_rewards)}\n"
                 f"Collisions: {collisions}\n"
                 f"Steps: {ep_steps}")
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"轨迹可视化已保存: {save_path}")
    print(f"  Steps: {ep_steps}, Goals: {goals_reached}/{cfg.n_agvs}, Collisions: {collisions}")
    print(f"  Rewards: {[f'{r:.1f}' for r in episode_rewards]}")


# ─── 主函数 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    log_path = r"C:\Users\33940\AppData\Local\Temp\claude\e--carAI\a91e73fb-9cc5-446c-b95b-07f6013f4487\tasks\bygbyajyh.output"
    model_path = "e:/carAI/dueling_dqn_agv_best.pth"
    out_dir = "e:/carAI"

    # 1. 训练曲线
    print("=" * 50)
    print("1. 绘制训练曲线...")
    data = parse_log(log_path)
    print(f"   解析到 {len(data['ep'])} 条记录 (ep {min(data['ep'])} ~ {max(data['ep'])})")
    plot_curves(data, os.path.join(out_dir, "training_curves.png"))

    # 2. Q 值分析
    print("\n" + "=" * 50)
    print("2. Q 值分析...")
    all_q, all_v, all_a = analyze_q_values(model_path)
    plot_q_analysis(all_q, all_v, all_a, os.path.join(out_dir, "q_value_analysis.png"))

    # 3. 轨迹可视化
    print("\n" + "=" * 50)
    print("3. 策略轨迹可视化...")
    render_trajectory(model_path, os.path.join(out_dir, "trajectory_analysis.png"))

    print("\n" + "=" * 50)
    print("分析完成！生成文件：")
    print(f"  {out_dir}/training_curves.png")
    print(f"  {out_dir}/q_value_analysis.png")
    print(f"  {out_dir}/trajectory_analysis.png")
