"""将 rl_advanced/ 所有文件合并为 D:/carAI_project.py"""
import os

PROJECT = "e:/carAI"
OUT = "D:/carAI_project.py"


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


header = """\
#!/usr/bin/env python3
\"\"\"
================================================================================
 AI大模型驱动的智能汽车自动驾驶决策优化与车路协同交互系统
 强化学习决策模块（完整版）

 用法:
   python D:/carAI_project.py              # 训练
   python D:/carAI_project.py --test        # 测试已保存模型
   python D:/carAI_project.py --show        # 训练（显示可视化窗口）

 整合内容:
   - 多AGV环境（动态障碍物 + 优先通行机制）
   - Dueling DQN 网络（NoisyNet + Prioritized Experience Replay）
   - Double DQN + SmoothL1Loss（Huber）
   - 课程学习训练流程 + 多指标追踪
   - 测试/演示入口

 原始项目: e:/carAI/rl_advanced/
 整合日期: 2026-05-23
================================================================================
\"\"\"

import json
import math
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""

# read train.py and fix
raw_train = read(f"{PROJECT}/rl_advanced/train.py")
# Remove module-level imports and the __name__ block at bottom
lines = raw_train.split("\n")
clean_lines = []
in_main_block = False
for line in lines:
    if line.startswith("if __name__"):
        in_main_block = True
        continue
    if in_main_block:
        continue
    if line.startswith("import ") or line.startswith("from "):
        continue
    clean_lines.append(line)
train_fixed = "\n".join(clean_lines)
# rename render -> render_env
train_fixed = train_fixed.replace("def render(", "def render_env(")
train_fixed = train_fixed.replace("render(env,", "render_env(env,")

# assemble
parts = [
    header,
    "# ========================================================================\n#  config.py\n# ========================================================================\n",
    read(f"{PROJECT}/rl_advanced/config.py"),
    "\n# ========================================================================\n#  network.py - NoisyNet + DuelingDQN + PrioritizedReplayBuffer\n# ========================================================================\n",
    read(f"{PROJECT}/rl_advanced/network.py"),
    "\n# ========================================================================\n#  env.py - MultiAGVEnv (动态障碍物 + 优先通行)\n# ========================================================================\n",
    read(f"{PROJECT}/rl_advanced/env.py"),
    "\n# ========================================================================\n#  agent.py - DuelingDQNAgent\n# ========================================================================\n",
    read(f"{PROJECT}/rl_advanced/agent.py"),
    "\n# ========================================================================\n#  train.py - 课程学习 + 指标追踪\n# ========================================================================\n",
    train_fixed,
    "\n# ========================================================================\n#  测试\n# ========================================================================\n",
    """
def test(model_path="D:/carAI_best_model.pth", n_trials=5):
    \"\"\"加载模型并测试\"\"\"
    cfg = Config()
    if not torch.cuda.is_available():
        cfg.device = "cpu"

    state_dim = compute_max_state_dim(cfg)
    agent = DuelingDQNAgent(state_dim, 5, cfg)

    last = cfg.curriculum_stages[-1]
    for key in ["n_agvs", "n_dynamic_obs", "enable_priority"]:
        setattr(cfg, key, last[key])

    try:
        ckpt = torch.load(model_path, map_location=agent.device, weights_only=True)
        agent.online.load_state_dict(ckpt["model_state"])
        agent.online.eval()
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}, 使用未训练模型测试")
        agent.online.eval()

    print(f"\\n测试配置: AGV={cfg.n_agvs}, 动态障碍={cfg.n_dynamic_obs}, "
          f"优先级={'开' if cfg.enable_priority else '关'}")
    print("=" * 60)

    total_goals = 0
    for trial in range(1, n_trials + 1):
        env = MultiAGVEnv(cfg, fixed_state_dim=state_dim)
        states = env.reset()
        scores = np.zeros(cfg.n_agvs)
        while True:
            actions = agent.act(states, eval_mode=True)
            states, rewards, done, info = env.step(actions)
            scores += rewards
            if done:
                break
        total_goals += info["goal_reached"]
        print(f"  trial {trial}: scores={[f'{s:.0f}' for s in scores]}, "
              f"goals={info['goal_reached']}, collisions={info['collision']}")

    print(f"\\n平均到达率: {total_goals / n_trials:.1f} / {cfg.n_agvs}")
""",
    "\n# ========================================================================\n#  入口\n# ========================================================================\n",
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI大模型驱动的智能汽车自动驾驶决策优化 — RL模块")
    parser.add_argument("--test", action="store_true", help="仅测试已保存模型")
    parser.add_argument("--show", action="store_true", help="显示可视化窗口")
    parser.add_argument("--model", type=str, default="D:/carAI_best_model.pth",
                        help="测试用模型路径")
    parser.add_argument("--trials", type=int, default=5, help="测试次数")
    args = parser.parse_args()

    if args.test:
        test(args.model, args.trials)
    else:
        cfg = Config()
        if not torch.cuda.is_available():
            cfg.device = "cpu"
            print("CUDA 不可用，使用 CPU 训练")
        train(cfg, headless=not args.show)
""",
]

merged = "\n".join(parts)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(merged)

print(f"OK -> {OUT}")
print(f"  size: {os.path.getsize(OUT)} bytes")
print(f"  lines: {len(merged.splitlines())}")
