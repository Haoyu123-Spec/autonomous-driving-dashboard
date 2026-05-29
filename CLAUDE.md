# CLAUDE.md

## 项目概述
AI大模型驱动的智能汽车自动驾驶决策优化与车路协同交互系统 — 探索端到端大模型（VLA）、多模态感知融合、强化学习决策、车路协同（V2X）等下一代自动驾驶技术。

## 技术栈
- **Python** (PyTorch, NumPy, Matplotlib) — 强化学习训练与仿真
- **HTML/Tailwind CSS/Chart.js/Font Awesome** — 可视化仪表盘

## 项目文件
- `Code_20260510.html` — 技术方案展示页面（仪表盘），包含技术趋势、架构、创新点等
- `dueling_dqn_agv.py` — 多AGV碰撞避免的 Dueling DQN 强化学习实现，对应决策层的强化学习模块

## 关键模块（HTML页面结构）
- 技术发展趋势 / 挑战分析
- 五层技术架构：感知层 → 预测层 → 决策层 → 控制层 → 车路协同层
- 三大创新点：多模态感知融合（3D占用网络/TPVFormer）、VLA端到端决策、车路云协同
- 指标体系与实验对比

## 关键模块（Python RL）
- `MultiAGVEnv` — 多AGV网格世界环境，支持碰撞检测、目标到达判定
- `DuelingDQN` — Dueling DQN 网络（feature → value + advantage 分支）
- `DuelingDQNAgent` — 智能体，含经验回放、Double DQN、软更新
- `train()` / `demo()` — 训练入口与演示入口

## 开发约定
- Python 代码使用 dataclass 管理配置
- 路径使用绝对路径（如 `E:\carAI\dueling_dqn_agv_best.pth`）
- 模型保存/加载使用 PyTorch 标准方式
