# Diffusion Policy for Dual-Arm Manipulation - 技术文档

## 📋 目录

1. [项目概述](#1-项目概述)
2. [训练流程架构](#2-训练流程架构)
3. [网络结构详解](#3-网络结构详解)
4. [训练算法](#4-训练算法)
5. [数据集处理](#5-数据集处理)
6. [评估与推理](#6-评估与推理)
7. [配置参数](#7-配置参数)
8. [故障排查](#8-故障排查)

---

## 1. 项目概述

本项目实现了基于 **Diffusion Policy** 的双臂机器人操作任务模仿学习，支持两种输入模态：

- **纯状态版本 (State-only)**: 使用低维状态向量 ( proprioceptive state )
- **视觉+状态版本 (Vision+State)**: 使用相机图像 + 低维状态

### 1.1 核心特性

- ✅ **DDPM 噪声调度**: 支持线性和余弦 beta 调度
- ✅ **EMA (指数滑动平均)**: 可选动态 warmup (power=0.8) 加速收敛
- ✅ **学习率 Warmup**: 按 steps  warmup (默认500 steps)，与官方一致
- ✅ **断点续训**: 支持从任意 checkpoint 恢复训练
- ✅ **GroupNorm**: 全网络使用 GroupNorm 替代 BatchNorm，兼容 EMA
- ✅ **数据增强**: 随机裁剪、ImageNet 归一化

---

## 2. 训练流程架构

### 2.1 流程对比

| 维度 | 纯状态 (`train_new.py`) | 视觉+状态 (`train_with_vision.py`) |
|------|------------------------|-----------------------------------|
| **输入** | 状态向量 (32-50维) | 多相机图像 + 状态 |
| **网络** | `ConditionalUNet1D` | `ConditionalUNet1DWithVision` |
| **编码器** | MLP State Encoder | ResNet18 Vision + MLP State |
| **数据加载** | `RobosuiteDiffusionDataset` | `VisionDiffusionDataset` |
| **计算量** | 小 (~1GB GPU) | 大 (~8-16GB GPU) |
| **收敛速度** | 快 | 慢 (需更多 epoch) |
| **成功率** | 中等 | 更高 |

### 2.2 文件结构

```
项目根目录
├── train_new.py              # 纯状态训练脚本
├── train_with_vision.py      # 视觉+状态训练脚本 (推荐)
├── network.py                # 纯状态网络定义
├── network_with_vision.py    # 视觉+状态网络定义
├── dataset.py                # 纯状态数据集
├── dataset_with_vision.py    # 视觉数据集
├── eval_diffusion.py         # 纯状态评估
├── eval_with_vision.py       # 视觉版本评估
└── config.py                 # 全局配置
```

### 2.3 快速开始

```bash
# 纯状态版本 (快速实验)
python train_new.py --data_dir expert_data --num_epochs 500

# 视觉+状态版本 (推荐，更高成功率)
python train_with_vision.py \
    --data_dir expert_data_with_images \
    --use_ema_warmup \
    --use_lr_warmup \
    --num_epochs 2000

# 断点续训
python train_with_vision.py \
    --resume checkpoints_v2/.../latest_model.pt \
    --use_ema_warmup --use_lr_warmup
```

---

## 3. 网络结构详解

### 3.1 纯状态版本 (`network.py`)

```
输入: states (B, obs_horizon, state_dim)
       ↓
[StateCondEncoder] (MLP: Linear -> SiLU -> Linear)
       ↓
条件特征: cond_dim (默认256)
       ↓
[TimeEncoder] (Sinusoidal Positional Encoding)
       ↓
时间特征: time_dim (默认128)
       ↓
拼接: [cond, time] → global_cond
       ↓
[ConditionalUNet1D]
├─ Input Projection: Conv1d(action_dim → hidden_dim)
├─ Downsampling Path (3层):
│   ├─ ResidualBlock1D + Conv1d(stride=2)
├─ Middle Block: ResidualBlock1D
├─ Upsampling Path (3层):
│   ├─ ConvTranspose1d + ResidualBlock1D + Skip Connection
└─ Output Projection: Conv1d → action_dim
       ↓
输出: noise_pred (B, pred_horizon, action_dim)
```

**关键组件:**
- **FiLM (Feature-wise Linear Modulation)**: 将条件注入特征图
  - `out = x * (1 + scale) + shift`
- **ResidualBlock1D**: Conv1d → GroupNorm → FiLM → SiLU
- **Skip Connections**: U-Net 标准跳跃连接

### 3.2 视觉+状态版本 (`network_with_vision.py`)

```
图像输入: {frontview, agentview} (B, T, 3, 240, 320)
       ↓
[VisionEncoder] (每个相机独立 ResNet18)
├─ ResNet18 Backbone (替换 BN → GN)
├─ Global Average Pooling
└─ 输出: 512-dim 特征
       ↓
多相机拼接 → Linear 投影 → vision_embed (256-dim)

状态输入: states (B, obs_horizon, state_dim)
       ↓
[StateEncoder] (MLP)
└─ state_embed (128-dim)

时间输入: timestep (B,)
       ↓
[TimeEncoder] (Sinusoidal)
└─ time_embed (128-dim)

拼接: [vision_embed, state_embed, time_embed] → cond (512-dim)
       ↓
[ConditionalUNet1DWithVision]
└─ 同纯状态版本的 U-Net 结构
```

**关键特性:**
- **ResNet18**: 预训练=False，替换 BatchNorm 为 GroupNorm
- **多相机融合**: 独立编码后拼接，均值池化时间维度
- **可选状态输入**: `use_states=True/False` 控制是否使用状态

### 3.3 网络参数对比

| 参数 | 纯状态 | 视觉+状态 |
|------|-------|----------|
| 隐藏维度 | [128, 256, 512] | [256, 512, 1024] |
| 条件维度 | 256 | 512 (视觉256+状态128+时间128) |
| 参数量 | ~5M | ~35M (主要 ResNet) |
| 输入归一化 | 手动计算 | ImageNet 预计算 |

---

## 4. 训练算法

### 4.1 Diffusion Policy 核心

**前向过程 (训练):**
```
1. 从数据采样干净动作: x_0 ~ p_data
2. 随机采样时间步: t ~ Uniform(0, T-1)
3. 采样噪声: ε ~ N(0, I)
4. 加噪: x_t = √(α_t) * x_0 + √(1-α_t) * ε
5. 网络预测噪声: ε_θ(x_t, t, obs)
6. 损失: L = MSE(ε_θ, ε)
```

**逆向过程 (推理):**
```
1. 从噪声初始化: x_T ~ N(0, I)
2. for t = T-1 to 0:
   - 预测噪声: ε_θ(x_t, t, obs)
   - 去噪: x_{t-1} = (x_t - β_t*ε_θ) / √(α_t) + σ_t * z
3. 返回 x_0 (预测动作序列)
```

### 4.2 EMA (指数滑动平均)

**标准 EMA:**
```
θ_ema = decay * θ_ema + (1 - decay) * θ
```

**动态 EMA Warmup (本项目实现):**
```python
def get_ema_decay(step):
    # 前 update_after_step 步 decay=0
    step = max(0, step - update_after_step - 1)
    if step <= 0:
        return 0.0
    # power 曲线: 1 - (1 + step)^(-power)
    value = 1 - (1 + step / inv_gamma) ** (-power)
    return clamp(value, min_value, max_value)
```

**power=0.8 衰减曲线:**
| Step | Decay | 说明 |
|------|-------|------|
| 0 | 0.0 | 纯新模型 |
| 10 | 0.48 | 新模型占主导 |
| 100 | 0.88 | 接近稳定 |
| 1000 | 0.99 | 基本锁定 |

### 4.3 学习率调度

**Warmup + Cosine Decay:**
```
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)  # 线性上升
else:
    progress = (step - warmup) / (total - warmup)
    lr = base_lr * 0.5 * (1 + cos(π * progress))  # 余弦下降
```

**推荐配置:**
- Warmup steps: 500 (与官方一致)
- Base LR: 1e-4
- Min LR: 1e-6 (1% of base)

### 4.4 训练技巧

| 技巧 | 效果 | 实现 |
|------|------|------|
| **Gradient Clipping** | 防止梯度爆炸 | max_norm=1.0 |
| **Weight Decay** | 正则化 | AdamW, 1e-6 |
| **Action Normalization** | 稳定训练 | Min-Max 到 [-1, 1] |
| **Observation Normalization** | 零均值单位方差 | 预计算统计量 |

---

## 5. 数据集处理

### 5.1 数据格式

**HDF5 结构 (RoboSuite):**
```
data/
├── demo_0/
│   ├── actions        (T, action_dim)     - 动作序列
│   ├── states         (T, state_dim)      - 状态 (可选)
│   └── observations/
│       ├── frontview_image  (T, H, W, 3) - 前视相机
│       └── agentview_image  (T, H, W, 3) - 代理视角
├── demo_1/
└── ...
```

### 5.2 滑动窗口切片

**参数:**
- `obs_horizon=2`: 观测历史长度
- `pred_horizon=16`: 动作预测长度
- `action_horizon=8`: 动作执行长度 (Receding Horizon)

**采样示例:**
```python
# 从第 i 步开始
obs = states[i : i+obs_horizon]      # 过去2帧
action = actions[i : i+pred_horizon] # 未来16帧
```

### 5.3 数据增强 (仅视觉版)

```python
# 训练时
RandomCrop(216, 288)  # 从 240x320 随机裁剪
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 评估时
CenterCrop(216, 288)  # 中心裁剪
```

---

## 6. 评估与推理

### 6.1 评估流程

```python
1. 加载 EMA 模型 (如果启用)
2. 重置环境 env.reset()
3. 初始化观测缓冲区 ObsBuffer
4. for step in range(max_steps):
   - 获取堆叠观测 (obs_horizon 帧)
   - 扩散采样动作序列 (DDPMScheduler.sample)
   - 执行动作 env.step(action)
   - 检查 success 标志
5. 计算成功率
```

### 6.2 推理加速

**减少扩散步数:**
```python
# 训练: num_steps=100
# 推理: num_steps=50 (加速2倍，质量略有下降)
scheduler = DiffusionScheduler(num_steps=50)
```

**Action Chunking:**
- 一次性预测 16 步动作
- 实际执行前 8 步 (action_horizon)
- 减少扩散采样频率

### 6.3 评估指标

| 指标 | 说明 |
|------|------|
| **Success Rate** | 任务完成成功率 (%) |
| **Avg Steps** | 完成任务所需平均步数 |
| **Val Loss** | 验证集噪声预测 MSE |

---

## 7. 配置参数

### 7.1 环境配置

```python
env_name: "TwoArmLift"          # 任务名称
robots: ["Panda", "Panda"]      # 双臂
horizon: 500                    # 最大步数
control_freq: 20                # 控制频率
```

### 7.2 模型配置

```python
# 纯状态
action_dim: 14
state_dim: 50
hidden_dims: [256, 512, 1024]

# 视觉+状态
camera_names: ["frontview", "agentview"]
image_shape: (3, 216, 288)
vision_embed_dim: 256
```

### 7.3 训练配置

```python
# 优化器
lr: 1e-4
weight_decay: 1e-6
batch_size: 128

# 调度
num_epochs: 2000
lr_warmup_steps: 500

# EMA
use_ema: True
ema_warmup_power: 0.8
max_decay: 0.9999

# 扩散
num_diffusion_steps: 100
beta_schedule: "linear"
```

---

## 8. 故障排查

### 8.1 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| **段错误 (Segmentation Fault)** | DataLoader 多进程与 MuJoCo 冲突 | `num_workers=0` |
| **显存不足** | Batch size 太大 | 减小 batch_size 或 image size |
| **损失不下降** | 学习率太大/数据未归一化 | 检查归一化，降低 lr |
| **推理失败** | 观测维度不匹配 | 检查训练/推理观测 key 一致 |

### 8.2 性能优化

```bash
# 快速训练 (实验)
python train_with_vision.py \
    --batch_size 64 \
    --num_epochs 500 \
    --eval_every 200

# 高质量训练 (部署)
python train_with_vision.py \
    --batch_size 128 \
    --num_epochs 2000 \
    --use_ema_warmup \
    --use_lr_warmup \
    --eval_every 100
```

### 8.3 Debug 模式

```python
# 小数据快速验证
python train_with_vision.py \
    --num_epochs 2 \
    --batch_size 4
```

---

## 附录: 引用

- **Diffusion Policy**: [Chi et al., ICLR 2023](https://diffusion-policy.cs.columbia.edu/)
- **RoboSuite**: [Zhu et al., ICRA 2020](https://robosuite.ai/)
- **EMA Warmup**: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)

---

*文档版本: v1.0*  
*最后更新: 2025-04-10*
