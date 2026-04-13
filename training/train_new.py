"""
Diffusion Policy Training Script - 新版本
包含 DDPM 调度器、EMA、完整的训练循环
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import json
from typing import Dict
from collections import deque

import robosuite as suite

from dataset import RobosuiteDiffusionDataset, collate_fn
from network import ConditionalUNet1D
from config import get_default_config


class DDPMScheduler:
    """
    DDPM 噪声调度器
    预计算加噪和去噪所需的参数
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cuda",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        
        # 生成 beta 调度
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_train_timesteps)
            self.betas = (beta_end - beta_start) * (1 - torch.cos(steps / num_train_timesteps * np.pi)) / 2 + beta_start
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 预计算 alpha 相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 加噪公式中的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 移动到指定设备
        for key in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                    'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, key, getattr(self, key).to(device))
    
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """前向加噪: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise


class EMAModel:
    """指数滑动平均模型 - 用于稳定推理"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow_params = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        # 保存原始参数，用于恢复
        self.original_params = {}
    
    def step(self, model: nn.Module):
        """更新 EMA 参数: shadow = decay * shadow + (1 - decay) * param"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data
    
    def state_dict(self) -> Dict:
        return self.shadow_params
    
    def load_state_dict(self, state_dict: Dict):
        self.shadow_params = state_dict
    
    def apply_shadow(self, model: nn.Module):
        """将 EMA 参数应用到模型，同时保存原始参数"""
        self.original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.original_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
    
    def restore_original(self, model: nn.Module):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original_params:
                param.data.copy_(self.original_params[name])
        self.original_params = {}


class SmartLRScheduler:
    """
    智能学习率调度器
    结合 Warmup + CosineAnnealing + ReduceLROnPlateau
    """
    
    def __init__(
        self,
        optimizer,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        plateau_patience: int = 30,
        plateau_factor: float = 0.5,
    ):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.current_plateau_factor = 1.0
        
        self.total_steps = num_epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        
    def step(self, step: int, current_loss: float = None):
        """更新学习率"""
        self.current_epoch = step // self.steps_per_epoch
        
        # 1. Warmup阶段
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
            lr = self.base_lr * warmup_factor
        else:
            # 2. Cosine退火
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # 3. Plateau衰减
        if current_loss is not None and step % self.steps_per_epoch == 0:
            if current_loss >= self.best_loss * 0.999:
                self.plateau_counter += 1
                if self.plateau_counter >= self.plateau_patience:
                    self.current_plateau_factor *= self.plateau_factor
                    self.plateau_counter = 0
                    print(f"\n🔄 Reducing LR on plateau: {self.get_lr():.2e} -> {lr * self.current_plateau_factor:.2e}")
            else:
                self.plateau_counter = max(0, self.plateau_counter - 1)
                self.best_loss = min(self.best_loss, current_loss)
        
        lr = max(lr * self.current_plateau_factor, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """保存状态，用于断点恢复"""
        return {
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'plateau_counter': self.plateau_counter,
            'current_plateau_factor': self.current_plateau_factor,
        }
    
    def load_state_dict(self, state_dict):
        """加载状态，恢复训练"""
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.plateau_counter = state_dict.get('plateau_counter', 0)
        self.current_plateau_factor = state_dict.get('current_plateau_factor', 1.0)


def create_eval_env():
    """创建评估环境"""
    env = suite.make(
        env_name="TwoArmLift",
        robots=["Panda", "Panda"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
    )
    return env


class DDPMSampler:
    """DDPM 采样器 - 用于推理时的去噪"""
    
    def __init__(self, num_steps: int = 100, device: str = "cuda"):
        self.num_steps = num_steps
        self.device = device
        
        # 预计算参数 (linear schedule)
        betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def sample(self, model: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        """
        从噪声中采样动作
        
        Args:
            model: U-Net 模型
            obs: (1, obs_horizon, state_dim) 观测条件
        Returns:
            actions: (1, pred_horizon, action_dim) 生成的动作
        """
        # 初始化随机噪声
        noisy_actions = torch.randn(1, 16, 14, device=self.device)
        
        # 逆向扩散
        for t in reversed(range(self.num_steps)):
            timestep = torch.tensor([t], device=self.device)
            
            with torch.no_grad():
                # 预测噪声
                noise_pred = model(noisy_actions, timestep, obs)
                
                # 计算当前步的 alpha (转为 tensor)
                alpha = torch.tensor(1.0 - (1e-4 + (0.02 - 1e-4) * t / self.num_steps), device=self.device)
                alpha_prod = self.alphas_cumprod[t]
                alpha_prod_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
                
                # 预测 x_0 (原始动作)
                pred_original = (noisy_actions - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / self.sqrt_alphas_cumprod[t]
                
                # 计算均值
                coef1 = torch.sqrt(alpha_prod_prev) * (1 - alpha) / (1 - alpha_prod)
                coef2 = torch.sqrt(alpha) * (1 - alpha_prod_prev) / (1 - alpha_prod)
                mean = coef1 * pred_original + coef2 * noisy_actions
                
                if t > 0:
                    # 添加噪声
                    variance = (1 - alpha_prod_prev) / (1 - alpha_prod) * (1 - alpha)
                    noise = torch.randn_like(noisy_actions)
                    noisy_actions = mean + torch.sqrt(variance) * noise
                else:
                    noisy_actions = mean
        
        return noisy_actions


def evaluate_model(
    model: nn.Module,
    dataset: RobosuiteDiffusionDataset,
    num_episodes: int = 10,
    num_diffusion_steps: int = 100,
    device: str = "cuda",
    obs_horizon: int = 2,
) -> float:
    """
    在真实环境中评估模型成功率
    """
    model.eval()
    env = create_eval_env()
    sampler = DDPMSampler(num_diffusion_steps, device)
    
    # 获取归一化参数
    stats = dataset.stats
    obs_mean = torch.from_numpy(stats['state_mean']).float().to(device)
    obs_std = torch.from_numpy(stats['state_std']).float().to(device)
    action_scale = torch.from_numpy(stats['action_scale']).float().to(device)
    action_offset = torch.from_numpy(stats['action_offset']).float().to(device)
    
    success_count = 0
    
    for ep in range(num_episodes):
        obs_dict = env.reset()
        action_queue = deque()
        obs_history = deque(maxlen=2)  # 🌟 维护观测历史
        done = False
        step = 0
        
        while not done and step < env.horizon:
            # 🌟 更新观测历史
            current_obs = np.array(obs_dict['robot0_proprio-state'])
            obs_history.append(current_obs)
            
            # 如果历史不足，用当前观测填充
            while len(obs_history) < 2:
                obs_history.append(current_obs)
            
            # 如果动作队列空了，重新预测
            if len(action_queue) == 0:
                # 🌟 使用观测历史 (与训练时一致)
                obs = np.stack(list(obs_history), axis=0)  # (2, 50)
                
                # 转换为 Tensor
                obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)  # (1, 2, 50)
                
                # 归一化
                obs = (obs - obs_mean) / (obs_std + 1e-8)
                
                # 扩散采样
                with torch.no_grad():
                    action_seq = sampler.sample(model, obs)[0]  # (16, 14)
                
                # 反归一化
                action_seq = action_seq * action_scale + action_offset
                action_seq = torch.clamp(action_seq, -1.0, 1.0)
                action_seq = action_seq.cpu().numpy()
                
                # 加入动作队列
                execution_steps = 8
                for i in range(min(execution_steps, len(action_seq))):
                    action_queue.append(action_seq[i])
            
            # 执行动作
            action = action_queue.popleft()
            low, high = env.action_spec
            action = np.clip(action, low, high)
            
            obs_dict, reward, done, info = env.step(action)
            step += 1
            
            if info.get('success', False):
                success_count += 1
                break
    
    env.close()
    return success_count / num_episodes


def train():
    """主训练函数"""
    
    # ==================== 配置 ====================
    config = {
        "data_dir": "./expert_data",
        "checkpoint_dir": "./checkpoints",
        "obs_horizon": 2,
        "pred_horizon": 16,
        "observation_keys": ["states"],  # 使用完整的 50 维 states，包含所有信息
        "action_dim": 14,
        "state_dim": 50,  # 使用完整的 50 维状态 (robot0_proprio-state)
        "hidden_dim": 128,
        "down_dims": [128, 256, 512],
        "batch_size": 256,
        "num_epochs": 2000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "ema_decay": 0.9999,
        "warmup_epochs": 5,           # Warmup轮数
        "min_lr": 1e-6,               # 最小学习率
        "plateau_patience": 50,       # Loss停滞多少轮后降学习率
        "plateau_factor": 0.5,        # 学习率衰减因子
        "num_train_timesteps": 100,
        "beta_schedule": "linear",
        "save_every": 50,
        "eval_every": 100,        # 每20个epoch评估一次
        "eval_episodes": 10,     # 每次评估10个回合
    }
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    with open(os.path.join(config["checkpoint_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==================== 数据 ====================
    print("\nLoading Dataset...")
    dataset = RobosuiteDiffusionDataset(
        data_dir=config["data_dir"],
        observation_keys=config["observation_keys"],
        obs_horizon=config["obs_horizon"],
        pred_horizon=config["pred_horizon"],
    )
    
    # 验证状态维度
    sample = dataset[0]
    actual_state_dim = sample['obs'].shape[-1]
    print(f"Actual state dimension: {actual_state_dim}")
    if actual_state_dim != config["state_dim"]:
        print(f"⚠️  Warning: config state_dim ({config['state_dim']}) != actual ({actual_state_dim})")
        print(f"Updating state_dim to {actual_state_dim}")
        config["state_dim"] = actual_state_dim
    
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")
    
    # ==================== 模型 ====================
    print("\nInitializing Model...")
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        pred_horizon=config["pred_horizon"],
        obs_horizon=config["obs_horizon"],
        state_dim=config["state_dim"],
        hidden_dim=config["hidden_dim"],
        down_dims=config["down_dims"],
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    ema = EMAModel(model, decay=config["ema_decay"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # 使用智能学习率调度器
    scheduler = SmartLRScheduler(
        optimizer,
        num_epochs=config["num_epochs"],
        steps_per_epoch=len(dataloader),
        warmup_epochs=config["warmup_epochs"],
        min_lr=config["min_lr"],
        plateau_patience=config["plateau_patience"],
        plateau_factor=config["plateau_factor"],
    )
    print(f"📚 Smart LR Scheduler: warmup={config['warmup_epochs']}epochs, "
          f"min_lr={config['min_lr']}, plateau_patience={config['plateau_patience']}")
    
    noise_scheduler = DDPMScheduler(config["num_train_timesteps"], device=device)
    
    # ==================== 训练 ====================
    print("\nStarting Training...")
    
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in pbar:
            obs = batch["obs"].to(device)
            action_target = batch["action"].to(device)
            
            # 随机采样时间步和噪声
            timesteps = torch.randint(0, config["num_train_timesteps"], (obs.shape[0],), device=device)
            noise = torch.randn_like(action_target)
            
            # 加噪 -> 预测 -> 损失
            noisy_actions = noise_scheduler.add_noise(action_target, noise, timesteps)
            noise_pred = model(noisy_actions, timesteps, obs)
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step = epoch * len(dataloader) + pbar.n
            current_lr = scheduler.step(global_step, avg_loss if pbar.n == len(dataloader) - 1 else None)
            ema.step(model)
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_lr()
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}, lr = {current_lr:.2e}")
        
        # ==================== 定期评估 ====================
        if (epoch + 1) % config["eval_every"] == 0:
            print(f"\n[Eval] Running evaluation at epoch {epoch+1}...")
            
            # 将 EMA 参数应用到模型进行评估
            ema.apply_shadow(model)
            
            success_rate = evaluate_model(
                model=model,
                dataset=dataset,
                num_episodes=config["eval_episodes"],
                num_diffusion_steps=config["num_train_timesteps"],
                device=device,
                obs_horizon=config["obs_horizon"],
            )
            
            print(f"[Eval] Success Rate: {success_rate*100:.1f}% ({int(success_rate*config['eval_episodes'])}/{config['eval_episodes']})")
            
            # 保存基于成功率的最佳模型（保存EMA参数）
            best_success_rate = getattr(train, 'best_success_rate', 0.0)
            if success_rate > best_success_rate:
                train.best_success_rate = success_rate
                best_path = os.path.join(config["checkpoint_dir"], "best_success_model.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),  # 此时已经是EMA参数
                    "ema_state_dict": ema.state_dict(),
                    "config": config,
                    "dataset_stats": dataset.get_stats(),
                    "success_rate": success_rate,
                }, best_path)
                print(f"🌟 New best success rate! Model saved: {best_path}")
            
            # 恢复原始参数，继续训练
            ema.restore_original(model)
            model.train()
            print()
        
        # 🌟 每轮都保存最新模型（覆盖式，用于断点恢复）
        latest_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "dataset_stats": dataset.get_stats(),
        }
        torch.save(latest_checkpoint, os.path.join(config["checkpoint_dir"], "latest_model.pt"))
        
        # 定期保存历史检查点（用于回溯）
        if (epoch + 1) % config["save_every"] == 0:
            torch.save(latest_checkpoint, os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pt"))
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
    
    # 保存最终模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "config": config,
        "dataset_stats": dataset.get_stats(),
    }, os.path.join(config["checkpoint_dir"], "final_model.pt"))
    
    print("\nTraining Complete!")


def inference(checkpoint_path: str, num_episodes: int = 10, render: bool = True):
    """
    加载训练好的模型进行推理
    
    Args:
        checkpoint_path: 检查点文件路径
        num_episodes: 推理回合数
        render: 是否渲染
    """
    print("=" * 70)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    stats = checkpoint["dataset_stats"]
    
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"State dim: {stats['state_mean'].shape[0]}")
    print(f"Action dim: {stats['action_scale'].shape[0]}")
    
    # 创建模型
    model = ConditionalUNet1D(
        action_dim=config["action_dim"],
        pred_horizon=config["pred_horizon"],
        obs_horizon=config["obs_horizon"],
        state_dim=stats['state_mean'].shape[0],  # 使用实际维度
        hidden_dim=config["hidden_dim"],
        down_dims=config["down_dims"],
    ).to(device)
    
    # 加载EMA模型参数（如果存在）
    if "ema_state_dict" in checkpoint:
        print("Loading EMA model weights...")
        # 创建一个临时EMA对象来加载参数
        ema_temp = EMAModel(model)
        ema_temp.load_state_dict(checkpoint["ema_state_dict"])
        ema_temp.apply_shadow(model)
    else:
        print("Loading standard model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    # 创建数据集对象（仅用于获取归一化参数）
    class DummyDataset:
        def __init__(self, stats):
            self.stats = stats
    
    dummy_dataset = DummyDataset(stats)
    
    # 评估
    print(f"\nRunning inference for {num_episodes} episodes...")
    success_rate = evaluate_model(
        model=model,
        dataset=dummy_dataset,
        num_episodes=num_episodes,
        num_diffusion_steps=config["num_train_timesteps"],
        device=device,
        obs_horizon=config["obs_horizon"],
    )
    
    print(f"\nInference Complete!")
    print(f"Success Rate: {success_rate * 100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diffusion Policy Training and Inference")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for eval mode")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        if args.checkpoint is None:
            # 默认使用最新的检查点
            checkpoint_dir = "./checkpoints"
            args.checkpoint = os.path.join(checkpoint_dir, "best_success_model.pt")
            if not os.path.exists(args.checkpoint):
                args.checkpoint = os.path.join(checkpoint_dir, "final_model.pt")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
        else:
            inference(args.checkpoint, num_episodes=args.episodes)
