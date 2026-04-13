"""
训练支持图像和状态输入的 Diffusion Policy
- 添加状态输入（50维）
- 动作使用 min-max 归一化到 [-1, 1]
- 每100 epoch 环境评估
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, Subset
from typing import Optional, Dict
from tqdm import tqdm
import argparse
import robosuite as suite
from datetime import datetime

# 导入自定义模块
from network_with_vision import ConditionalUNet1DWithVision
from dataset_with_vision import VisionDiffusionDataset, collate_fn
from config import get_default_config


class ImageTransforms:
    """图像变换类，支持训练和评估模式
    
    使用 216x288 裁剪（240x320 的 90%），保留更多视野信息
    """
    
    def __init__(self, crop_height: int = 216, crop_width: int = 288):
        self.crop_height = crop_height
        self.crop_width = crop_width
        
    def get_train_transform(self):
        """训练时：随机裁剪
        
        输入: float32 numpy array [0, 1], shape (H, W, 3)
              (HDF5 图像是 uint8 [0,255]，需要在传入前做 /255)
        输出: float32 tensor, shape (3, crop_height, crop_width), ImageNet normalized
        """
        def transform(img: np.ndarray) -> torch.Tensor:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # 已经是 [0,1]，不需要 /255
            else:
                img_tensor = img.permute(2, 0, 1).float()
            
            _, H, W = img_tensor.shape
            
            # 随机裁剪到目标尺寸
            if H > self.crop_height and W > self.crop_width:
                top = np.random.randint(0, H - self.crop_height + 1)
                left = np.random.randint(0, W - self.crop_width + 1)
                img_tensor = img_tensor[:, top:top+self.crop_height, left:left+self.crop_width]
            else:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(self.crop_height, self.crop_width),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # ImageNet归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (img_tensor - mean) / std
        return transform
    
    def get_eval_transform(self):
        """评估时：中心裁剪
        
        输入: float32 numpy array [0, 1], shape (H, W, 3)
              (ObsBuffer 在 get_stacked_obs 中做了 /255)
        输出: float32 tensor, shape (3, crop_height, crop_width), ImageNet normalized
        """
        def transform(img: np.ndarray) -> torch.Tensor:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # 已经是 [0,1]，不需要 /255
            else:
                img_tensor = img.permute(2, 0, 1).float()
            
            _, H, W = img_tensor.shape
            
            if H > self.crop_height and W > self.crop_width:
                top = (H - self.crop_height) // 2
                left = (W - self.crop_width) // 2
                img_tensor = img_tensor[:, top:top+self.crop_height, left:left+self.crop_width]
            else:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.crop_height, self.crop_width),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (img_tensor - mean) / std
        return transform


class VisionDatasetWithAug(VisionDiffusionDataset):
    """支持数据增强的数据集"""
    
    def __init__(self, *args, crop_height=216, crop_width=288, train_mode=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = ImageTransforms(crop_height=crop_height, crop_width=crop_width)
        self.train_mode = train_mode
        self.image_transform = self.transforms.get_train_transform() if train_mode else self.transforms.get_eval_transform()
    
    def set_train_mode(self, train_mode: bool):
        self.train_mode = train_mode
        self.image_transform = self.transforms.get_train_transform() if train_mode else self.transforms.get_eval_transform()
    
    def _process_images(self, images_dict, start, end):
        result = {}
        for cam_name, img_seq in images_dict.items():
            imgs = img_seq[start:end]
            # HDF5 图像是 uint8 [0,255]，需要转换为 float32 [0,1]
            # 然后传给 image_transform 做裁剪和 ImageNet 归一化
            processed = [self.image_transform(img.astype(np.float32) / 255.0) for img in imgs]
            result[cam_name] = torch.stack(processed, dim=0)
        return result


class DiffusionScheduler:
    """扩散调度器"""
    
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x, noise, t):
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
    
    @torch.no_grad()
    def sample(self, model, obs_images, obs_states, pred_horizon, action_dim):
        """采样动作序列"""
        shape = (1, pred_horizon, action_dim)
        actions = torch.randn(shape, device=self.device)
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
            
            # 根据是否有状态输入调用模型
            if obs_states is not None:
                noise_pred = model(actions, t_batch, obs_images, obs_states)
            else:
                noise_pred = model(actions, t_batch, obs_images)
            
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            actions = (actions - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(actions)
                sigma_t = torch.sqrt(self.betas[t] * (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t]))
                actions = actions + sigma_t * noise
        
        return actions.squeeze(0)


class ObsBuffer:
    """观测缓冲区，用于维护最近的观测历史"""
    
    def __init__(self, obs_horizon, camera_names, state_dim=None):
        self.obs_horizon = obs_horizon
        self.camera_names = camera_names
        self.state_dim = state_dim
        
        self.image_buffers = {cam: [] for cam in camera_names}
        self.state_buffer = [] if state_dim is not None else None
    
    def add_obs(self, obs_dict):
        """添加观测"""
        for cam in self.camera_names:
            if cam in obs_dict:
                self.image_buffers[cam].append(obs_dict[cam].copy())
                if len(self.image_buffers[cam]) > self.obs_horizon:
                    self.image_buffers[cam].pop(0)
        
        if self.state_buffer is not None and 'state' in obs_dict:
            self.state_buffer.append(obs_dict['state'].copy())
            if len(self.state_buffer) > self.obs_horizon:
                self.state_buffer.pop(0)
    
    def get_stacked_obs(self, transform=None, crop_height=216, crop_width=288):
        """获取堆叠的观测"""
        result = {}
        
        # 处理图像
        for cam in self.camera_names:
            while len(self.image_buffers[cam]) < self.obs_horizon:
                if len(self.image_buffers[cam]) == 0:
                    empty = np.zeros((crop_height, crop_width, 3), dtype=np.float32)
                    self.image_buffers[cam].append(empty)
                else:
                    self.image_buffers[cam].insert(0, self.image_buffers[cam][0])
            
            frames = self.image_buffers[cam][-self.obs_horizon:]
            processed = []
            for img in frames:
                if transform:
                    img_float = img.astype(np.float32) / 255.0
                    tensor = transform(img_float)
                else:
                    img_float = img.astype(np.float32) / 255.0
                    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
                processed.append(tensor)
            
            result[cam] = torch.stack(processed, dim=0).unsqueeze(0)
        
        # 处理状态
        if self.state_buffer is not None:
            while len(self.state_buffer) < self.obs_horizon:
                if len(self.state_buffer) == 0:
                    empty = np.zeros(self.state_dim, dtype=np.float32)
                    self.state_buffer.append(empty)
                else:
                    self.state_buffer.insert(0, self.state_buffer[0])
            
            states = self.state_buffer[-self.obs_horizon:]
            result['state'] = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(0)
        
        return result


class VisionDiffusionPolicyTrainer:
    """V2训练器，支持状态输入和定期环境评估
    
    新增特性（可选）：
    - 动态EMA warmup (power=0.8): use_ema_warmup=True
    - 学习率warmup (100 epochs): use_lr_warmup=True
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=2000,
        lr=1e-4,
        weight_decay=1e-6,
        device='cuda',
        checkpoint_dir='checkpoints',
        use_ema=True,
        ema_decay=0.999,
        eval_every=100,
        num_eval_episodes=5,
        camera_names=['frontview', 'agentview'],
        crop_height=216,
        crop_width=288,
        # 新增开关参数
        use_ema_warmup=False,      # 是否使用EMA动态warmup
        ema_warmup_power=0.8,      # EMA warmup曲线power值
        use_lr_warmup=False,       # 是否使用学习率warmup
        lr_warmup_steps=500,       # 学习率warmup的steps数（与原文一致，默认500）
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset if val_loader is not None else None
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_ema = use_ema
        self.eval_every = eval_every
        self.num_eval_episodes = num_eval_episodes
        self.camera_names = camera_names
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.base_lr = lr
        
        # 获取归一化统计信息（从数据集）
        base_dataset = self.train_dataset.dataset if hasattr(self.train_dataset, 'dataset') else self.train_dataset
        self.normalization_stats = base_dataset.get_normalization_stats()
        self.use_states = 'state_min' in self.normalization_stats
        self.state_dim = base_dataset.state_dim if self.use_states else None
        
        self.scheduler = DiffusionScheduler(num_steps=100, device=device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # ===== 学习率调度配置 =====
        self.use_lr_warmup = use_lr_warmup
        self.steps_per_epoch = len(train_loader)
        self.total_steps = num_epochs * self.steps_per_epoch
        
        if use_lr_warmup:
            # 使用 warmup + cosine（按 steps，与原文一致）
            self.warmup_steps = lr_warmup_steps
            self.lr_scheduler = None
            print(f"[LR Warmup] warmup={lr_warmup_steps} steps, "
                  f"total={self.total_steps} steps ({num_epochs} epochs)")
        else:
            # 使用纯 cosine annealing
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=lr * 0.01
            )
        
        # ===== EMA 配置 =====
        self.use_ema_warmup = use_ema_warmup
        self.ema_optimization_step = 0
        
        # 根据warmup配置调整保存路径
        if use_ema_warmup or use_lr_warmup:
            suffix_parts = []
            if use_ema_warmup:
                suffix_parts.append(f"ema_warmup_{ema_warmup_power}")
            if use_lr_warmup:
                suffix_parts.append(f"lr_warmup_{lr_warmup_steps}steps")
            suffix = "_".join(suffix_parts)
            self.checkpoint_dir = os.path.join(checkpoint_dir, suffix)
            print(f"[Checkpoint] Using warmup-specific directory: {self.checkpoint_dir}")
        else:
            self.checkpoint_dir = checkpoint_dir
        
        if use_ema:
            from copy import deepcopy
            self.ema_model = deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            
            if use_ema_warmup:
                # 动态EMA配置 (power=0.8)
                self.ema_config = {
                    'update_after_step': 0,
                    'inv_gamma': 1.0,
                    'power': ema_warmup_power,
                    'min_value': 0.0,
                    'max_value': ema_decay,
                }
                print(f"[EMA Warmup] enabled: power={ema_warmup_power}, max_decay={ema_decay}")
            else:
                # 固定EMA衰减率
                self.ema_decay = ema_decay
                print(f"[EMA] fixed decay: {ema_decay}")
        else:
            self.ema_model = None
        
        self.epoch = 0
        self.global_step = 0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 创建评估环境
        print("Creating eval environment...")
        config = get_default_config()
        self.eval_env = suite.make(
            env_name=config.env.env_name,
            robots=config.env.robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=240,
            camera_widths=320,
            camera_depths=False,
            horizon=config.env.horizon,
            control_freq=config.env.control_freq,
        )
        self.eval_transform = ImageTransforms(crop_height=crop_height, crop_width=crop_width).get_eval_transform()
    
    def get_ema_decay_dynamic(self, optimization_step):
        """计算动态EMA衰减率 (power曲线)"""
        step = max(0, optimization_step - self.ema_config['update_after_step'] - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.ema_config['inv_gamma']) ** -self.ema_config['power']
        return max(self.ema_config['min_value'], 
                   min(value, self.ema_config['max_value']))
    
    def get_lr_with_warmup(self, step):
        """计算带warmup的学习率"""
        if step < self.warmup_steps:
            # 线性warmup
            return self.base_lr * (step / self.warmup_steps)
        else:
            # cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _update_ema(self):
        """更新EMA模型"""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            if self.use_ema_warmup:
                # 动态decay (power warmup)
                decay = self.get_ema_decay_dynamic(self.ema_optimization_step)
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.mul_(decay)
                    ema_param.add_(param.data, alpha=1 - decay)
                self.ema_optimization_step += 1
            else:
                # 固定decay
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def unnormalize_action_for_eval(self, action):
        """在评估时对动作进行反归一化"""
        action_min = self.normalization_stats['action_min']
        action_max = self.normalization_stats['action_max']
        scale = 2.0 / (action_max - action_min)
        offset = -1.0 - scale * action_min
        return (action - offset) / scale
    
    def train_epoch(self):
        self.model.train()
        if hasattr(self.train_dataset, 'set_train_mode'):
            self.train_dataset.set_train_mode(True)
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch [{self.epoch+1}/{self.num_epochs}]")
        
        for batch in pbar:
            actions = batch['actions'].to(self.device)
            obs_images = {k: v.to(self.device) for k, v in batch['obs_images'].items()}
            
            # 如果有状态输入
            obs_states = batch.get('obs_states', None)
            if obs_states is not None:
                obs_states = obs_states.to(self.device)
            
            B = actions.shape[0]
            t = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)
            noise = torch.randn_like(actions)
            noisy_actions = self.scheduler.add_noise(actions, noise, t)
            
            # 根据是否有状态输入调用模型
            if obs_states is not None:
                noise_pred = self.model(noisy_actions, t, obs_images, obs_states)
            else:
                noise_pred = self.model(noisy_actions, t, obs_images)
            
            loss = F.mse_loss(noise_pred, noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self._update_ema()
            
            # 学习率更新
            if self.use_lr_warmup:
                current_lr = self.get_lr_with_warmup(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 准备进度条显示信息
            postfix_dict = {
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            }
            
            # 显示EMA decay（如果使用动态warmup）
            if self.use_ema and self.use_ema_warmup:
                ema_decay_display = self.get_ema_decay_dynamic(self.ema_optimization_step - 1)
                postfix_dict['ema'] = f"{ema_decay_display:.4f}"
            
            pbar.set_postfix(postfix_dict)
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        if hasattr(self.val_dataset, 'set_train_mode'):
            self.val_dataset.set_train_mode(False)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            actions = batch['actions'].to(self.device)
            obs_images = {k: v.to(self.device) for k, v in batch['obs_images'].items()}
            obs_states = batch.get('obs_states', None)
            if obs_states is not None:
                obs_states = obs_states.to(self.device)
            
            B = actions.shape[0]
            t = torch.randint(0, self.scheduler.num_steps, (B,), device=self.device)
            noise = torch.randn_like(actions)
            noisy_actions = self.scheduler.add_noise(actions, noise, t)
            
            if obs_states is not None:
                noise_pred = self.model(noisy_actions, t, obs_images, obs_states)
            else:
                noise_pred = self.model(noisy_actions, t, obs_images)
            
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate_in_environment(self):
        """在真实环境中评估"""
        model_to_eval = self.ema_model if self.ema_model is not None else self.model
        model_to_eval.eval()
        
        print(f"\n{'='*60}")
        print(f"Environment Evaluation at Epoch {self.epoch+1}/{self.num_epochs}")
        print(f"{'='*60}")
        
        successes = []
        steps_list = []
        
        for ep in range(self.num_eval_episodes):
            obs = self.eval_env.reset()
            
            # 初始化观测缓冲区
            obs_buffer = ObsBuffer(
                obs_horizon=2, 
                camera_names=self.camera_names,
                state_dim=self.state_dim
            )
            
            # 获取初始观测
            obs_images = {}
            for cam in self.camera_names:
                img_key = f"{cam}_image"
                if img_key in obs:
                    obs_images[cam] = obs[img_key]
            
            # 获取初始状态并归一化
            if self.use_states:
                if 'robot0_proprio-state' in obs:
                    obs_state = obs['robot0_proprio-state']
                elif 'states' in obs:
                    obs_state = obs['states']
                else:
                    # 从其他观测构造状态
                    obs_state = np.concatenate([
                        obs.get('robot0_eef_pos', np.zeros(3)),
                        obs.get('robot0_eef_quat', np.zeros(4)),
                        obs.get('robot0_gripper_qpos', np.zeros(1)),
                        obs.get('robot1_eef_pos', np.zeros(3)),
                        obs.get('robot1_eef_quat', np.zeros(4)),
                        obs.get('robot1_gripper_qpos', np.zeros(1)),
                    ])
                
                # 归一化状态到 [-1, 1]
                if self.normalization_stats is not None:
                    state_min = self.normalization_stats['state_min']
                    state_max = self.normalization_stats['state_max']
                    scale = 2.0 / (state_max - state_min)
                    offset = -1.0 - scale * state_min
                    obs_state = obs_state * scale + offset
                
                obs_images['state'] = obs_state
            
            obs_buffer.add_obs(obs_images)
            
            action_queue = []
            step_idx = 0
            max_steps = 500
            done = False
            episode_success = False
            
            while step_idx < max_steps and not done:
                if len(action_queue) == 0:
                    # 获取堆叠观测
                    obs_tensors = obs_buffer.get_stacked_obs(
                        transform=self.eval_transform,
                        crop_height=self.crop_height,
                        crop_width=self.crop_width
                    )
                    
                    # 提取图像和状态
                    obs_imgs = {k: v.to(self.device) for k, v in obs_tensors.items() if k != 'state'}
                    obs_state_tensor = obs_tensors.get('state', None)
                    if obs_state_tensor is not None:
                        obs_state_tensor = obs_state_tensor.to(self.device)
                    
                    # 采样动作序列
                    action_seq = self.scheduler.sample(
                        model_to_eval, obs_imgs, obs_state_tensor, 
                        pred_horizon=16, action_dim=14
                    )
                    
                    # 反归一化动作
                    action_seq_np = action_seq.cpu().numpy()
                    action_seq_unnorm = self.unnormalize_action_for_eval(action_seq_np)
                    
                    action_queue = list(action_seq_unnorm[:8])  # action_horizon=8
                
                # 执行动作
                action = action_queue.pop(0)
                obs, reward, done, info = self.eval_env.step(action)
                
                # 检查成功
                if info.get('success', False):
                    episode_success = True
                
                # 更新观测缓冲区
                obs_images = {}
                for cam in self.camera_names:
                    img_key = f"{cam}_image"
                    if img_key in obs:
                        obs_images[cam] = obs[img_key]
                
                if self.use_states:
                    if 'robot0_proprio-state' in obs:
                        obs_state = obs['robot0_proprio-state']
                    elif 'states' in obs:
                        obs_state = obs['states']
                    else:
                        obs_state = np.concatenate([
                            obs.get('robot0_eef_pos', np.zeros(3)),
                            obs.get('robot0_eef_quat', np.zeros(4)),
                            obs.get('robot0_gripper_qpos', np.zeros(1)),
                            obs.get('robot1_eef_pos', np.zeros(3)),
                            obs.get('robot1_eef_quat', np.zeros(4)),
                            obs.get('robot1_gripper_qpos', np.zeros(1)),
                        ])
                    
                    # 归一化状态到 [-1, 1]
                    if self.normalization_stats is not None:
                        state_min = self.normalization_stats['state_min']
                        state_max = self.normalization_stats['state_max']
                        scale = 2.0 / (state_max - state_min)
                        offset = -1.0 - scale * state_min
                        obs_state = obs_state * scale + offset
                    
                    obs_images['state'] = obs_state
                
                obs_buffer.add_obs(obs_images)
                step_idx += 1
            
            successes.append(episode_success)
            steps_list.append(step_idx)
            print(f"  Episode {ep+1}: Success={episode_success}, Steps={step_idx}")
        
        success_rate = np.mean(successes)
        avg_steps = np.mean(steps_list)
        
        print(f"\n  Summary: Success Rate = {success_rate*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"          Avg Steps = {avg_steps:.1f}")
        print(f"{'='*60}\n")
        
        return success_rate
    
    def train(self):
        """训练循环"""
        best_val_loss = float('inf')
        
        print(f"\n{'='*60}")
        print(f"Training started!")
        print(f"  Total epochs: {self.num_epochs}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Save frequency: every 50 epochs + best model")
        print(f"  Eval frequency: every {self.eval_every} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率（仅在非warmup模式下调用scheduler）
            if not self.use_lr_warmup and self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # 打印日志
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 更新最佳验证损失
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # 每轮都保存最新模型（覆盖 latest_model.pt）
            self.save_checkpoint(epoch, val_loss, is_best=False, save_latest=True)
            
            # 每50轮额外保存历史检查点 + 最佳模型
            if (epoch + 1) % 50 == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch, val_loss, is_best=is_best, save_latest=False)
            
            # 定期环境评估
            if (epoch + 1) % self.eval_every == 0:
                success_rate = self.evaluate_in_environment()
                self.save_checkpoint(epoch, val_loss, success_rate=success_rate, save_latest=True)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点进行断点续训"""
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        # 使用 weights_only=False 加载（PyTorch 2.6+ 兼容性）
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model weights")
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded optimizer state")
        
        # 加载学习率调度器状态
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Loaded scheduler state")
        
        # 加载EMA模型
        if 'ema_model_state_dict' in checkpoint and self.ema_model is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            print(f"✅ Loaded EMA model")
        
        # 恢复训练进度
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch'] + 1  # 从下一epoch开始
            print(f"✅ Resuming from epoch {self.epoch}")
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            print(f"✅ Global step: {self.global_step}")
        
        if 'ema_optimization_step' in checkpoint and self.use_ema_warmup:
            self.ema_optimization_step = checkpoint['ema_optimization_step']
            print(f"✅ EMA step: {self.ema_optimization_step}")
        
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False, success_rate=None, save_latest=True):
        """保存检查点
        
        Args:
            save_latest: 是否保存/覆盖 latest_model.pt（每轮都保存最新）
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'normalization_stats': self.normalization_stats,
        }
        
        # 处理 lr_scheduler 可能为 None 的情况
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        else:
            checkpoint['scheduler_state_dict'] = None
        
        # 保存 warmup 配置（用于恢复训练）
        checkpoint['use_lr_warmup'] = self.use_lr_warmup
        checkpoint['use_ema_warmup'] = self.use_ema_warmup
        checkpoint['global_step'] = self.global_step
        checkpoint['ema_optimization_step'] = self.ema_optimization_step if self.use_ema_warmup else 0
        
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        if success_rate is not None:
            checkpoint['success_rate'] = success_rate
        
        # 保存/覆盖最新检查点（每轮都执行）
        if save_latest:
            latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
            torch.save(checkpoint, latest_path)
            print(f"  [Checkpoint] Saved latest model (epoch {epoch+1}) to {latest_path}")
        
        # 保存最佳检查点（仅当创记录时）
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  [Checkpoint] Saved BEST model (val_loss={val_loss:.4f}) to {best_path}")
        
        # 定期保存历史检查点（不覆盖，保留多个）
        if (epoch + 1) % 100 == 0:
            periodic_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pt')
            torch.save(checkpoint, periodic_path)
            print(f"  [Checkpoint] Saved periodic checkpoint to {periodic_path}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='expert_data_with_images')
    parser.add_argument('--camera_names', nargs='+', default=['frontview', 'agentview'])
    parser.add_argument('--batch_size', type=int, default=128)  # 减小batch_size因为状态输入增加了计算
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--crop_height', type=int, default=216)
    parser.add_argument('--crop_width', type=int, default=288)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=5)
    parser.add_argument('--use_states', action='store_true', default=True, help='使用状态输入')
    parser.add_argument('--no_states', action='store_true', help='禁用状态输入（纯视觉）')
    # 新增EMA和学习率warmup参数
    parser.add_argument('--use_ema_warmup', action='store_true', help='启用EMA动态warmup (power=0.8)')
    parser.add_argument('--ema_warmup_power', type=float, default=0.8, help='EMA warmup曲线power值 (默认0.8)')
    parser.add_argument('--use_lr_warmup', action='store_true', help='启用学习率warmup (按steps, 与原文一致)')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help='学习率warmup的steps数 (默认500, 与原文一致)')
    parser.add_argument('--resume', type=str, default=None, help='断点续训：checkpoint文件路径')
    
    args = parser.parse_args()
    
    # 处理 use_states 参数
    use_states = not args.no_states
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Image crop size: {args.crop_height}x{args.crop_width}")
    print(f"Use states: {use_states}")
    
    # 创建完整数据集
    full_dataset = VisionDatasetWithAug(
        data_dir=args.data_dir,
        camera_names=args.camera_names,
        pred_horizon=16,
        obs_horizon=2,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        train_mode=True,
        use_states=use_states,
    )
    
    # 划分数据集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 创建DataLoader (num_workers=0 避免 RoboSuite 多进程段错误)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=False
    ) if val_size > 0 else None
    
    # 获取状态维度
    state_dim = full_dataset.state_dim if use_states else None
    if use_states:
        print(f"State dimension: {state_dim}")
    
    # 创建模型
    model = ConditionalUNet1DWithVision(
        action_dim=14,
        pred_horizon=16,
        camera_names=args.camera_names,
        image_shape=(3, args.crop_height, args.crop_width),
        vision_embed_dim=256,
        state_dim=state_dim,
        state_embed_dim=128,
        down_dims=[256, 512, 1024],
        time_dim=128,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 创建训练器
    trainer = VisionDiffusionPolicyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_ema=True,
        eval_every=args.eval_every,
        num_eval_episodes=args.num_eval_episodes,
        camera_names=args.camera_names,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        # 新增warmup参数
        use_ema_warmup=args.use_ema_warmup,
        ema_warmup_power=args.ema_warmup_power,
        use_lr_warmup=args.use_lr_warmup,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    
    # 断点续训
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == '__main__':
    main()
