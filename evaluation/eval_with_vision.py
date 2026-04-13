"""
评估支持图像和状态输入的 Diffusion Policy，并生成视频
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from pathlib import Path
import robosuite as suite

from network_with_vision import ConditionalUNet1DWithVision


class EvalImageTransform:
    """评估时的图像变换（中心裁剪，与训练一致）
    
    使用 216x288 裁剪（240x320 的 90%）
    """
    
    def __init__(self, crop_height: int = 216, crop_width: int = 288):
        self.crop_height = crop_height
        self.crop_width = crop_width
        
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Args:
            img: (H, W, 3) numpy array in [0, 1]
        Returns:
            tensor: (3, crop_height, crop_width)
        """
        # 转换为 tensor
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_tensor = img.permute(2, 0, 1).float()
        
        _, H, W = img_tensor.shape
        
        # 中心裁剪
        if H > self.crop_height and W > self.crop_width:
            top = (H - self.crop_height) // 2
            left = (W - self.crop_width) // 2
            img_tensor = img_tensor[:, top:top+self.crop_height, left:left+self.crop_width]
        else:
            # Resize 到目标尺寸
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.crop_height, self.crop_width),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_tensor - mean) / std


class DiffusionScheduler:
    """扩散调度器（用于推理）"""
    
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_steps = num_steps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    @torch.no_grad()
    def sample(self, model, obs_images, obs_states, pred_horizon, action_dim):
        """DDPM 采样生成动作序列"""
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
    """观测缓冲区"""
    
    def __init__(self, obs_horizon, camera_names, state_dim=None):
        self.obs_horizon = obs_horizon
        self.camera_names = camera_names
        self.state_dim = state_dim
        
        self.buffers = {cam: [] for cam in camera_names}
        self.state_buffer = [] if state_dim is not None else None
    
    def add_obs(self, obs_dict):
        """添加观测"""
        for cam in self.camera_names:
            if cam in obs_dict:
                self.buffers[cam].append(obs_dict[cam].copy())
                if len(self.buffers[cam]) > self.obs_horizon:
                    self.buffers[cam].pop(0)
        
        if self.state_buffer is not None and 'state' in obs_dict:
            self.state_buffer.append(obs_dict['state'].copy())
            if len(self.state_buffer) > self.obs_horizon:
                self.state_buffer.pop(0)
    
    def get_stacked_obs(self, transform=None, crop_height=216, crop_width=288):
        """获取堆叠的观测"""
        result = {}
        
        # 处理图像
        for cam in self.camera_names:
            while len(self.buffers[cam]) < self.obs_horizon:
                if len(self.buffers[cam]) == 0:
                    empty = np.zeros((crop_height, crop_width, 3), dtype=np.float32)
                    self.buffers[cam].append(empty)
                else:
                    self.buffers[cam].insert(0, self.buffers[cam][0])
            
            frames = self.buffers[cam][-self.obs_horizon:]
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


def create_env_with_camera(env_config, camera_names):
    """创建带有相机观测的环境"""
    env = suite.make(
        env_name=env_config.env_name,
        robots=env_config.robots,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=240,
        camera_widths=320,
        camera_depths=False,
        horizon=env_config.horizon,
        control_freq=env_config.control_freq,
    )
    return env


def save_video(frames, output_path, fps=20, flip_vertical=True):
    """保存视频"""
    if len(frames) == 0:
        print(f"Warning: No frames to save to {output_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    first_frame = frames[0]
    if flip_vertical:
        first_frame = np.flipud(first_frame)
    
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if flip_vertical:
            frame = np.flipud(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video: {output_path}")


def evaluate_episode(env, model, scheduler, camera_names, obs_horizon, 
                     pred_horizon, action_horizon, max_steps, device,
                     crop_height=216, crop_width=288,
                     normalization_stats=None, use_states=False, state_dim=None,
                     debug=False):
    """
    评估一个 episode
    
    Args:
        normalization_stats: 包含 action_min, action_max 等归一化统计信息
        use_states: 是否使用状态输入
        state_dim: 状态维度
    """
    obs = env.reset()
    obs_buffer = ObsBuffer(obs_horizon, camera_names, state_dim=state_dim)
    
    # 处理初始观测
    obs_images = {}
    for cam in camera_names:
        img_key = f"{cam}_image"
        if img_key in obs:
            obs_images[cam] = obs[img_key]
    
    # 处理初始状态
    if use_states:
        if 'robot0_proprio-state' in obs:
            obs_state = obs['robot0_proprio-state']
        elif 'states' in obs:
            obs_state = obs['states']
        else:
            # 从观测构造状态
            obs_state = np.concatenate([
                obs.get('robot0_eef_pos', np.zeros(3)),
                obs.get('robot0_eef_quat', np.zeros(4)),
                obs.get('robot0_gripper_qpos', np.zeros(1)),
                obs.get('robot1_eef_pos', np.zeros(3)),
                obs.get('robot1_eef_quat', np.zeros(4)),
                obs.get('robot1_gripper_qpos', np.zeros(1)),
            ])
        
        # 归一化状态
        if normalization_stats is not None:
            state_min = normalization_stats['state_min']
            state_max = normalization_stats['state_max']
            scale = 2.0 / (state_max - state_min)
            offset = -1.0 - scale * state_min
            obs_state = obs_state * scale + offset
        
        obs_images['state'] = obs_state
    
    obs_buffer.add_obs(obs_images)
    
    frames = []
    action_queue = []
    step_idx = 0
    done = False
    success = False
    executed_actions = []
    
    transform = EvalImageTransform(crop_height=crop_height, crop_width=crop_width)
    
    while step_idx < max_steps and not done:
        if len(action_queue) == 0:
            obs_tensors = obs_buffer.get_stacked_obs(transform, crop_height, crop_width)
            
            obs_imgs = {k: v.to(device) for k, v in obs_tensors.items() if k != 'state'}
            obs_state_tensor = obs_tensors.get('state', None)
            if obs_state_tensor is not None:
                obs_state_tensor = obs_state_tensor.to(device)
            
            action_seq = scheduler.sample(model, obs_imgs, obs_state_tensor, pred_horizon, env.action_dim)
            
            # 反归一化动作
            action_seq_np = action_seq.cpu().numpy()
            if normalization_stats is not None:
                action_min = normalization_stats['action_min']
                action_max = normalization_stats['action_max']
                scale = 2.0 / (action_max - action_min)
                offset = -1.0 - scale * action_min
                action_seq_np = (action_seq_np - offset) / scale
            
            action_queue = list(action_seq_np[:action_horizon])
            
            if debug:
                print(f"Step {step_idx}: Generated {len(action_queue)} actions")
        
        action = action_queue.pop(0)
        executed_actions.append(action)
        
        obs, reward, done, info = env.step(action)
        
        if info.get('success', False):
            success = True
        
        # 记录帧
        img_key = f"{camera_names[0]}_image"
        if img_key in obs:
            frames.append(obs[img_key].copy())
        
        # 更新观测缓冲区
        obs_images = {}
        for cam in camera_names:
            img_key = f"{cam}_image"
            if img_key in obs:
                obs_images[cam] = obs[img_key]
        
        if use_states:
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
            
            # 归一化状态
            if normalization_stats is not None:
                state_min = normalization_stats['state_min']
                state_max = normalization_stats['state_max']
                scale = 2.0 / (state_max - state_min)
                offset = -1.0 - scale * state_min
                obs_state = obs_state * scale + offset
            
            obs_images['state'] = obs_state
        
        obs_buffer.add_obs(obs_images)
        step_idx += 1
    
    return frames, executed_actions, success


def main():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Policy with Vision and State')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_videos',
                        help='Directory to save evaluation videos')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--camera_names', nargs='+', default=['frontview', 'agentview'],
                        help='Camera names to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--pred_horizon', type=int, default=16,
                        help='Prediction horizon')
    parser.add_argument('--action_horizon', type=int, default=8,
                        help='Action horizon')
    parser.add_argument('--obs_horizon', type=int, default=2,
                        help='Observation horizon')
    parser.add_argument('--fps', type=int, default=20,
                        help='Video frame rate')
    parser.add_argument('--no_flip', action='store_true',
                        help='Do not flip video vertically')
    parser.add_argument('--crop_height', type=int, default=216,
                        help='Image crop height (must match training)')
    parser.add_argument('--crop_width', type=int, default=288,
                        help='Image crop width (must match training)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载检查点
    print("="*60)
    print("Loading model...")
    print("="*60)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 获取归一化统计信息
    normalization_stats = checkpoint.get('normalization_stats', None)
    if normalization_stats is None:
        print("Warning: No normalization stats found in checkpoint!")
        print("Actions will not be unnormalized properly.")
    else:
        print("Loaded normalization stats from checkpoint")
        use_states = 'state_min' in normalization_stats
        if use_states:
            print(f"  Using state input (dim: {len(normalization_stats['state_min'])})")
        else:
            print("  Using vision only (no states)")
    
    use_states = normalization_stats is not None and 'state_min' in normalization_stats
    state_dim = len(normalization_stats['state_min']) if use_states else None
    
    # 创建模型
    model = ConditionalUNet1DWithVision(
        action_dim=14,
        pred_horizon=args.pred_horizon,
        camera_names=args.camera_names,
        image_shape=(3, 240, 320),
        vision_embed_dim=256,
        state_dim=state_dim,
        state_embed_dim=128,
        down_dims=[256, 512, 1024],
        time_dim=128,
    ).to(device)
    
    # 加载模型权重
    if 'ema_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print(f"Loaded EMA model from {args.checkpoint}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.checkpoint}")
    
    model.eval()
    
    # 创建扩散调度器
    scheduler = DiffusionScheduler(num_steps=100, device=device)
    
    # 创建环境
    print("\nCreating environment...")
    from config import get_default_config
    config = get_default_config()
    env = create_env_with_camera(config.env, args.camera_names)
    
    print("="*60)
    print(f"Evaluating for {args.num_episodes} episodes")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Prediction horizon: {args.pred_horizon}")
    print(f"Action horizon: {args.action_horizon}")
    print("="*60)
    
    # 评估
    results = []
    
    for ep_idx in range(args.num_episodes):
        print(f"\nEpisode {ep_idx + 1}/{args.num_episodes}")
        
        frames, actions, success = evaluate_episode(
            env, model, scheduler, 
            args.camera_names, args.obs_horizon, 
            args.pred_horizon, args.action_horizon,
            args.max_steps, device,
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            normalization_stats=normalization_stats,
            use_states=use_states,
            state_dim=state_dim,
            debug=(ep_idx == 0)
        )
        
        # 保存视频
        video_path = os.path.join(args.output_dir, f"episode_{ep_idx+1}_{'success' if success else 'fail'}.mp4")
        save_video(frames, video_path, args.fps, flip_vertical=not args.no_flip)
        
        # 记录结果
        results.append({
            'episode': ep_idx + 1,
            'success': success,
            'length': len(actions),
        })
        
        print(f"  Success: {success}, Steps: {len(actions)}")
    
    env.close()
    
    # 打印总结
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    successes = [r['success'] for r in results]
    success_rate = sum(successes) / len(successes)
    avg_length = np.mean([r['length'] for r in results])
    
    print(f"Success Rate: {success_rate*100:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Videos saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
