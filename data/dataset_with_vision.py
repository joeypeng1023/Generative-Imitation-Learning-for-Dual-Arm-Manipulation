"""
支持图像和状态观测的数据集类
从 HDF5 文件加载图像、状态和动作数据
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, List, Optional, Tuple


class VisionDiffusionDataset(Dataset):
    """
    支持多相机图像和状态观测的 Diffusion Policy 数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        camera_names: List[str] = ['frontview', 'agentview'],
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        image_size: tuple = (240, 320),
        normalize_images: bool = True,
        normalize_actions: bool = True,
        use_states: bool = True,  # 新增：是否使用状态
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.camera_names = camera_names
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_states = use_states
        
        # 加载所有 HDF5 文件
        self.episodes = self._load_episodes()
        
        # 构建样本索引
        self.samples = self._build_samples()
        
        # 图像预处理
        if normalize_images:
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = transforms.ToTensor()
        
        # 动作归一化参数 (min-max 到 [-1, 1])
        self.normalize_actions = normalize_actions
        if normalize_actions:
            self.action_min, self.action_max = self._compute_action_minmax()
        else:
            self.action_min, self.action_max = None, None
        
        # 状态归一化参数 (min-max 到 [-1, 1])
        if self.use_states:
            self.state_min, self.state_max = self._compute_state_minmax()
            # 从第一个 episode 获取状态维度
            if len(self.episodes) > 0 and 'states' in self.episodes[0]:
                self.state_dim = self.episodes[0]['states'].shape[-1]
            else:
                self.state_dim = None
        else:
            self.state_min, self.state_max = None, None
            self.state_dim = None
    
    def _load_episodes(self) -> List[Dict]:
        """加载所有 episode 数据"""
        episodes = []
        
        hdf5_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')])
        
        print(f"Loading {len(hdf5_files)} HDF5 files from {self.data_dir}")
        
        for filename in hdf5_files:
            filepath = os.path.join(self.data_dir, filename)
            
            with h5py.File(filepath, 'r') as f:
                if 'data' not in f:
                    continue
                
                data_group = f['data']
                
                for demo_key in data_group.keys():
                    if not demo_key.startswith('demo_'):
                        continue
                    
                    demo = data_group[demo_key]
                    
                    # 加载动作
                    actions = np.array(demo['actions'])  # (T, action_dim)
                    
                    # 加载状态 (如果启用)
                    states = None
                    if self.use_states:
                        if 'states' in demo:
                            states = np.array(demo['states'])  # (T, state_dim)
                        else:
                            print(f"  Warning: {demo_key} in {filename} has no states, skipping")
                            continue
                    
                    # 加载图像
                    if 'observations' not in demo:
                        continue
                    
                    obs_group = demo['observations']
                    
                    # 检查是否有所有需要的相机图像
                    has_all_cameras = all(
                        f'{cam}_image' in obs_group for cam in self.camera_names
                    )
                    
                    if not has_all_cameras:
                        missing = [cam for cam in self.camera_names 
                                  if f'{cam}_image' not in obs_group]
                        print(f"  Skipping {demo_key} in {filename}: missing {missing}")
                        continue
                    
                    # 加载所有相机的图像
                    images = {}
                    for cam in self.camera_names:
                        img_key = f'{cam}_image'
                        img_data = np.array(obs_group[img_key])
                        images[cam] = img_data
                    
                    episode = {
                        'filename': filename,
                        'demo_key': demo_key,
                        'actions': actions,
                        'images': images,
                        'length': len(actions),
                    }
                    
                    if self.use_states:
                        episode['states'] = states
                    
                    episodes.append(episode)
        
        print(f"Loaded {len(episodes)} valid episodes")
        if self.use_states and len(episodes) > 0:
            print(f"  Using states with dim: {episodes[0]['states'].shape[-1]}")
        return episodes
    
    def _build_samples(self) -> List[Dict]:
        """构建样本索引"""
        samples = []
        
        for ep_idx, episode in enumerate(self.episodes):
            length = episode['length']
            
            # 滑动窗口采样
            for start_idx in range(length - self.pred_horizon + 1):
                obs_start = max(0, start_idx - self.obs_horizon + 1)
                obs_end = start_idx + 1
                
                pred_start = start_idx
                pred_end = start_idx + self.pred_horizon
                
                samples.append({
                    'episode_idx': ep_idx,
                    'obs_start': obs_start,
                    'obs_end': obs_end,
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                })
        
        print(f"Total samples: {len(samples)}")
        return samples
    
    def _compute_action_minmax(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算动作的 min 和 max (用于 min-max 归一化到 [-1, 1])"""
        all_actions = []
        for episode in self.episodes:
            all_actions.append(episode['actions'])
        
        all_actions = np.concatenate(all_actions, axis=0)
        action_min = np.min(all_actions, axis=0)
        action_max = np.max(all_actions, axis=0)
        
        # 处理常数维度
        range_eps = 1e-4
        action_range = action_max - action_min
        constant_mask = action_range < range_eps
        action_range[constant_mask] = 2.0  # 设为2，这样 scale=1, 保持原值
        action_max[constant_mask] = action_min[constant_mask] + 2.0
        
        print(f"Action range - min: {action_min.mean():.4f}, max: {action_max.mean():.4f}")
        return action_min, action_max
    
    def _compute_state_minmax(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算状态的 min 和 max (用于 min-max 归一化到 [-1, 1])"""
        all_states = []
        for episode in self.episodes:
            all_states.append(episode['states'])
        
        all_states = np.concatenate(all_states, axis=0)
        state_min = np.min(all_states, axis=0)
        state_max = np.max(all_states, axis=0)
        
        # 处理常数维度
        range_eps = 1e-4
        state_range = state_max - state_min
        constant_mask = state_range < range_eps
        state_range[constant_mask] = 2.0
        state_max[constant_mask] = state_min[constant_mask] + 2.0
        
        print(f"State range - min: {state_min.mean():.4f}, max: {state_max.mean():.4f}")
        return state_min, state_max
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """将动作归一化到 [-1, 1]"""
        if not self.normalize_actions:
            return action
        # min-max 归一化: (x - min) / (max - min) * 2 - 1
        scale = 2.0 / (self.action_max - self.action_min)
        offset = -1.0 - scale * self.action_min
        return action * scale + offset
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """将动作从 [-1, 1] 反归一化"""
        if not self.normalize_actions:
            return action
        scale = 2.0 / (self.action_max - self.action_min)
        offset = -1.0 - scale * self.action_min
        return (action - offset) / scale
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """将状态归一化到 [-1, 1]"""
        if not self.use_states:
            return state
        scale = 2.0 / (self.state_max - self.state_min)
        offset = -1.0 - scale * self.state_min
        return state * scale + offset
    
    def unnormalize_state(self, state: np.ndarray) -> np.ndarray:
        """将状态从 [-1, 1] 反归一化"""
        if not self.use_states:
            return state
        scale = 2.0 / (self.state_max - self.state_min)
        offset = -1.0 - scale * self.state_min
        return (state - offset) / scale
    
    def __len__(self):
        return len(self.samples)
    
    def _process_images(self, images_dict: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, torch.Tensor]:
        """处理图像序列
        
        注意：self.image_transform 包含 ToTensor() 会自己做 /255
        所以这里直接传入 uint8 [0,255] 的图像
        """
        result = {}
        
        for cam_name, img_seq in images_dict.items():
            imgs = img_seq[start:end]
            # 注意：ToTensor() 内部会做 /255，所以这里不要提前做
            processed = []
            for img in imgs:
                tensor = self.image_transform(img)  # ToTensor() 内部做 /255
                processed.append(tensor)
            
            result[cam_name] = torch.stack(processed, dim=0)
        
        return result
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            Dict 包含:
                - obs_images: Dict {camera_name: (obs_horizon, 3, H, W)}
                - obs_states: (obs_horizon, state_dim)  (如果 use_states=True)
                - actions: (pred_horizon, action_dim)
        """
        sample_info = self.samples[idx]
        episode = self.episodes[sample_info['episode_idx']]
        
        # 获取观测图像
        obs_images = self._process_images(
            episode['images'],
            sample_info['obs_start'],
            sample_info['obs_end']
        )
        
        # 如果观测帧数不足，用第一帧填充
        for cam_name in obs_images:
            num_frames = obs_images[cam_name].shape[0]
            if num_frames < self.obs_horizon:
                first_frame = obs_images[cam_name][0:1]
                padding = first_frame.repeat(self.obs_horizon - num_frames, 1, 1, 1)
                obs_images[cam_name] = torch.cat([padding, obs_images[cam_name]], dim=0)
        
        result = {'obs_images': obs_images}
        
        # 获取状态序列
        if self.use_states:
            obs_states = episode['states'][sample_info['obs_start']:sample_info['obs_end']]
            obs_states = self.normalize_state(obs_states)
            
            # 填充状态序列
            if len(obs_states) < self.obs_horizon:
                first_state = obs_states[0:1]
                padding = np.repeat(first_state, self.obs_horizon - len(obs_states), axis=0)
                obs_states = np.concatenate([padding, obs_states], axis=0)
            
            result['obs_states'] = torch.tensor(obs_states, dtype=torch.float32)
        
        # 获取动作序列
        actions = episode['actions'][sample_info['pred_start']:sample_info['pred_end']]
        actions = self.normalize_action(actions)
        result['actions'] = torch.tensor(actions, dtype=torch.float32)
        
        return result
    
    def get_normalization_stats(self) -> Dict:
        """获取归一化统计信息用于保存/加载"""
        stats = {
            'action_min': self.action_min,
            'action_max': self.action_max,
            'normalize_actions': self.normalize_actions,
        }
        if self.use_states:
            stats['state_min'] = self.state_min
            stats['state_max'] = self.state_max
        return stats


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 collate 函数，处理多相机图像和状态
    """
    obs_images_dict = {cam: [] for cam in batch[0]['obs_images'].keys()}
    actions_list = []
    has_states = 'obs_states' in batch[0]
    
    if has_states:
        states_list = []
    
    for sample in batch:
        for cam_name, img_tensor in sample['obs_images'].items():
            obs_images_dict[cam_name].append(img_tensor)
        actions_list.append(sample['actions'])
        if has_states:
            states_list.append(sample['obs_states'])
    
    # 堆叠批次
    batch_obs_images = {}
    for cam_name, img_list in obs_images_dict.items():
        batch_obs_images[cam_name] = torch.stack(img_list, dim=0)
    
    batch_actions = torch.stack(actions_list, dim=0)
    
    result = {
        'obs_images': batch_obs_images,
        'actions': batch_actions,
    }
    
    if has_states:
        result['obs_states'] = torch.stack(states_list, dim=0)
    
    return result


if __name__ == "__main__":
    """测试数据集"""
    import sys
    
    data_dir = "expert_data_with_images"
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("="*60)
    print("Testing Vision Dataset with States")
    print("="*60)
    
    dataset = VisionDiffusionDataset(
        data_dir=data_dir,
        camera_names=['frontview', 'agentview'],
        pred_horizon=16,
        obs_horizon=2,
        use_states=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 测试获取一个样本
    sample = dataset[0]
    
    print(f"\nSample shapes:")
    for cam_name, img_tensor in sample['obs_images'].items():
        print(f"  {cam_name}: {img_tensor.shape}")
    if 'obs_states' in sample:
        print(f"  states: {sample['obs_states'].shape}")
    print(f"  actions: {sample['actions'].shape}")
    
    # 测试反归一化
    action_norm = sample['actions'].numpy()
    action_unnorm = dataset.unnormalize_action(action_norm)
    print(f"\nAction normalization check:")
    print(f"  Normalized range: [{action_norm.min():.2f}, {action_norm.max():.2f}]")
    print(f"  Unnormalized sample: {action_unnorm[:3]}")
    
    print("\n✅ Dataset test successful!")
