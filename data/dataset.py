"""
Robosuite Diffusion Policy Dataset
支持 observation horizon 和 action horizon 的滑动窗口切片
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class RobosuiteDiffusionDataset(Dataset):
    """
    Robosuite HDF5 数据集加载器
    
    Args:
        data_dir: HDF5 文件目录
        observation_keys: 需要从 HDF5 提取的观测 key 列表
        obs_horizon: 观测历史长度 (默认 2)
        pred_horizon: 动作预测长度 (默认 16)
        normalize_stats: 预计算的归一化统计量 (可选)
    """
    
    def __init__(
        self,
        data_dir: str,
        observation_keys: List[str],
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        normalize_stats: Dict = None,
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.observation_keys = observation_keys
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # 加载所有演示数据
        self.demos = self._load_demos()
        
        # 构建滑动窗口索引
        self.indices = self._build_indices()
        
        # 计算或加载归一化统计量
        if normalize_stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = normalize_stats
    
    def _load_demos(self) -> List[Dict]:
        """加载所有 HDF5 演示文件"""
        demos = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(('.h5', '.hdf5')):
                continue
                
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                with h5py.File(filepath, 'r') as f:
                    # 标准 Robosuite 格式: /data/demo_*/
                    if 'data' in f:
                        data_group = f['data']
                        for demo_key in data_group.keys():
                            if demo_key.startswith('demo_'):
                                demo_group = data_group[demo_key]
                                
                                # 加载动作
                                actions = np.array(demo_group['actions'])
                                
                                # 处理单臂动作 (7维) -> 双臂 (14维)
                                if actions.shape[-1] == 7:
                                    actions = self._expand_action_dim(actions)
                                
                                # 加载状态 (observations)
                                if 'states' in demo_group:
                                    states = np.array(demo_group['states'])
                                else:
                                    # 如果没有 states，尝试用 observations 组
                                    states = self._extract_observations(demo_group)
                                
                                # 对轨迹进行 padding，保留末尾数据
                                states, actions = self._pad_trajectory(states, actions)
                                
                                demos.append({
                                    'states': states,      # [T, state_dim]
                                    'actions': actions,    # [T, action_dim]
                                    'length': len(states),
                                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(demos)} valid demonstrations")
        return demos
    
    def _expand_action_dim(self, actions: np.ndarray) -> np.ndarray:
        """将单臂动作 (7维) 扩展为双臂动作 (14维)"""
        dual_actions = np.zeros((actions.shape[0], 14))
        dual_actions[:, :7] = actions
        return dual_actions
    
    def _pad_trajectory(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对轨迹进行 padding，使其长度至少为 obs_horizon + pred_horizon
        用最后一步的状态/动作进行复制填充
        """
        T = len(states)
        min_len = self.obs_horizon + self.pred_horizon
        
        if T >= min_len:
            return states, actions
        
        # 需要 padding 的长度
        pad_len = min_len - T
        
        # 用最后一步进行 padding
        last_state = states[-1:]      # [1, state_dim]
        last_action = actions[-1:]    # [1, action_dim]
        
        state_pad = np.repeat(last_state, pad_len, axis=0)
        action_pad = np.repeat(last_action, pad_len, axis=0)
        
        states = np.concatenate([states, state_pad], axis=0)
        actions = np.concatenate([actions, action_pad], axis=0)
        
        return states, actions
    
    def _extract_observations(self, demo_group) -> np.ndarray:
        """
        提取观测状态，优先使用完整的 'states' (50维)，
        如果不存在则根据 observation_keys 从 observations 组中提取并拼接
        """
        # 优先使用完整的 states (50维，包含所有信息)
        if 'states' in demo_group:
            states = np.array(demo_group['states'])
            print(f"  Using full states: shape={states.shape}")
            return states
        
        # 如果没有 states，则根据 observation_keys 拼接
        if 'observations' not in demo_group:
            raise ValueError("No 'observations' or 'states' group found in demo!")
            
        obs_group = demo_group['observations']
        obs_parts = []
        
        # 严格按照给定的 keys 顺序提取
        for key in self.observation_keys:
            if key not in obs_group:
                raise KeyError(f"Required observation key '{key}' not found in data!")
            
            # 提取数据并确保是 2D 数组 (T, dim)
            val = np.array(obs_group[key])
            if len(val.shape) == 1:
                val = val.reshape(-1, 1)  # 把标量变成 (T, 1)
            obs_parts.append(val)
            
        # 在特征维度上进行拼接
        concatenated = np.concatenate(obs_parts, axis=-1)
        print(f"  Using observation_keys: shape={concatenated.shape}")
        return concatenated
    
    def _build_indices(self) -> List[Tuple[int, int]]:
        """
        构建滑动窗口索引
        每个元素: (demo_idx, start_idx)
        """
        indices = []
        
        for demo_idx, demo in enumerate(self.demos):
            length = demo['length']
            
            # 滑动窗口: 从 obs_horizon-1 开始，确保前面有足够的历史
            # 到 length - pred_horizon 结束，确保后面有足够的未来动作
            for start_idx in range(self.obs_horizon - 1, length - self.pred_horizon + 1):
                indices.append((demo_idx, start_idx))
        
        print(f"Built {len(indices)} training samples")
        return indices
    
    def _compute_stats(self) -> Dict:
        """
        计算归一化统计量
        - 状态: 使用 z-score 标准化 (mean, std)
        - 动作: 使用 Min-Max 归一化到 [-1, 1] (min, max)
        """
        # 收集所有状态和动作
        all_states = []
        all_actions = []
        
        for demo in self.demos:
            all_states.append(demo['states'])
            all_actions.append(demo['actions'])
        
        all_states = np.concatenate(all_states, axis=0)    # [N, state_dim]
        all_actions = np.concatenate(all_actions, axis=0)  # [N, action_dim]
        
        # 状态 z-score 标准化
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0) + 1e-8  # 防止除零
        
        # 动作 Min-Max 归一化到 [-1, 1]
        action_min = np.min(all_actions, axis=0)
        action_max = np.max(all_actions, axis=0)
        # 防止除零和极端值
        action_range = (action_max - action_min) + 1e-8
        
        stats = {
            'state_mean': state_mean,
            'state_std': state_std,
            'action_min': action_min,
            'action_max': action_max,
            'action_range': action_range,
            # 预计算用于快速反归一化的参数
            'action_scale': action_range / 2.0,
            'action_offset': (action_min + action_max) / 2.0,
        }
        
        print(f"Statistics computed:")
        print(f"  States: mean range [{state_mean.min():.3f}, {state_mean.max():.3f}], "
              f"std range [{state_std.min():.3f}, {state_std.max():.3f}]")
        print(f"  Actions: min range [{action_min.min():.3f}, {action_min.max():.3f}], "
              f"max range [{action_max.min():.3f}, {action_max.max():.3f}]")
        
        return stats
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """z-score 标准化状态"""
        return (state - self.stats['state_mean']) / self.stats['state_std']
    
    def _denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """反标准化状态"""
        return state * self.stats['state_std'] + self.stats['state_mean']
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Min-Max 归一化动作到 [-1, 1]"""
        return (action - self.stats['action_offset']) / self.stats['action_scale']
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反归一化动作"""
        return action * self.stats['action_scale'] + self.stats['action_offset']
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Returns:
            dict with keys:
                - obs: 观测历史 [obs_horizon, state_dim]
                - action: 动作序列 [pred_horizon, action_dim]
        """
        demo_idx, start_idx = self.indices[idx]
        demo = self.demos[demo_idx]
        
        # 提取观测历史: [start_idx - obs_horizon + 1, start_idx]
        obs_start = start_idx - self.obs_horizon + 1
        obs_end = start_idx + 1
        obs = demo['states'][obs_start:obs_end]  # [obs_horizon, state_dim]
        
        # 提取动作序列: [start_idx, start_idx + pred_horizon]
        act_start = start_idx
        act_end = start_idx + self.pred_horizon
        action = demo['actions'][act_start:act_end]  # [pred_horizon, action_dim]
        
        # 归一化
        obs = self._normalize_state(obs)
        action = self._normalize_action(action)
        
        return {
            'obs': torch.from_numpy(obs).float(),
            'action': torch.from_numpy(action).float(),
        }
    
    def get_stats(self) -> Dict:
        """获取归一化统计量，用于保存和后续推理"""
        return self.stats


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批量数据整理函数
    将 list of dicts 转换为 dict of tensors
    """
    obs = torch.stack([item['obs'] for item in batch])
    action = torch.stack([item['action'] for item in batch])
    
    return {
        'obs': obs,      # [B, obs_horizon, state_dim]
        'action': action, # [B, pred_horizon, action_dim]
    }


if __name__ == "__main__":
    # ==================== 测试数据集 ====================
    print("=" * 70)
    print("Dataset Loading Test - 检测训练数据来源")
    print("=" * 70)
    
    data_dir = "./expert_data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please collect demonstration data first!")
        exit(1)
    
    # 测试1: 使用完整的 states (50维，推荐方式)
    print("\n" + "=" * 70)
    print("Test 1: 使用 observation_keys=['states'] (推荐，50维完整状态)")
    print("=" * 70)
    
    dataset_with_states = RobosuiteDiffusionDataset(
        data_dir=data_dir,
        observation_keys=["states"],  # 直接使用 states
        obs_horizon=2,
        pred_horizon=16,
    )
    
    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {len(dataset_with_states)}")
    print(f"  状态维度: {dataset_with_states.stats['state_mean'].shape[0]}")
    print(f"  动作维度: {dataset_with_states.stats['action_scale'].shape[0]}")
    
    sample1 = dataset_with_states[0]
    print(f"\n📋 样本形状:")
    print(f"  obs shape: {sample1['obs'].shape} (obs_horizon, state_dim)")
    print(f"  action shape: {sample1['action'].shape} (pred_horizon, action_dim)")
    print(f"  obs range: [{sample1['obs'].min():.3f}, {sample1['obs'].max():.3f}]")
    print(f"  action range: [{sample1['action'].min():.3f}, {sample1['action'].max():.3f}]")
    
    # 测试2: 使用 observation_keys 手动拼接 (16维，旧方式)
    print("\n" + "=" * 70)
    print("Test 2: 使用 observation_keys 手动拼接 (旧方式，16维)")
    print("=" * 70)
    
    observation_keys_manual = [
        "robot0_eef_pos",      # 左臂末端位置 (3,)
        "robot0_eef_quat",     # 左臂末端姿态 (4,)
        "robot0_gripper_qpos", # 左臂夹爪 (1,)
        "robot1_eef_pos",      # 右臂末端位置 (3,)
        "robot1_eef_quat",     # 右臂末端姿态 (4,)
        "robot1_gripper_qpos", # 右臂夹爪 (1,)
    ]
    
    try:
        dataset_manual = RobosuiteDiffusionDataset(
            data_dir=data_dir,
            observation_keys=observation_keys_manual,
            obs_horizon=2,
            pred_horizon=16,
        )
        
        print(f"\n📊 数据集统计:")
        print(f"  总样本数: {len(dataset_manual)}")
        print(f"  状态维度: {dataset_manual.stats['state_mean'].shape[0]}")
        
        sample2 = dataset_manual[0]
        print(f"\n📋 样本形状:")
        print(f"  obs shape: {sample2['obs'].shape} (obs_horizon, state_dim)")
        print(f"  action shape: {sample2['action'].shape} (pred_horizon, action_dim)")
        
    except Exception as e:
        print(f"\n❌ 手动拼接方式失败: {e}")
        print("   说明数据中可能没有 observations 组，只有 states")
    
    # 测试3: DataLoader 测试
    print("\n" + "=" * 70)
    print("Test 3: DataLoader 批量加载测试")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset_with_states,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    print(f"\n📦 Batch shapes:")
    print(f"  obs: {batch['obs'].shape} (B, obs_horizon, state_dim)")
    print(f"  action: {batch['action'].shape} (B, pred_horizon, action_dim)")
    
    # 总结
    print("\n" + "=" * 70)
    print("Summary - 数据来源检测结果")
    print("=" * 70)
    print(f"✅ 完整 states 方式: state_dim = {dataset_with_states.stats['state_mean'].shape[0]}")
    print(f"   包含: 关节位置/速度、末端位姿、夹爪状态、物体状态等完整信息")
    try:
        print(f"⚠️  手动拼接方式: state_dim = {dataset_manual.stats['state_mean'].shape[0]}")
        print(f"   仅包含: 末端位姿、夹爪状态 (信息丢失严重)")
    except:
        print(f"   手动拼接方式不可用（数据中无 observations 组）")
    print("=" * 70)
