"""
Conditional 1D U-Net for Diffusion Policy
接收带噪动作 + 观测条件 + 时间步，预测噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 - 用于时间步编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) 时间步张量
        Returns:
            (B, dim) 位置编码
        """
        device = x.device
        half_dim = self.dim // 2
        
        # 计算位置编码
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # (B, 1) * (1, half_dim) -> (B, half_dim)
        
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return emb


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    将条件特征映射为 scale 和 shift，注入到卷积特征中
    """
    
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, feature_dim)
        self.shift_proj = nn.Linear(cond_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) 卷积特征
            cond: (B, cond_dim) 条件特征
        Returns:
            (B, C, L) 调制后的特征
        """
        scale = self.scale_proj(cond)  # (B, C)
        shift = self.shift_proj(cond)  # (B, C)
        
        # 扩展维度以匹配 x
        scale = scale.unsqueeze(-1)  # (B, C, 1)
        shift = shift.unsqueeze(-1)  # (B, C, 1)
        
        # FiLM: out = x * (1 + scale) + shift
        return x * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    """
    1D 残差块，包含 FiLM 条件注入
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.film = FiLM(cond_dim, out_channels)
        
        self.act = nn.SiLU()  # Swish 激活
        
        # 如果输入输出维度不匹配，使用 1x1 卷积进行投影
        if in_channels != out_channels or stride != 1:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, L)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, L)
        """
        residual = self.residual_proj(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.film(out, cond)  # 注入条件
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        
        return out + residual


class ConditionalUNet1D(nn.Module):
    """
    用于 Diffusion Policy 的条件 1D U-Net
    
    架构:
    - 条件编码器: 将 obs 编码为全局条件
    - 时间步编码器: Sinusoidal 编码
    - 下采样路径: 1D Conv + ResidualBlock
    - 上采样路径: 1D ConvTranspose + ResidualBlock + Skip Connection
    """
    
    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        obs_horizon: int,
        state_dim: int,
        hidden_dim: int = 128,
        down_dims: list = [128, 256, 512],
        cond_dim: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.state_dim = state_dim
        
        # ========== 条件编码器 ==========
        # 将 obs (B, obs_horizon, state_dim) -> global_cond (B, cond_dim)
        obs_input_dim = obs_horizon * state_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        
        # ========== 时间步编码器 ==========
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        
        # 最终条件特征维度
        self.cond_dim = cond_dim + time_dim
        
        # ========== 输入投影 ==========
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], kernel_size=3, padding=1)
        
        # ========== 下采样路径 (Encoder) ==========
        self.down_blocks = nn.ModuleList()
        in_channels = down_dims[0]
        for out_channels in down_dims[1:]:
            self.down_blocks.append(nn.ModuleDict({
                'resnet': ResidualBlock1D(in_channels, in_channels, self.cond_dim),
                'downsample': nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            }))
            in_channels = out_channels
        
        # 中间层
        self.mid_block = ResidualBlock1D(down_dims[-1], down_dims[-1], self.cond_dim)
        
        # ========== 上采样路径 (Decoder) ==========
        self.up_blocks = nn.ModuleList()
        reversed_dims = list(reversed(down_dims))
        for i in range(len(reversed_dims) - 1):
            in_channels = reversed_dims[i]
            out_channels = reversed_dims[i + 1]
            
            self.up_blocks.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose1d(
                    in_channels, out_channels, 
                    kernel_size=4, stride=2, padding=1
                ),
                'resnet': ResidualBlock1D(out_channels * 2, out_channels, self.cond_dim),  # *2 for skip
            }))
        
        # ========== 输出投影 ==========
        self.output_proj = nn.Conv1d(down_dims[0], action_dim, kernel_size=3, padding=1)
    
    def encode_condition(self, obs: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        编码条件: obs + timestep
        
        Args:
            obs: (B, obs_horizon, state_dim)
            timestep: (B,)
        Returns:
            cond: (B, cond_dim + time_dim)
        """
        # 编码 obs
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)  # (B, obs_horizon * state_dim)
        obs_cond = self.obs_encoder(obs_flat)  # (B, cond_dim)
        
        # 编码 timestep
        time_emb = self.time_encoder(timestep)  # (B, time_dim)
        
        # 拼接条件
        cond = torch.cat([obs_cond, time_emb], dim=-1)  # (B, cond_dim + time_dim)
        return cond
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noisy_actions: (B, pred_horizon, action_dim)
            timestep: (B,)
            obs: (B, obs_horizon, state_dim)
        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        # 编码条件
        cond = self.encode_condition(obs, timestep)  # (B, cond_dim + time_dim)
        
        # 调整维度以适应 1D 卷积: (B, C, L)
        x = noisy_actions.permute(0, 2, 1)  # (B, action_dim, pred_horizon)
        
        # 输入投影
        x = self.input_proj(x)  # (B, down_dims[0], pred_horizon)
        
        # 下采样并保存 skip connections
        skip_features = []
        for block in self.down_blocks:
            x = block['resnet'](x, cond)  # ResNet + FiLM
            skip_features.append(x)
            x = block['downsample'](x)    # 下采样
        
        # 中间层
        x = self.mid_block(x, cond)
        
        # 上采样 + Skip Connections
        for i, block in enumerate(self.up_blocks):
            x = block['upsample'](x)  # 上采样
            skip = skip_features[-(i + 1)]  # 对应的 skip feature
            
            # 确保尺寸匹配
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)  # 拼接 skip
            x = block['resnet'](x, cond)  # ResNet + FiLM
        
        # 输出投影
        x = self.output_proj(x)  # (B, action_dim, pred_horizon)
        
        # 恢复维度
        noise_pred = x.permute(0, 2, 1)  # (B, pred_horizon, action_dim)
        
        return noise_pred


if __name__ == "__main__":
    """测试网络前向传播"""
    print("="*60)
    print("Testing Conditional 1D U-Net")
    print("="*60)
    
    # 测试参数 (匹配你的数据形状)
    B = 32              # Batch size
    pred_horizon = 16   # 动作预测长度
    action_dim = 14     # 双臂动作维度
    obs_horizon = 2     # 观测历史长度
    state_dim = 16      # 观测特征维度 (假设 2个arms * 8维特征)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建网络
    model = ConditionalUNet1D(
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        state_dim=state_dim,
        hidden_dim=128,
        down_dims=[128, 256, 512],
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 创建测试输入
    noisy_actions = torch.randn(B, pred_horizon, action_dim).to(device)
    timestep = torch.randint(0, 100, (B,)).to(device)
    obs = torch.randn(B, obs_horizon, state_dim).to(device)
    
    print(f"\nInput shapes:")
    print(f"  noisy_actions: {noisy_actions.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  obs: {obs.shape}")
    
    # 前向传播
    with torch.no_grad():
        noise_pred = model(noisy_actions, timestep, obs)
    
    print(f"\nOutput shape: {noise_pred.shape}")
    print(f"Expected shape: ({B}, {pred_horizon}, {action_dim})")
    
    assert noise_pred.shape == (B, pred_horizon, action_dim), "Output shape mismatch!"
    print("\n✅ Forward pass successful!")
    print(f"   Output range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
