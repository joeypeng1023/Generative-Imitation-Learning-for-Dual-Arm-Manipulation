"""
Conditional 1D U-Net for Diffusion Policy with Vision + State Encoder
接收带噪动作 + 图像观测 + 状态观测 + 时间步，预测噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict
import torchvision.models as models


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 - 用于时间步编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class StateEncoder(nn.Module):
    """
    状态编码器 - 将低维状态序列编码为特征向量
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_horizon: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.embed_dim = embed_dim
        
        # MLP 编码器
        input_dim = state_dim * obs_horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (B, obs_horizon, state_dim)
        Returns:
            features: (B, embed_dim)
        """
        B = states.shape[0]
        # 展平时间维度
        states_flat = states.reshape(B, -1)  # (B, obs_horizon * state_dim)
        return self.mlp(states_flat)


class VisionEncoder(nn.Module):
    """
    视觉编码器 - 使用 ResNet18 提取图像特征
    支持多个相机输入
    """
    
    def __init__(
        self,
        image_shape: tuple = (3, 240, 320),
        camera_names: List[str] = ['frontview', 'agentview'],
        embed_dim: int = 256,
        use_group_norm: bool = True,
    ):
        super().__init__()
        
        self.camera_names = camera_names
        self.image_shape = image_shape
        self.embed_dim = embed_dim
        
        # 为每个相机创建独立的 ResNet18 编码器
        self.camera_encoders = nn.ModuleDict()
        
        for cam_name in camera_names:
            resnet = models.resnet18(pretrained=False)
            
            if use_group_norm:
                resnet = self._replace_bn_with_gn(resnet)
            
            resnet.fc = nn.Identity()
            self.camera_encoders[cam_name] = resnet
        
        # ResNet18 输出 512 维特征，投影到 embed_dim
        resnet_output_dim = 512 * len(camera_names)
        self.feature_proj = nn.Sequential(
            nn.Linear(resnet_output_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def _replace_bn_with_gn(self, model):
        """将 BatchNorm 替换为 GroupNorm"""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_groups = min(32, module.num_features)
                while module.num_features % num_groups != 0:
                    num_groups -= 1
                gn = nn.GroupNorm(num_groups, module.num_features, affine=True)
                setattr(model, name, gn)
            else:
                self._replace_bn_with_gn(module)
        return model
    
    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: Dict {camera_name: tensor (B, T, C, H, W)}
        Returns:
            features: (B, embed_dim)
        """
        B = list(images.values())[0].shape[0]
        
        camera_features = []
        
        for cam_name in self.camera_names:
            if cam_name not in images:
                raise ValueError(f"Missing camera: {cam_name}")
            
            img_seq = images[cam_name]  # (B, T, C, H, W)
            T = img_seq.shape[1]
            
            img_flat = img_seq.reshape(B * T, *img_seq.shape[2:])
            feat = self.camera_encoders[cam_name](img_flat)
            feat = feat.reshape(B, T, -1)
            feat = feat.mean(dim=1)
            
            camera_features.append(feat)
        
        combined = torch.cat(camera_features, dim=-1)
        output = self.feature_proj(combined)
        
        return output


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, feature_dim)
        self.shift_proj = nn.Linear(cond_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale_proj(cond).unsqueeze(-1)
        shift = self.shift_proj(cond).unsqueeze(-1)
        return x * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    """1D 残差块，包含 FiLM 条件注入"""
    
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
        self.act = nn.SiLU()
        
        if in_channels != out_channels or stride != 1:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.film(out, cond)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        
        return out + residual


class ConditionalUNet1DWithVision(nn.Module):
    """
    支持图像和状态输入的条件 1D U-Net
    """
    
    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        camera_names: List[str] = ['frontview', 'agentview'],
        image_shape: tuple = (3, 240, 320),
        vision_embed_dim: int = 256,
        state_dim: Optional[int] = None,  # 新增：状态维度
        state_embed_dim: int = 128,        # 新增：状态嵌入维度
        down_dims: List[int] = [256, 512, 1024],
        time_dim: int = 128,
        use_group_norm: bool = True,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.camera_names = camera_names
        self.use_states = state_dim is not None
        self.state_dim = state_dim
        
        # ========== 视觉编码器 ==========
        self.vision_encoder = VisionEncoder(
            image_shape=image_shape,
            camera_names=camera_names,
            embed_dim=vision_embed_dim,
            use_group_norm=use_group_norm,
        )
        
        # ========== 状态编码器 ==========
        if self.use_states:
            self.state_encoder = StateEncoder(
                state_dim=state_dim,
                obs_horizon=2,  # 默认 obs_horizon
                embed_dim=state_embed_dim,
            )
            cond_dim = vision_embed_dim + state_embed_dim + time_dim
        else:
            cond_dim = vision_embed_dim + time_dim
        
        self.cond_dim = cond_dim
        
        # ========== 时间步编码器 ==========
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        
        # ========== U-Net 主体 ==========
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        in_channels = down_dims[0]
        for out_channels in down_dims[1:]:
            self.down_blocks.append(nn.ModuleDict({
                'resnet': ResidualBlock1D(in_channels, in_channels, cond_dim),
                'downsample': nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            }))
            in_channels = out_channels
        
        # 中间层
        self.mid_block = ResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim)
        
        # 上采样路径
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
                'resnet': ResidualBlock1D(out_channels * 2, out_channels, cond_dim),
            }))
        
        # 输出投影
        self.output_proj = nn.Conv1d(down_dims[0], action_dim, kernel_size=3, padding=1)
    
    def encode_condition(
        self,
        images: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码条件: 图像 + 状态 + 时间步
        
        Args:
            images: Dict {camera_name: (B, T, C, H, W)}
            timestep: (B,)
            states: (B, obs_horizon, state_dim) - 可选
        Returns:
            cond: (B, cond_dim)
        """
        # 编码图像
        vision_cond = self.vision_encoder(images)  # (B, vision_embed_dim)
        
        # 编码时间步
        time_emb = self.time_encoder(timestep)  # (B, time_dim)
        
        # 拼接条件
        if self.use_states and states is not None:
            state_cond = self.state_encoder(states)  # (B, state_embed_dim)
            cond = torch.cat([vision_cond, state_cond, time_emb], dim=-1)
        else:
            cond = torch.cat([vision_cond, time_emb], dim=-1)
        
        return cond
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        images: Dict[str, torch.Tensor],
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noisy_actions: (B, pred_horizon, action_dim)
            timestep: (B,)
            images: Dict {camera_name: (B, T, C, H, W)}
            states: (B, obs_horizon, state_dim) - 可选
        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        # 编码条件
        cond = self.encode_condition(images, timestep, states)
        
        # 调整维度
        x = noisy_actions.permute(0, 2, 1)  # (B, action_dim, pred_horizon)
        
        # 输入投影
        x = self.input_proj(x)
        
        # 下采样
        skip_features = []
        for block in self.down_blocks:
            x = block['resnet'](x, cond)
            skip_features.append(x)
            x = block['downsample'](x)
        
        # 中间层
        x = self.mid_block(x, cond)
        
        # 上采样
        for i, block in enumerate(self.up_blocks):
            x = block['upsample'](x)
            skip = skip_features[-(i + 1)]
            
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = block['resnet'](x, cond)
        
        # 输出
        x = self.output_proj(x)
        noise_pred = x.permute(0, 2, 1)
        
        return noise_pred


if __name__ == "__main__":
    """测试网络"""
    print("="*60)
    print("Testing Conditional U-Net with Vision + State")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试参数
    B = 4
    pred_horizon = 16
    action_dim = 14
    obs_horizon = 2
    state_dim = 50
    
    # 创建带状态输入的网络
    print("\n--- Test with states ---")
    model_with_state = ConditionalUNet1DWithVision(
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        camera_names=['frontview', 'agentview'],
        image_shape=(3, 240, 320),
        vision_embed_dim=256,
        state_dim=state_dim,  # 启用状态输入
        state_embed_dim=128,
        down_dims=[256, 512, 1024],
    ).to(device)
    
    total_params = sum(p.numel() for p in model_with_state.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 测试输入
    noisy_actions = torch.randn(B, pred_horizon, action_dim).to(device)
    timestep = torch.randint(0, 100, (B,)).to(device)
    images = {
        'frontview': torch.randn(B, obs_horizon, 3, 240, 320).to(device),
        'agentview': torch.randn(B, obs_horizon, 3, 240, 320).to(device),
    }
    states = torch.randn(B, obs_horizon, state_dim).to(device)
    
    # 前向传播
    with torch.no_grad():
        noise_pred = model_with_state(noisy_actions, timestep, images, states)
    
    print(f"Output shape: {noise_pred.shape}")
    assert noise_pred.shape == (B, pred_horizon, action_dim)
    print("✅ Forward pass with states successful!")
    
    # 测试纯视觉（无状态）
    print("\n--- Test without states ---")
    model_vision_only = ConditionalUNet1DWithVision(
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        camera_names=['frontview', 'agentview'],
        image_shape=(3, 240, 320),
        vision_embed_dim=256,
        state_dim=None,  # 禁用状态输入
        down_dims=[256, 512, 1024],
    ).to(device)
    
    with torch.no_grad():
        noise_pred2 = model_vision_only(noisy_actions, timestep, images)
    
    print(f"Output shape: {noise_pred2.shape}")
    assert noise_pred2.shape == (B, pred_horizon, action_dim)
    print("✅ Forward pass without states successful!")
