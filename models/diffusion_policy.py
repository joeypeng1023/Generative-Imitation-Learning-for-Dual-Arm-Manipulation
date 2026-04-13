"""
Diffusion Policy for Dual-Arm Manipulation
Based on "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.Mish()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels),
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUNet1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: List[int],
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv1d(input_dim, down_dims[0], kernel_size, padding=kernel_size // 2)
        
        # Downsampling path
        self.down_modules = nn.ModuleList([])
        for i in range(len(down_dims) - 1):
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1d(down_dims[i], down_dims[i+1], global_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1d(down_dims[i+1], down_dims[i+1], global_cond_dim, kernel_size, n_groups),
                    Downsample1d(down_dims[i+1]),
                ])
            )
        
        # Middle blocks
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], global_cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1d(down_dims[-1], down_dims[-1], global_cond_dim, kernel_size, n_groups),
        ])
        
        # Upsampling path
        self.up_modules = nn.ModuleList([])
        for i in reversed(range(len(down_dims) - 1)):
            in_channels = down_dims[i+1] + down_dims[i+1]  # 跳过连接的通道数与当前通道数相同
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1d(in_channels, down_dims[i], global_cond_dim, kernel_size, n_groups),
                    ConditionalResidualBlock1d(down_dims[i], down_dims[i], global_cond_dim, kernel_size, n_groups),
                    Upsample1d(down_dims[i]),
                ])
            )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0] + down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )
        
    def forward(self, x, cond):
        x = self.conv_in(x)
        
        # Store downsampled features for skip connections
        down_features = [x]  # 保存初始特征
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = downsample(x)
            down_features.append(x)  # 保存下采样后的特征
        
        # Middle processing
        for mid_module in self.mid_modules:
            x = mid_module(x, cond)
        
        # Upsampling with skip connections
        for i, (resnet1, resnet2, upsample) in enumerate(self.up_modules):
            skip = down_features.pop()  # 获取对应的下采样特征
            # 确保特征图大小匹配
            if x.shape[-1] != skip.shape[-1]:
                # 调整skip的大小以匹配x
                import torch.nn.functional as F
                skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # 连接跳过连接
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)
        
        # Final skip connection and output
        skip = down_features.pop()
        if x.shape[-1] != skip.shape[-1]:
            import torch.nn.functional as F
            skip = F.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.final_conv(x)
        return x


class DiffusionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.num_diffusion_steps = config.num_diffusion_steps
        self.predict_noise = config.predict_noise
        
        self.noise_scheduler = None
        self.num_train_timesteps = config.num_diffusion_steps
        self.betas = torch.linspace(1e-4, 0.02, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(256),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        
        self.down_dims = config.down_dims
        self.kernel_size = config.kernel_size
        self.n_groups = config.n_groups
        
        global_cond_dim = 256 + 256
        
        self.model = ConditionalUNet1d(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )
        
    def forward(self, noisy_actions, timestep, obs):
        timestep_emb = self.diffusion_step_encoder(timestep)
        obs_emb = self.obs_encoder(obs)
        global_cond = torch.cat([timestep_emb, obs_emb], dim=-1)
        
        noisy_actions = rearrange(noisy_actions, 'b h a -> b a h')
        
        pred = self.model(noisy_actions, global_cond)
        
        pred = rearrange(pred, 'b a h -> b h a')
        
        return pred
    
    def add_noise(self, actions, timestep, noise=None):
        if noise is None:
            noise = torch.randn_like(actions)
        
        alpha_prod = self.alphas_cumprod[timestep].to(actions.device)
        alpha_prod = alpha_prod.view(-1, 1, 1)
        
        noisy_actions = torch.sqrt(alpha_prod) * actions + torch.sqrt(1 - alpha_prod) * noise
        
        return noisy_actions, noise
    
    def predict_start_from_noise(self, noisy_actions, timestep, noise_pred):
        alpha_prod = self.alphas_cumprod[timestep].to(noisy_actions.device)
        alpha_prod = alpha_prod.view(-1, 1, 1)
        
        pred_start = (noisy_actions - torch.sqrt(1 - alpha_prod) * noise_pred) / torch.sqrt(alpha_prod)
        
        return pred_start
    
    def step(self, noisy_actions, timestep, noise_pred):
        alpha_prod = self.alphas_cumprod[timestep].to(noisy_actions.device)
        alpha_prod_t = self.alphas_cumprod[timestep - 1].to(noisy_actions.device) if timestep > 0 else torch.tensor(1.0).to(noisy_actions.device)
        
        alpha_prod = alpha_prod.view(-1, 1, 1)
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1)
        
        pred_start = self.predict_start_from_noise(noisy_actions, timestep, noise_pred)
        
        noise = torch.randn_like(noisy_actions) if timestep > 0 else 0
        
        prev_actions = torch.sqrt(alpha_prod_t) * pred_start + torch.sqrt(1 - alpha_prod_t) * noise
        
        return prev_actions
    
    def generate(self, obs, num_samples=1):
        device = next(self.parameters()).device
        batch_size = obs.shape[0]
        
        noisy_actions = torch.randn(batch_size, self.horizon, self.action_dim, device=device)
        
        for t in reversed(range(self.num_train_timesteps)):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            noise_pred = self.forward(noisy_actions, timestep, obs)
            
            noisy_actions = self.step(noisy_actions, timestep, noise_pred)
        
        return noisy_actions


class DiffusionPolicyTrainer:
    def __init__(self, config, observation_keys):
        self.config = config.diffusion
        self.train_config = config.train
        self.observation_keys = observation_keys
        
        self.policy = None
        self.optimizer = None
        self.ema_policy = None
        
    def _compute_obs_dim(self, demonstrations):
        # Check if 'states' is available (official RoboSuite format)
        if 'states' in demonstrations[0]['observations']:
            return demonstrations[0]['observations']['states'].shape[-1]
        
        # Otherwise, try to use the configured observation keys
        obs_dim = 0
        for key in self.observation_keys:
            if key in demonstrations[0]['observations']:
                obs_dim += demonstrations[0]['observations'][key].shape[-1]
        return obs_dim
    
    def _create_sequences(self, demonstrations):
        sequences = []
        horizon = self.config.horizon
        
        for demo in demonstrations:
            # Check if 'states' is available (official RoboSuite format)
            if 'states' in demo['observations']:
                obs = demo['observations']['states']
                actions = demo['actions']
                
                for i in range(len(actions) - horizon + 1):
                    seq = {
                        'observation': obs[i],
                        'action_sequence': actions[i:i+horizon],
                    }
                    sequences.append(seq)
            else:
                # Otherwise, try to use the configured observation keys
                obs_list = []
                for key in self.observation_keys:
                    if key in demo['observations']:
                        obs_list.append(demo['observations'][key])
                
                if obs_list:
                    obs = np.concatenate(obs_list, axis=-1)
                    actions = demo['actions']
                    
                    for i in range(len(actions) - horizon + 1):
                        seq = {
                            'observation': obs[i],
                            'action_sequence': actions[i:i+horizon],
                        }
                        sequences.append(seq)
        
        return sequences
    
    def train(self, demonstrations):
        obs_dim = self._compute_obs_dim(demonstrations)
        action_dim = demonstrations[0]['actions'].shape[-1]
        
        self.config.observation_dim = obs_dim
        self.config.action_dim = action_dim
        
        print(f"Training Diffusion Policy: obs_dim={obs_dim}, action_dim={action_dim}")
        
        self.policy = DiffusionPolicy(self.config).to(self.train_config.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )
        
        sequences = self._create_sequences(demonstrations)
        print(f"Created {len(sequences)} training sequences")
        
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        
        for epoch in range(num_epochs):
            np.random.shuffle(sequences)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                
                obs = torch.tensor(
                    np.array([s['observation'] for s in batch]),
                    dtype=torch.float32, device=self.train_config.device
                )
                actions = torch.tensor(
                    np.array([s['action_sequence'] for s in batch]),
                    dtype=torch.float32, device=self.train_config.device
                )
                
                batch_size_actual = obs.shape[0]
                timesteps = torch.randint(
                    0, self.config.num_diffusion_steps,
                    (batch_size_actual,), device=self.train_config.device
                )
                
                noisy_actions, noise = self.policy.add_noise(actions, timesteps)
                
                noise_pred = self.policy(noisy_actions, timesteps, obs)
                
                loss = F.mse_loss(noise_pred, noise if self.config.predict_noise else actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        return self.policy
    
    def predict(self, observation):
        self.policy.eval()
        with torch.no_grad():
            # Check if 'states' is available (official RoboSuite format)
            if 'states' in observation:
                obs = observation['states']
                obs_tensor = torch.from_numpy(obs).float().to(self.train_config.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                action_seq = self.policy.generate(obs_tensor)
                return action_seq[0, 0].cpu().numpy()
            else:
                # Otherwise, try to use the configured observation keys
                obs_list = []
                for key in self.observation_keys:
                    if key in observation:
                        obs_list.append(observation[key])
                
                if obs_list:
                    obs = np.concatenate(obs_list, axis=-1)
                    obs_tensor = torch.from_numpy(obs).float().to(self.train_config.device)
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    action_seq = self.policy.generate(obs_tensor)
                    return action_seq[0, 0].cpu().numpy()
        
        return np.zeros(self.config.action_dim)
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
        }, path)
        print(f"Saved Diffusion Policy to {path}")
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.train_config.device)
        self.config = checkpoint['config']
        self.policy = DiffusionPolicy(self.config).to(self.train_config.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded Diffusion Policy from {path}")


if __name__ == "__main__":
    from config import get_default_config
    from data_collection import load_demonstrations
    
    config = get_default_config()
    
    data_dir = config.data.data_dir
    demonstrations = load_demonstrations(data_dir)
    
    if len(demonstrations) > 0:
        trainer = DiffusionPolicyTrainer(config, config.data.observation_keys)
        trainer.train(demonstrations)
        trainer.save(os.path.join(config.train.model_dir, "diffusion_policy.pt"))
