"""
Configuration for Dual-Arm Diffusion Policy Imitation Learning
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EnvironmentConfig:
    env_name: str = "TwoArmLift"
    robots: List[str] = field(default_factory=lambda: ["Panda", "Panda"])
    has_renderer: bool = True
    has_offscreen_renderer: bool = False
    use_camera_obs: bool = False
    horizon: int = 500
    control_freq: int = 20
    controller_config: str = "default"
    arm: str = "bimanual"
    camera: List[str] = field(default_factory=lambda: ["frontview"])
    renderer: str = "mjviewer"
    max_fr: int = 20
    goal_update_mode: str = "target"
    device: str = "keyboard"
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    reverse_xy: bool = False
    controller: Optional[str] = None


@dataclass
class DataConfig:
    data_dir: str = "data/demonstrations"
    num_demonstrations: int = 50
    horizon: int = 16
    observation_keys: List[str] = field(default_factory=lambda: [
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
        "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
        "object-state"
    ])


@dataclass
class BCConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    weight_decay: float = 1e-5


@dataclass
class DiffusionPolicyConfig:
    observation_dim: int = 28
    action_dim: int = 14
    horizon: int = 16
    num_diffusion_steps: int = 100
    down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    n_groups: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 300
    ema_power: float = 0.75
    predict_noise: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    log_dir: str = "data/logs"
    model_dir: str = "data/models"
    save_every: int = 10
    eval_every: int = 5
    num_eval_episodes: int = 10


@dataclass
class Config:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    diffusion: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    @classmethod
    def from_dict(cls, d: dict):
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
        return config


def get_default_config():
    return Config()
