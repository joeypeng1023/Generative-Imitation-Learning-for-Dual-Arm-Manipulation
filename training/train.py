"""
Training and Evaluation Pipeline for Dual-Arm Manipulation
"""

import os
import numpy as np
import torch
import robosuite as suite
from tqdm import tqdm

from config import get_default_config
from data_collection import load_demonstrations
from behavioral_cloning import BehavioralCloning
from diffusion_policy import DiffusionPolicyTrainer


class Trainer:
    def __init__(self, config):
        self.config = config
        
    def collect_data(self, num_demos=None):
        print("=" * 60)
        print("Step 1: Preparing Expert Demonstrations")
        print("=" * 60)
        
        # Check if data already exists
        import os
        h5_files = []
        if os.path.exists(self.config.data.data_dir):
            h5_files = [f for f in os.listdir(self.config.data.data_dir) if f.endswith('.h5')]
            if h5_files:
                print(f"Found {len(h5_files)} HDF5 files in the data directory:")
                for f in h5_files:
                    print(f"  - {f}")
            else:
                print("No HDF5 files found in the data directory.")
                print("Please download expert demonstrations manually and place them in:")
                print(f"  {self.config.data.data_dir}")
        else:
            print(f"Data directory does not exist. Creating it: {self.config.data.data_dir}")
            os.makedirs(self.config.data.data_dir, exist_ok=True)
            print("Please download expert demonstrations manually and place them in the created directory.")
        
        print("\nExpert demonstrations are ready for training!")
        
    def train_bc(self):
        print("\n" + "=" * 60)
        print("Step 2: Training Behavioral Cloning")
        print("=" * 60)
        
        # Load demonstrations with preprocessing
        demonstrations = load_demonstrations(
            self.config.data.data_dir,
            augment=True,           # 启用数据增强
            normalize=True,         # 启用观测值归一化
            chunk_size=None         # BC不需要动作分块
        )
        
        if len(demonstrations) == 0:
            print("No demonstrations found! Please collect data first.")
            return None
            
        bc = BehavioralCloning(self.config, self.config.data.observation_keys)
        bc.train(demonstrations)
        
        os.makedirs(self.config.train.model_dir, exist_ok=True)
        bc.save(os.path.join(self.config.train.model_dir, "bc_model.pt"))
        
        return bc
    
    def train_diffusion(self):
        print("\n" + "=" * 60)
        print("Step 3: Training Diffusion Policy")
        print("=" * 60)
        
        # Load demonstrations with preprocessing (including action chunking for diffusion)
        demonstrations = load_demonstrations(
            self.config.data.data_dir,
            augment=True,           # 启用数据增强
            normalize=True,         # 启用观测值归一化
            chunk_size=self.config.diffusion.horizon  # 使用配置的horizon作为分块大小
        )
        
        if len(demonstrations) == 0:
            print("No demonstrations found! Please collect data first.")
            return None
            
        trainer = DiffusionPolicyTrainer(self.config, self.config.data.observation_keys)
        trainer.train(demonstrations)
        
        os.makedirs(self.config.train.model_dir, exist_ok=True)
        trainer.save(os.path.join(self.config.train.model_dir, "diffusion_policy.pt"))
        
        return trainer


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.env = suite.make(
            env_name=config.env.env_name,
            robots=config.env.robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=config.env.horizon,
            control_freq=config.env.control_freq,
        )
        
    def evaluate_policy(self, policy, policy_type='bc', num_episodes=10):
        print(f"\nEvaluating {policy_type} policy for {num_episodes} episodes...")
        
        rewards = []
        successes = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.config.env.horizon:
                if policy_type == 'bc':
                    action = policy.predict(obs)
                elif policy_type == 'diffusion':
                    action = policy.predict(obs)
                else:
                    action = np.zeros(self.env.action_dim)
                
                low, high = self.env.action_spec
                action = np.clip(action, low, high)
                
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                step += 1
                
                if episode == 0:
                    self.env.render()
            
            rewards.append(episode_reward)
            successes.append(episode_reward > 0)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        success_rate = np.mean(successes)
        
        print(f"\nResults:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate * 100:.1f}%")
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'rewards': rewards,
        }
    
    def close(self):
        self.env.close()


def run_full_pipeline():
    print("Starting full pipeline...")
    config = get_default_config()
    
    print(f"Config loaded: env={config.env.env_name}, robots={config.env.robots}")
    print(f"Data directory: {config.data.data_dir}")
    
    config.data.num_demonstrations = 20
    config.bc.num_epochs = 50
    config.diffusion.num_epochs = 100
    
    print("Creating trainer...")
    trainer = Trainer(config)
    
    print("Calling collect_data...")
    trainer.collect_data(num_demos=20)
    print("collect_data completed")
    
    print("Calling train_bc...")
    bc = trainer.train_bc()
    print("train_bc completed")
    
    print("Calling train_diffusion...")
    diffusion = trainer.train_diffusion()
    print("train_diffusion completed")
    
    print("\n" + "=" * 60)
    print("Step 4: Evaluating Policies")
    print("=" * 60)
    
    evaluator = Evaluator(config)
    
    if bc is not None:
        bc_results = evaluator.evaluate_policy(bc, policy_type='bc', num_episodes=5)
    
    if diffusion is not None:
        diffusion_results = evaluator.evaluate_policy(diffusion, policy_type='diffusion', num_episodes=5)
    
    evaluator.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
