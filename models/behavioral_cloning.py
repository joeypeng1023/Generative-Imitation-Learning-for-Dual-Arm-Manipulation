"""
Behavioral Cloning Baseline Model
Classic imitation learning approach for dual-arm manipulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class BCDataset(Dataset):
    def __init__(self, demonstrations, observation_keys):
        self.observations = []
        self.actions = []
        
        for demo in demonstrations:
            # Check if 'states' is available (official RoboSuite format)
            if 'states' in demo['observations']:
                obs = demo['observations']['states']
                self.observations.extend(obs)
                self.actions.extend(demo['actions'])
            else:
                # Otherwise, try to use the configured observation keys
                obs_list = []
                for key in observation_keys:
                    if key in demo['observations']:
                        obs_list.append(demo['observations'][key])
                
                if obs_list:
                    obs = np.concatenate(obs_list, axis=-1)
                    self.observations.extend(obs)
                    self.actions.extend(demo['actions'])
        
        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        
        print(f"BC Dataset: {len(self.observations)} samples")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': torch.from_numpy(self.observations[idx]),
            'action': torch.from_numpy(self.actions[idx]),
        }


class BCNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.network(obs)


class BehavioralCloning:
    def __init__(self, config, observation_keys):
        self.config = config.bc
        self.train_config = config.train
        self.observation_keys = observation_keys
        
        self.obs_dim = None
        self.action_dim = None
        self.model = None
        self.optimizer = None
        
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
    
    def train(self, demonstrations):
        self.obs_dim = self._compute_obs_dim(demonstrations)
        self.action_dim = demonstrations[0]['actions'].shape[-1]
        
        print(f"Training BC: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        
        self.model = BCNetwork(
            self.obs_dim, 
            self.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(self.train_config.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        dataset = BCDataset(demonstrations, self.observation_keys)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                obs = batch['observation'].to(self.train_config.device)
                action = batch['action'].to(self.train_config.device)
                
                self.optimizer.zero_grad()
                pred_action = self.model(obs)
                loss = F.mse_loss(pred_action, action)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")
        
        return self.model
    
    def predict(self, observation):
        self.model.eval()
        with torch.no_grad():
            # Check if 'states' is available (official RoboSuite format)
            if 'states' in observation:
                obs = observation['states']
                obs_tensor = torch.from_numpy(obs).float().to(self.train_config.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                action = self.model(obs_tensor)
                return action.cpu().numpy().squeeze()
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
                    action = self.model(obs_tensor)
                    return action.cpu().numpy().squeeze()
        
        return np.zeros(self.action_dim)
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'config': self.config,
        }, path)
        print(f"Saved BC model to {path}")
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.train_config.device)
        self.obs_dim = checkpoint['obs_dim']
        self.action_dim = checkpoint['action_dim']
        self.model = BCNetwork(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.num_layers
        ).to(self.train_config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded BC model from {path}")


if __name__ == "__main__":
    from config import get_default_config
    from data_collection import load_demonstrations
    
    config = get_default_config()
    
    data_dir = config.data.data_dir
    demonstrations = load_demonstrations(data_dir)
    
    if len(demonstrations) > 0:
        bc = BehavioralCloning(config, config.data.observation_keys)
        bc.train(demonstrations)
        bc.save(os.path.join(config.train.model_dir, "bc_model.pt"))
