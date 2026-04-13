"""
Data Collection Module for Dual-Arm Manipulation
Loads expert demonstrations from HDF5 files
"""

import numpy as np
import h5py
import os


def load_demonstrations(data_dir, augment=False, normalize=True, chunk_size=None):
    """Load demonstrations from HDF5 files with optional preprocessing"""
    demonstrations = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return demonstrations
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            filepath = os.path.join(data_dir, filename)
            try:
                with h5py.File(filepath, 'r') as f:
                    # Check if it's the official RoboSuite format (from collect_demonstrations_auto.py)
                    if 'data' in f:
                        # Load official RoboSuite format
                        data_group = f['data']
                        for demo_key in data_group.keys():
                            if demo_key.startswith('demo_'):
                                demo_group = data_group[demo_key]
                                if 'actions' in demo_group:
                                    # Create demonstration from official format
                                    actions = np.array(demo_group['actions'])
                                    
                                    # Check action dimension and adjust if needed
                                    if actions.shape[-1] == 7:
                                        # For single arm actions, expand to dual arm (14 dimensions)
                                        dual_arm_actions = np.zeros((actions.shape[0], 14))
                                        dual_arm_actions[:, :7] = actions  # First arm uses the original actions
                                        actions = dual_arm_actions
                                        print(f"Expanded action dimension from 7 to 14 for {demo_key} in {filename}")
                                    elif actions.shape[-1] == 14:
                                        # Dual arm actions, no need to expand
                                        print(f"Using dual arm actions (14 dimensions) for {demo_key} in {filename}")
                                    else:
                                        print(f"Unexpected action dimension {actions.shape[-1]} for {demo_key} in {filename}")
                                    
                                    # Get states if available, otherwise use empty dict
                                    states = np.array(demo_group['states']) if 'states' in demo_group else np.array([])
                                    
                                    # Get initial state if available
                                    initial_state = None
                                    if 'initial_state' in demo_group:
                                        initial_state = np.array(demo_group['initial_state'])
                                    
                                    demo = {
                                        'observations': {'states': states},
                                        'actions': actions,
                                        'rewards': np.zeros(len(actions)),  # Add dummy rewards
                                        'dones': np.zeros(len(actions), dtype=bool),  # Add dummy dones
                                        'initial_state': initial_state,
                                    }
                                    # Mark the last step as done
                                    demo['dones'][-1] = True
                                    demonstrations.append(demo)
                                    print(f"Loaded official demonstration: {demo_key} from {filename}")
                    else:
                        # Load our custom format
                        actions = np.array(f['actions'])
                        # Check if action dimension is 7 (single arm) and need to expand to 14 (dual arm)
                        if actions.shape[-1] == 7:
                            # Duplicate actions for both arms or add zeros for the second arm
                            # Here we add zeros for the second arm
                            dual_arm_actions = np.zeros((actions.shape[0], 14))
                            dual_arm_actions[:, :7] = actions  # First arm uses the original actions
                            # Second arm uses zeros (no action)
                            actions = dual_arm_actions
                            print(f"Expanded action dimension from 7 to 14 for {filename}")
                        
                        demo = {
                            'observations': {},
                            'actions': actions,
                            'rewards': np.array(f['rewards']),
                            'dones': np.array(f['dones']),
                        }
                        
                        if 'observations' in f:
                            obs_group = f['observations']
                            for key in obs_group.keys():
                                demo['observations'][key] = np.array(obs_group[key])
                        else:
                            print(f"Warning: No observations group in {filename}")
                        
                        demonstrations.append(demo)
                        print(f"Loaded demonstration: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    print(f"\nLoaded {len(demonstrations)} demonstrations from {data_dir}")
    
    # Apply preprocessing
    if demonstrations:
        if augment:
            demonstrations = augment_data(demonstrations)
        if normalize:
            demonstrations = normalize_observations(demonstrations)
        if chunk_size:
            demonstrations = chunk_actions(demonstrations, chunk_size)
    
    return demonstrations


def augment_data(demonstrations, noise_std=0.05):
    """Augment data by adding Gaussian noise to actions"""
    augmented_demos = []
    
    for demo in demonstrations:
        # Create a copy of the demonstration
        augmented_demo = {
            'observations': demo['observations'].copy(),
            'actions': demo['actions'].copy(),
            'rewards': demo['rewards'].copy(),
            'dones': demo['dones'].copy(),
        }
        
        # Add Gaussian noise to actions
        noise = np.random.normal(0, noise_std, size=augmented_demo['actions'].shape)
        augmented_demo['actions'] = augmented_demo['actions'] + noise
        
        # Clip actions to valid range
        augmented_demo['actions'] = np.clip(augmented_demo['actions'], -1, 1)
        
        augmented_demos.append(augmented_demo)
    
    print(f"Augmented data: Added Gaussian noise (std={noise_std}) to {len(augmented_demos)} demonstrations")
    return augmented_demos


def normalize_observations(demonstrations):
    """Normalize observations to zero mean and unit variance"""
    # Collect all observations to compute stats
    all_obs = {}
    for demo in demonstrations:
        for key, value in demo['observations'].items():
            if key not in all_obs:
                all_obs[key] = []
            all_obs[key].append(value)
    
    # Compute mean and std for each observation key
    stats = {}
    for key, values in all_obs.items():
        concatenated = np.concatenate(values, axis=0)
        stats[key] = {
            'mean': np.mean(concatenated, axis=0),
            'std': np.std(concatenated, axis=0) + 1e-8  # Avoid division by zero
        }
    
    # Normalize each demonstration
    normalized_demos = []
    for demo in demonstrations:
        normalized_demo = {
            'observations': {},
            'actions': demo['actions'].copy(),
            'rewards': demo['rewards'].copy(),
            'dones': demo['dones'].copy(),
        }
        
        for key, value in demo['observations'].items():
            if key in stats:
                normalized_demo['observations'][key] = (value - stats[key]['mean']) / stats[key]['std']
            else:
                normalized_demo['observations'][key] = value
        
        normalized_demos.append(normalized_demo)
    
    print(f"Normalized observations: Applied zero-mean, unit-variance normalization to {len(normalized_demos)} demonstrations")
    return normalized_demos


def chunk_actions(demonstrations, chunk_size):
    """Chunk actions into sequences of specified length"""
    chunked_demos = []
    
    for demo in demonstrations:
        actions = demo['actions']
        obs = demo['observations']
        rewards = demo['rewards']
        dones = demo['dones']
        
        # Create chunks
        for i in range(0, len(actions) - chunk_size + 1):
            chunk = {
                'observations': {},
                'actions': actions[i:i+chunk_size],
                'rewards': rewards[i:i+chunk_size],
                'dones': dones[i:i+chunk_size],
            }
            
            # Extract observation chunks
            for key, value in obs.items():
                chunk['observations'][key] = value[i:i+chunk_size]
            
            chunked_demos.append(chunk)
    
    print(f"Chunked actions: Created {len(chunked_demos)} action chunks of size {chunk_size}")
    return chunked_demos


def check_demonstration_structure(demo):
    """Check the structure of a demonstration"""
    print("\nDemonstration structure:")
    print(f"Actions shape: {demo['actions'].shape}")
    print(f"Rewards shape: {demo['rewards'].shape}")
    print(f"Dones shape: {demo['dones'].shape}")
    print(f"Observations keys: {list(demo['observations'].keys())}")
    for key, value in demo['observations'].items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    from config import get_default_config
    
    config = get_default_config()
    
    # Load demonstrations
    demonstrations = load_demonstrations(config.data.data_dir)
    
    # Check demonstration structure if any loaded
    if demonstrations:
        check_demonstration_structure(demonstrations[0])
    else:
        print("No demonstrations loaded. Please download expert data and place it in the data directory.")

