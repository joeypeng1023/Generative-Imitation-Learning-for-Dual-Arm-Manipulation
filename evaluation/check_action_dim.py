import h5py
import numpy as np
import os
from config import get_default_config


def check_demonstration_action_dim(data_dir):
    """Check the action dimension in demonstration files"""
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5') or f.endswith('.hdf5')]
    
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files in {data_dir}")
    
    for filename in h5_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with h5py.File(filepath, 'r') as f:
                # Check if it's the official RoboSuite format
                if 'data' in f:
                    # Load official RoboSuite format
                    data_group = f['data']
                    for demo_key in data_group.keys():
                        if demo_key.startswith('demo_'):
                            demo_group = data_group[demo_key]
                            if 'actions' in demo_group:
                                actions = np.array(demo_group['actions'])
                                print(f"{filename} - {demo_key}: actions shape = {actions.shape}")
                                print(f"  Action dimension: {actions.shape[-1]}")
                else:
                    # Load our custom format
                    if 'actions' in f:
                        actions = np.array(f['actions'])
                        print(f"{filename}: actions shape = {actions.shape}")
                        print(f"  Action dimension: {actions.shape[-1]}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")


if __name__ == "__main__":
    config = get_default_config()
    data_dir = config.data.data_dir
    print(f"Checking action dimensions in: {data_dir}")
    check_demonstration_action_dim(data_dir)
