"""
Check the contents of the demonstrations HDF5 file
"""

import h5py
import numpy as np

def check_demonstrations(hdf5_path):
    """Check the contents of the demonstrations HDF5 file"""
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check if 'data' group exists
        if 'data' not in f:
            print("No 'data' group found in HDF5 file")
            return
        
        data_group = f['data']
        
        # Print metadata
        print("=== Metadata ===")
        print(f"Date: {data_group.attrs['date']}")
        print(f"Time: {data_group.attrs['time']}")
        print(f"Environment: {data_group.attrs['env']}")
        print(f"Robots: {data_group.attrs['robots']}")
        print()
        
        # Count demonstrations
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        print(f"=== Total Demonstrations: {len(demo_keys)} ===")
        print()
        
        # Check each demonstration
        for i, demo_key in enumerate(demo_keys[:5]):  # Only check first 5 for brevity
            demo_group = data_group[demo_key]
            print(f"=== {demo_key} ===")
            
            # Check states
            if 'states' in demo_group:
                states = demo_group['states']
                print(f"  States shape: {states.shape}")
                print(f"  States dtype: {states.dtype}")
            else:
                print("  No states found")
            
            # Check actions
            if 'actions' in demo_group:
                actions = demo_group['actions']
                print(f"  Actions shape: {actions.shape}")
                print(f"  Actions dtype: {actions.dtype}")
            else:
                print("  No actions found")
            
            print()
        
        if len(demo_keys) > 5:
            print(f"... and {len(demo_keys) - 5} more demonstrations")

if __name__ == "__main__":
    hdf5_path = r"C:\Users\86152\robosuite_demos\TwoArmLift\demonstrations.hdf5"
    check_demonstrations(hdf5_path)
