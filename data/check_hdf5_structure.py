#!/usr/bin/env python3
"""
Script to check the structure of HDF5 demonstration files
"""

import os
import h5py
import numpy as np

def print_hdf5_structure(filepath, indent=0):
    """Recursively print HDF5 file structure"""
    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj):
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}Group: {name}")
                for key, val in obj.attrs.items():
                    print(f"{prefix}  Attribute: {key} = {val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{prefix}Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        
        print(f"\n=== File: {filepath} ===")
        f.visititems(print_structure)

def main():
    # Check expert_data directory
    expert_data_dir = "expert_data"
    if os.path.exists(expert_data_dir):
        print(f"\n=== Checking {expert_data_dir} directory ===")
        for file in os.listdir(expert_data_dir):
            if file.endswith('.hdf5'):
                filepath = os.path.join(expert_data_dir, file)
                print_hdf5_structure(filepath)
    else:
        print(f"Directory {expert_data_dir} does not exist")
    
    # Check data/demonstrations directory
    data_dir = "data/demonstrations"
    if os.path.exists(data_dir):
        print(f"\n=== Checking {data_dir} directory ===")
        for file in os.listdir(data_dir):
            if file.endswith('.h5') or file.endswith('.hdf5'):
                filepath = os.path.join(data_dir, file)
                print_hdf5_structure(filepath)
    else:
        print(f"Directory {data_dir} does not exist")

if __name__ == "__main__":
    main()
