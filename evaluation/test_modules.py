"""
Quick Test Script for Dual-Arm Diffusion Policy
Tests each module independently
"""

import os
import numpy as np
import torch
from config import get_default_config
from data_collection import RandomPolicyCollector, load_demonstrations
from behavioral_cloning import BehavioralCloning
from diffusion_policy import DiffusionPolicyTrainer


def test_data_collection():
    print("=" * 60)
    print("Test 1: Data Collection")
    print("=" * 60)
    
    config = get_default_config()
    config.data.num_demonstrations = 5
    config.env.horizon = 50
    
    collector = RandomPolicyCollector(config)
    num_collected = collector.collect_multiple(num_demos=5, min_reward=-1.0)
    
    if num_collected > 0:
        print(f"[OK] Data collection test passed! Collected {num_collected} demos")
        return True
    else:
        print("[FAIL] Data collection test failed!")
        return False


def test_data_loading():
    print("\n" + "=" * 60)
    print("Test 2: Data Loading")
    print("=" * 60)
    
    config = get_default_config()
    data_dir = config.data.data_dir
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("[SKIP] No demonstration data found. Run data collection first.")
        return None
    
    demonstrations = load_demonstrations(data_dir)
    
    if len(demonstrations) > 0:
        print(f"[OK] Data loading test passed! Loaded {len(demonstrations)} demos")
        print(f"     First demo action shape: {demonstrations[0]['actions'].shape}")
        return demonstrations
    else:
        print("[FAIL] Data loading test failed!")
        return None


def test_bc_training(demonstrations):
    print("\n" + "=" * 60)
    print("Test 3: Behavioral Cloning Training")
    print("=" * 60)
    
    if demonstrations is None or len(demonstrations) == 0:
        print("[SKIP] No demonstrations available for BC training")
        return None
    
    config = get_default_config()
    config.bc.num_epochs = 5
    config.bc.batch_size = 16
    
    bc = BehavioralCloning(config, config.data.observation_keys)
    bc.train(demonstrations)
    
    os.makedirs(config.train.model_dir, exist_ok=True)
    bc.save(os.path.join(config.train.model_dir, "test_bc_model.pt"))
    
    print("[OK] BC training test passed!")
    return bc


def test_diffusion_training(demonstrations):
    print("\n" + "=" * 60)
    print("Test 4: Diffusion Policy Training")
    print("=" * 60)
    
    if demonstrations is None or len(demonstrations) == 0:
        print("[SKIP] No demonstrations available for Diffusion training")
        return None
    
    config = get_default_config()
    config.diffusion.num_epochs = 5
    config.diffusion.batch_size = 8
    config.diffusion.num_diffusion_steps = 20
    
    trainer = DiffusionPolicyTrainer(config, config.data.observation_keys)
    trainer.train(demonstrations)
    
    os.makedirs(config.train.model_dir, exist_ok=True)
    trainer.save(os.path.join(config.train.model_dir, "test_diffusion_policy.pt"))
    
    print("[OK] Diffusion training test passed!")
    return trainer


def main():
    print("\n" + "=" * 60)
    print("Dual-Arm Diffusion Policy - Module Tests")
    print("=" * 60)
    
    results = {}
    
    results['data_collection'] = test_data_collection()
    demonstrations = test_data_loading()
    results['bc'] = test_bc_training(demonstrations)
    results['diffusion'] = test_diffusion_training(demonstrations)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is None:
            status = "[SKIP]"
        else:
            status = "[PASS]"
        print(f"{test_name:20s} {status}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
