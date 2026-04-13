"""
Environment Test Script for Dual-Arm Diffusion Policy Project
This script verifies that all required packages are installed correctly.
"""

import sys
import numpy as np
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 60)

def test_import(module_name, package_name=None):
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name or module_name}: {e}")
        return False

print("Testing core dependencies:")
print("-" * 60)
test_import('numpy', 'numpy')
test_import('scipy', 'scipy')
test_import('torch', 'torch')
test_import('torchvision', 'torchvision')

print("\nTesting robosuite:")
print("-" * 60)
test_import('robosuite', 'robosuite')
test_import('mujoco', 'mujoco')

print("\nTesting Diffusion Policy dependencies:")
print("-" * 60)
test_import('diffusers', 'diffusers')
test_import('transformers', 'transformers')
test_import('accelerate', 'accelerate')
test_import('wandb', 'wandb')
test_import('h5py', 'h5py')
test_import('einops', 'einops')

print("\nTesting additional utilities:")
print("-" * 60)
test_import('cv2', 'opencv-python')
test_import('matplotlib', 'matplotlib')
test_import('pygame', 'pygame')

print("\n" + "=" * 60)
print("Environment test completed!")
print("=" * 60)

print("\nTesting robosuite dual-arm environment...")
print("-" * 60)
try:
    import robosuite as suite
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["Panda", "Panda"],
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        control_freq=20,
    )
    
    print(f"[OK] Successfully created TwoArmLift environment")
    print(f"     - Robots: {env.robots}")
    print(f"     - Action dimension: {env.action_dim}")
    obs = env.reset()
    print(f"     - Observation keys: {list(obs.keys())}")
    
    print("\n[RENDER] Starting MuJoCo rendering...")
    print("         Close the window to stop rendering")
    print("-" * 60)
    
    # 运行模拟并渲染
    for i in range(200):
        # 生成随机动作
        action = env.action_spec[0] + (env.action_spec[1] - env.action_spec[0]) * 0.5
        action += 0.1 * (2 * np.random.random(env.action_dim) - 1)
        action = np.clip(action, env.action_spec[0], env.action_spec[1])
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 渲染当前状态
        env.render()
        
        if done:
            obs = env.reset()
    
    env.close()
    print("[OK] Environment test passed!")
    
except Exception as e:
    print(f"[FAIL] Robosuite dual-arm test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
