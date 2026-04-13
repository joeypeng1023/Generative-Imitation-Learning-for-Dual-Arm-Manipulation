import robosuite as suite
import numpy as np
from config import get_default_config
from data_collection import load_demonstrations


def test_action_dimension():
    """Test if the action dimension is correctly expanded to 14"""
    config = get_default_config()
    
    # Load demonstrations with our modified function
    demonstrations = load_demonstrations(config.data.data_dir)
    
    if not demonstrations:
        print("No demonstrations found!")
        return False
    
    # Check action dimensions
    for i, demo in enumerate(demonstrations):
        action_dim = demo['actions'].shape[-1]
        print(f"Demonstration {i+1}: action dimension = {action_dim}")
        if action_dim != 14:
            print(f"ERROR: Expected action dimension 14, got {action_dim}")
            return False
    
    print("All demonstrations have correct action dimension (14)!")
    
    # Test environment creation and action step
    print("\nTesting environment creation...")
    try:
        env = suite.make(
            env_name=config.env.env_name,
            robots=config.env.robots,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=config.env.horizon,
            control_freq=config.env.control_freq,
        )
        print(f"Environment created successfully: {config.env.env_name}")
        print(f"Environment action dimension: {env.action_dim}")
        
        # Test with a sample action
        obs = env.reset()
        sample_action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(sample_action)
        print("Successfully stepped with action of dimension:", sample_action.shape[-1])
        
        env.close()
        return True
    except Exception as e:
        print(f"Error creating environment: {e}")
        return False


if __name__ == "__main__":
    print("Testing action dimension fix...")
    success = test_action_dimension()
    if success:
        print("\n✅ Action dimension fix was successful!")
    else:
        print("\n❌ Action dimension fix failed!")
