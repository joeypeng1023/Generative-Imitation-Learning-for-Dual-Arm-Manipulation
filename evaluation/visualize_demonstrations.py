"""
Visualize demonstrations from HDF5 files
"""

import os
import numpy as np
import h5py
import robosuite as suite
from config import get_default_config


def load_demonstration(filepath, target_action_dim=None, demo_index=0):
    """Load a single demonstration from HDF5 file and adjust action dimension if needed"""
    with h5py.File(filepath, 'r') as f:
        # Check if it's the official RoboSuite format (from collect_demonstrations_auto.py)
        if 'data' in f:
            # Load official RoboSuite format
            data_group = f['data']
            # Find all demo groups
            demo_keys = sorted([key for key in data_group.keys() if key.startswith('demo_')])
            if demo_keys:
                # Select the specified demo index
                if demo_index < len(demo_keys):
                    demo_group = data_group[demo_keys[demo_index]]
                    if 'actions' in demo_group:
                        actions = np.array(demo_group['actions'])
                        
                        # Adjust action dimension if target is specified
                        if target_action_dim is not None and actions.shape[-1] != target_action_dim:
                            print(f"Adjusting action dimension from {actions.shape[-1]} to {target_action_dim} for visualization")
                            # Create new action array with target dimension
                            new_actions = np.zeros((actions.shape[0], target_action_dim))
                            # Copy as much of the original actions as possible
                            min_dim = min(actions.shape[-1], target_action_dim)
                            new_actions[:, :min_dim] = actions[:, :min_dim]
                            actions = new_actions
                        
                        # Get states if available
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
                        return demo, len(demo_keys)
                else:
                    print(f"Demo index {demo_index} out of range. Total demos: {len(demo_keys)}")
                    return None, len(demo_keys)
            else:
                print(f"No demo groups found in file. Available keys: {list(data_group.keys())}")
                return None, 0
        
        # Load our custom format
        actions = np.array(f['actions'])
        
        # Adjust action dimension if target is specified
        if target_action_dim is not None and actions.shape[-1] != target_action_dim:
            print(f"Adjusting action dimension from {actions.shape[-1]} to {target_action_dim} for visualization")
            # Create new action array with target dimension
            new_actions = np.zeros((actions.shape[0], target_action_dim))
            # Copy as much of the original actions as possible
            min_dim = min(actions.shape[-1], target_action_dim)
            new_actions[:, :min_dim] = actions[:, :min_dim]
            actions = new_actions
        
        demo = {
            'observations': {},
            'actions': actions,
            'rewards': np.array(f['rewards']),
        }
        
        # Handle optional 'dones' field
        if 'dones' in f:
            demo['dones'] = np.array(f['dones'])
        else:
            # Create default dones array if not present
            demo['dones'] = np.zeros(len(demo['actions']), dtype=bool)
        
        # Handle observations
        if 'observations' in f:
            obs_obj = f['observations']
            # Check if observations is a group or a dataset
            if isinstance(obs_obj, h5py.Group):
                # If it's a group, iterate through its keys
                for key in obs_obj.keys():
                    demo['observations'][key] = np.array(obs_obj[key])
            else:
                # If it's a dataset, treat it as a single observation array
                demo['observations']['observation'] = np.array(obs_obj)
        
    return demo, 1


def visualize_demonstration(demo_file, config, demo_index=0):
    """Visualize a demonstration"""
    print(f"Visualizing demonstration: {demo_file}")
    print(f"Demo index: {demo_index}")
    
    # Import robosuite here to avoid import issues
    import robosuite as suite
    
    # Create environment with fixed camera for better visualization
    # Set horizon to a large value to avoid episode termination during visualization
    env_config = config.env
    env_config.horizon = 10000  # Set to a value larger than any demonstration length
    
    env = suite.make(
        env_name=env_config.env_name,
        robots=env_config.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=env_config.horizon,
        control_freq=env_config.control_freq,
        render_camera=env_config.camera[0],
    )
    
    # Load demonstration and adjust action dimension to match environment
    demo, total_demos = load_demonstration(demo_file, env.action_dim, demo_index)
    
    if demo is None:
        print("Failed to load demonstration")
        env.close()
        return
    
    # Reset environment
    if 'initial_state' in demo and demo['initial_state'] is not None:
        print(f"Using stored initial state for demonstration...")
        # Set the environment to the initial state
        env.reset()
        # For robosuite, we need to set the state directly using flattened state
        if hasattr(env, 'sim') and hasattr(env.sim, 'set_state_from_flattened'):
            env.sim.set_state_from_flattened(demo['initial_state'])
            env.sim.forward()  # Need to forward the simulation after setting state
        elif hasattr(env, 'set_state_from_flattened'):
            env.set_state_from_flattened(demo['initial_state'])
        obs = env._get_observations()
    else:
        print(f"No initial state found, using random reset...")
        obs = env.reset()
    
    print(f"Environment reset. Starting visualization...")
    print(f"Number of steps: {len(demo['actions'])} ")
    print("\n=== Controls ===")
    print("SPACE: Pause/Resume")
    print("N: Next step (when paused)")
    print("Q: Quit visualization")
    print("================\n")
    
    # Playback loop with pause/resume functionality
    paused = False
    step_idx = 0
    
    while step_idx < len(demo['actions']):
        if not paused:
            # Get action for current step
            action = demo['actions'][step_idx]
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            
            # Render
            env.render()
            
            step_idx += 1
            
            # Check if we should stop
            if done or step_idx >= len(demo['actions']):
                print(f"\nDemonstration completed! Total steps: {step_idx}")
                break
        else:
            # When paused, just render the current state
            env.render()
        
        # Handle keyboard input for pause/resume/quit
        # Note: This is a simplified version - actual implementation may vary based on renderer
        # For now, we'll use a simple time-based approach
        import time
        time.sleep(0.05)  # 20 FPS
    
    print("Closing environment...")
    env.close()
    
    return total_demos


def main():
    import sys
    
    # Get config
    config = get_default_config()
    
    # Check if a specific file was provided as command line argument
    if len(sys.argv) > 1:
        # Use the provided file path
        demo_path = sys.argv[1]
        if not os.path.exists(demo_path):
            print(f"Error: File not found: {demo_path}")
            return
        
        print(f"Visualizing specific file: {demo_path}")
        
        # Get total demos in this file
        _, file_demos = load_demonstration(demo_path, demo_index=0)
        
        if file_demos == 0:
            print(f"No valid demonstrations in {demo_path}")
            return
        
        # Visualize all demos in the file
        print(f"\nThis file contains {file_demos} demonstration(s).")
        for j in range(file_demos):
            print(f"\n--- Demonstration {j+1}/{file_demos} ---")
            visualize_demonstration(demo_path, config, demo_index=j)
            
            # Add a short delay between demonstrations
            if j != file_demos - 1:
                import time
                print("\nWaiting 2 seconds before next demonstration...")
                time.sleep(2)
        
        return
    
    # Otherwise, use the default directory
    # Get demonstration files
    demo_dir = config.data.data_dir
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith('.h5') or f.endswith('.hdf5')]
    
    if not demo_files:
        print(f"No HDF5 files found in {demo_dir}")
        print("Please place your expert demonstration files in this directory.")
        return
    
    print(f"Found {len(demo_files)} demonstration files:")
    for i, f in enumerate(demo_files):
        print(f"  {i+1}. {f}")
    
    # Show dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total demonstration files: {len(demo_files)}")
    
    # Calculate total steps across all demonstrations
    total_steps = 0
    total_demos = 0
    file_info = []  # Store info about each file
    
    for demo_file in demo_files:
        demo_path = os.path.join(demo_dir, demo_file)
        try:
            # Load the first demo to get total demos in this file
            demo, file_demos = load_demonstration(demo_path, demo_index=0)
            if demo:
                total_demos += file_demos
                file_steps = 0
                # For each demo in the file, get the number of steps
                for i in range(file_demos):
                    demo, _ = load_demonstration(demo_path, demo_index=i)
                    if demo:
                        file_steps += len(demo['actions'])
                total_steps += file_steps
                file_info.append({
                    'file': demo_file,
                    'demos': file_demos,
                    'steps': file_steps
                })
            else:
                file_info.append({
                    'file': demo_file,
                    'demos': 0,
                    'steps': 0,
                    'error': 'No valid demonstrations found'
                })
        except Exception as e:
            print(f"Error loading {demo_file}: {e}")
            file_info.append({
                'file': demo_file,
                'demos': 0,
                'steps': 0,
                'error': str(e)
            })
    
    print(f"Total demonstrations: {total_demos}")
    print(f"Total steps: {total_steps}")
    if total_demos > 0:
        print(f"Average steps per demonstration: {total_steps / total_demos:.1f}")
    
    # Interactive menu for selecting which file to visualize
    print("\n=== Visualization Options ===")
    print("0. Visualize ALL demonstrations")
    for i, info in enumerate(file_info):
        if 'error' in info:
            print(f"{i+1}. {info['file']} - Error: {info['error']}")
        else:
            print(f"{i+1}. {info['file']} - {info['demos']} demos, {info['steps']} steps")
    print("Q. Quit")
    
    choice = input("\nEnter your choice (number or Q to quit): ").strip().lower()
    
    if choice == 'q':
        print("Exiting...")
        return
    
    try:
        choice_num = int(choice)
        if choice_num == 0:
            # Visualize all demonstrations
            print("\n=== Visualizing all demonstrations ===")
            for i, demo_file in enumerate(demo_files):
                demo_path = os.path.join(demo_dir, demo_file)
                print(f"\n=== Visualizing file {i+1}/{len(demo_files)}: {demo_file} ===")
                
                # Get total demos in this file
                _, file_demos = load_demonstration(demo_path, demo_index=0)
                
                if file_demos == 0:
                    print(f"Skipping {demo_file} - no valid demonstrations")
                    continue
                
                # Visualize each demo in the file
                for j in range(file_demos):
                    print(f"\n--- Demonstration {j+1}/{file_demos} in file {demo_file} ---")
                    visualize_demonstration(demo_path, config, demo_index=j)
                    
                    # Add a short delay between demonstrations
                    if j != file_demos - 1:
                        import time
                        print("\nWaiting 2 seconds before next demonstration...")
                        time.sleep(2)
        elif 1 <= choice_num <= len(demo_files):
            # Visualize specific file
            selected_file = demo_files[choice_num - 1]
            demo_path = os.path.join(demo_dir, selected_file)
            print(f"\n=== Visualizing file: {selected_file} ===")
            
            # Get total demos in this file
            _, file_demos = load_demonstration(demo_path, demo_index=0)
            
            if file_demos == 0:
                print(f"No valid demonstrations in {selected_file}")
                return
            
            # Ask if user wants to see all demos or a specific one
            if file_demos > 1:
                print(f"\nThis file contains {file_demos} demonstrations.")
                demo_choice = input(f"Enter demonstration number (1-{file_demos}) or 0 for all: ").strip()
                try:
                    demo_choice_num = int(demo_choice)
                    if demo_choice_num == 0:
                        # Visualize all demos in this file
                        for j in range(file_demos):
                            print(f"\n--- Demonstration {j+1}/{file_demos} ---")
                            visualize_demonstration(demo_path, config, demo_index=j)
                            if j != file_demos - 1:
                                import time
                                print("\nWaiting 2 seconds before next demonstration...")
                                time.sleep(2)
                    elif 1 <= demo_choice_num <= file_demos:
                        # Visualize specific demo
                        print(f"\n--- Demonstration {demo_choice_num}/{file_demos} ---")
                        visualize_demonstration(demo_path, config, demo_index=demo_choice_num-1)
                    else:
                        print(f"Invalid demonstration number. Please enter 0-{file_demos}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                # Only one demo in file
                visualize_demonstration(demo_path, config, demo_index=0)
        else:
            print(f"Invalid choice. Please enter 0-{len(demo_files)} or Q")
    except ValueError:
        print("Invalid input. Please enter a number or Q.")
    
    print("\n=== How to use this dataset ===")
    print("1. The dataset is now ready for training")
    print("2. Run 'python train.py' to start training with this dataset")
    print("3. The dataset will be automatically loaded from the data/demonstrations directory")
    print("4. You can modify the training parameters in config.py if needed")


if __name__ == "__main__":
    main()
