#!/usr/bin/env python3
"""
Script to help collect human demonstrations using collect_demonstrations_auto.py
and move the generated data to the correct directory.
"""

import os
import sys
import shutil
import subprocess
from config import get_default_config

def run_collect_demonstrations():
    """Run collect_demonstrations_auto.py for human data collection"""
    config = get_default_config()
    
    # Expert data directory
    expert_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_data")
    
    # Build the command
    cmd = [
        sys.executable,
        "collect_demonstrations_auto.py",
        "--directory", expert_data_dir,
        "--environment", config.env.env_name,
        "--robots", *config.env.robots,
        "--arm", config.env.arm,
        "--camera", *config.env.camera,
        "--device", config.env.device,
        "--renderer", config.env.renderer,
        "--max_fr", str(config.env.max_fr),
        "--goal_update_mode", config.env.goal_update_mode,
        "--pos-sensitivity", str(config.env.pos_sensitivity),
        "--rot-sensitivity", str(config.env.rot_sensitivity),
        "--reverse_xy", str(config.env.reverse_xy),
    ]
    
    if config.env.controller:
        cmd.extend(["--controller", config.env.controller])
    
    # Print command for user reference
    print("Running command:")
    print(' '.join(cmd))
    print("\n=== Human Demonstration Collection ===")
    print("Instructions:")
    print("1. Use the keyboard to control the robot:")
    print("   - WASD: Move arm in X-Y plane")
    print("   - QE: Move arm up/down (Z axis)")
    print("   - RF: Rotate arm around X axis")
    print("   - TG: Rotate arm around Y axis")
    print("   - YH: Rotate arm around Z axis")
    print("   - V: Open/close gripper")
    print("   - B: Switch between arms (if bimanual)")
    print("   - Space: Reset the environment")
    print("   - ESC: Exit data collection")
    print("2. Try to complete the task (lift the object) successfully")
    print("3. After each successful demonstration, you'll be prompted to continue")
    print("4. Press ESC when you're done collecting demonstrations")
    print("\nPress Enter to start...")
    input()
    
    # Run the command
    subprocess.run(cmd)

def move_demonstrations():
    """Move collected demonstrations to the correct directory"""
    config = get_default_config()
    
    # Expert data directory where collect_demonstrations_auto.py saves data
    expert_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_data")
    target_dir = config.data.data_dir
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n=== Moving Demonstrations ===")
    print(f"Looking for demonstrations in: {expert_data_dir}")
    print(f"Will move to: {target_dir}")
    
    if not os.path.exists(expert_data_dir):
        print(f"Directory {expert_data_dir} does not exist. No demonstrations found.")
        return
    
    # Find all demonstration files
    demo_files = []
    for file in os.listdir(expert_data_dir):
        if file.endswith('.hdf5'):
            demo_files.append(os.path.join(expert_data_dir, file))
    
    if not demo_files:
        print("No demonstration files found.")
        return
    
    print(f"Found {len(demo_files)} demonstration files.")
    
    # Move files to target directory
    # Find existing demo files to determine next number
    existing_demos = [f for f in os.listdir(target_dir) if f.startswith('demo_') and f.endswith('.h5')]
    existing_numbers = []
    for demo_file in existing_demos:
        try:
            num = int(demo_file.split('_')[1].split('.')[0])
            existing_numbers.append(num)
        except:
            pass
    
    # Determine the starting number
    if existing_numbers:
        start_num = max(existing_numbers) + 1
    else:
        start_num = 0
    
    # Copy files with sequential numbering
    for i, demo_file in enumerate(demo_files):
        dest_file = os.path.join(target_dir, f"demo_{start_num + i}.h5")
        shutil.copy2(demo_file, dest_file)
        print(f"Copied {os.path.basename(demo_file)} to {dest_file}")
    
    print(f"\nSuccessfully moved {len(demo_files)} demonstration files to {target_dir}")
    print("You can now use these demonstrations for training.")

def main():
    """Main function"""
    print("=== Human Demonstration Collection Tool ===")
    print("This tool helps you collect human demonstrations for dual-arm manipulation.")
    print("\nSteps:")
    print("1. Run collect_demonstrations_auto.py to collect demonstrations")
    print("2. Move collected demonstrations to the data directory")
    print("3. Verify the demonstrations")
    print("\nPress 1 to start collecting demonstrations")
    print("Press 2 to move existing demonstrations (if you already collected them)")
    print("Press 3 to exit")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        run_collect_demonstrations()
        move_demonstrations()
    elif choice == '2':
        move_demonstrations()
    elif choice == '3':
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")
        return
    
    print("\n=== Next Steps ===")
    print("1. To visualize your demonstrations: python visualize_demonstrations.py")
    print("2. To start training with your demonstrations: python train.py")
    print("3. To collect more demonstrations, run this script again")

if __name__ == "__main__":
    main()
