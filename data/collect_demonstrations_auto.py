"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    import cv2
    
    env.reset()
    
    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    user_exited = False  # flag to indicate if user pressed Esc to exit
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break
        
        # Check for manual save request
        if 'manual_save' in input_ac_dict and input_ac_dict['manual_save']:
            print("Manual save requested, marking task as completed")
            task_completion_hold_count = 0  # Set to 0 to trigger task completion

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        
        # Use offscreen rendering and OpenCV to display
        img = env.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        cv2.imshow("RoboSuite Environment", img)
        
        # Handle OpenCV window events
        key = cv2.waitKey(1) & 0xFF
        # Press 'esc' to quit (avoid conflict with 'q' key used for vertical movement)
        if key == 27:  # 27 is the ASCII code for ESC
            user_exited = True
            break

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                # terminal state reached, continue for a few more steps to capture final state
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    cv2.destroyAllWindows()

    return user_exited


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, demo_number):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file with sequential naming.

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
        demo_number (int): The sequential number for this demonstration file
    """

    print(f"\n=== Starting gather_demonstrations_as_hdf5 ===")
    print(f"Input directory: {directory}")
    print(f"Output directory: {out_dir}")
    
    # Check if input directory exists
    if not os.path.exists(directory):
        print(f"ERROR: Input directory {directory} does not exist!")
        return
    
    # List contents of input directory
    print(f"Contents of input directory:")
    try:
        items = os.listdir(directory)
        print(f"  Found {len(items)} items:")
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                print(f"    [DIR]  {item}")
            else:
                print(f"    [FILE] {item}")
    except Exception as e:
        print(f"ERROR: Failed to list directory contents: {e}")
        return
    
    # Find existing demo files to determine next available number
    existing_files = [f for f in os.listdir(out_dir) if f.startswith('demo') and f.endswith('.hdf5')]
    existing_numbers = []
    for filename in existing_files:
        try:
            # Extract number from filename like "demo01.hdf5" or "demo_1.hdf5"
            num_str = filename.replace('demo', '').replace('.hdf5', '').replace('_', '')
            existing_numbers.append(int(num_str))
        except:
            pass
    
    # Determine the next available number
    if existing_numbers:
        next_num = max(existing_numbers) + 1
    else:
        next_num = 1
    
    # Format filename with leading zeros (e.g., demo01.hdf5, demo02.hdf5)
    hdf5_path = os.path.join(out_dir, f"demo{next_num:02d}.hdf5")
    print(f"Will save to: {hdf5_path}")
    
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")
    print(f"Created 'data' group in HDF5 file")

    num_eps = 0
    env_name = None  # will get populated at some point

    # Get all subdirectories in input directory
    ep_directories = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            ep_directories.append(item)
    
    print(f"Found {len(ep_directories)} subdirectories")
    
    for ep_directory in ep_directories:
        ep_path = os.path.join(directory, ep_directory)
        state_paths = os.path.join(ep_path, "state_*.npz")
        states = []
        actions = []
        success = False

        print(f"\nProcessing subdirectory: {ep_directory}")
        print(f"Looking for state files: {state_paths}")
        state_files = sorted(glob(state_paths))
        print(f"Found {len(state_files)} state files")

        for state_file in state_files:
            print(f"Loading state file: {state_file}")
            try:
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                print(f"  - Environment: {env_name}")
                print(f"  - States: {len(dic['states'])}")
                print(f"  - Action infos: {len(dic['action_infos'])}")
                print(f"  - Successful: {dic['successful']}")

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or dic["successful"]
            except Exception as e:
                print(f"  ERROR: Failed to load state file: {e}")

        print(f"Total states: {len(states)}, Total actions: {len(actions)}, Success: {success}")

        if len(states) == 0:
            print(f"No states found in {ep_directory}, skipping...")
            continue

        # Add demonstration to dataset (regardless of success status for debugging)
        print(f"Adding demonstration to dataset (success={success})...")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        try:
            del states[-1]
        except Exception as e:
            print(f"ERROR: Failed to delete last state: {e}")
            continue
        
        if len(states) != len(actions):
            print(f"Warning: states ({len(states)}) and actions ({len(actions)}) length mismatch!")
            # Adjust to minimum length
            min_len = min(len(states), len(actions))
            states = states[:min_len]
            actions = actions[:min_len]

        num_eps += 1
        ep_data_grp = grp.create_group(f"demo_{num_eps}")
        print(f"Created demo group: demo_{num_eps}")

        # store model xml as an attribute
        xml_path = os.path.join(ep_path, "model.xml")
        if os.path.exists(xml_path):
            try:
                with open(xml_path, "r") as xml_file:
                    xml_str = xml_file.read()
                ep_data_grp.attrs["model_file"] = xml_str
                print(f"Added model.xml as attribute")
            except Exception as e:
                print(f"ERROR: Failed to read model.xml: {e}")
        else:
            print(f"Warning: model.xml not found at {xml_path}")

        # write datasets for states and actions
        try:
            ep_data_grp.create_dataset("states", data=np.array(states))
            print(f"Created 'states' dataset with shape: {np.array(states).shape}")
        except Exception as e:
            print(f"ERROR: Failed to create 'states' dataset: {e}")
            
        try:
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            print(f"Created 'actions' dataset with shape: {np.array(actions).shape}")
        except Exception as e:
            print(f"ERROR: Failed to create 'actions' dataset: {e}")
            
        # Save initial state for consistent visualization and training
        if states:
            try:
                ep_data_grp.create_dataset("initial_state", data=np.array(states[0]))
                print(f"Created 'initial_state' dataset")
            except Exception as e:
                print(f"ERROR: Failed to create 'initial_state' dataset: {e}")
        
        # Mark if this was a successful demonstration
        ep_data_grp.attrs["successful"] = success
        print(f"Set 'successful' attribute to: {success}")
        
        if success:
            print(f"Demonstration {num_eps} is successful and has been saved")
        else:
            print(f"Demonstration {num_eps} is unsuccessful but has been saved for debugging")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__ if hasattr(suite, "__version__") else "unknown"
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    print(f"Added metadata attributes to 'data' group")

    f.close()
    print(f"Closed HDF5 file")
    
    print(f"\n=== gather_demonstrations_as_hdf5 completed ===")
    print(f"Saved demonstration to: {hdf5_path}")
    print(f"Total demonstrations saved: {num_eps}")
    if num_eps == 0:
        print("WARNING: No demonstrations were saved! Check the input directory structure.")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="E:\\DataScience\\Course02\\6019 Embodied AI and Applications\\6019group\\expert_data",
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment with offscreen rendering to avoid MuJoCo viewer issues
    # This will completely bypass the default MuJoCo viewer and its keyboard controls
    env = suite.make(
        **config,
        has_renderer=False,  # Disable default renderer
        has_offscreen_renderer=True,  # Enable offscreen rendering
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap environment with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab character from environment
    if args.device == "keyboard":
        from custom_keyboard import CustomKeyboard
        device = CustomKeyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # Create the base directory if it doesn't exist
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created base directory: {args.directory}")

    # Create environment info for storing in HDF5 file
    env_info = json.dumps({
        "env_name": args.environment,
        "robots": args.robots,
        "controller": args.controller,
        "env_configuration": args.config if "TwoArm" in args.environment else None,
    })

    # collect demonstrations
    demo_count = 0
    while True:
        # Create a new temporary directory for each demonstration
        # Use os.path.join for cross-platform compatibility
        tmp_directory = os.path.join(args.directory, "tmp_{}_{}".format(str(time.time()).replace(".", "_"), demo_count))
        print(f"Creating new temporary directory for demonstration {demo_count + 1}: {tmp_directory}")
        
        # Unwrap the environment if it's already wrapped
        if hasattr(env, 'env'):
            base_env = env.env
        else:
            base_env = env
        
        # Create a fresh environment with a new data collection wrapper
        # First, close the current environment
        if hasattr(env, 'close'):
            env.close()
        
        # Reset the base environment to get a new random initial position for each demonstration
        print(f"Resetting environment to get new random initial position for demonstration {demo_count + 1}...")
        base_env.reset()
        
        # Create a new data collection wrapper with the base environment
        env = DataCollectionWrapper(base_env, tmp_directory)
        
        # collect_human_trajectory returns True if user wants to exit
        if collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode):
            print("Exiting data collection...")
            break
        
        # Gather the demonstration and increment count
        print(f"Gathering demonstration {demo_count + 1} data...")
        gather_demonstrations_as_hdf5(tmp_directory, args.directory, env_info, demo_count + 1)
        demo_count += 1
        print(f"Demonstration {demo_count} saved successfully!")
        
        # Reset the environment for next demonstration
        base_env.reset()
