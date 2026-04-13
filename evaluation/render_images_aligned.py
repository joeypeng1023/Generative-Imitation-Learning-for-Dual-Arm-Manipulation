"""
从 HDF5 状态序列渲染图像观测（与 Diffusion Policy 对齐方式一致）
观测和动作一一对应，都是 T 帧

Usage:
    python render_images_aligned.py --input_dir expert_data --output_dir expert_data_imgs --camera frontview
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import robosuite as suite
from config import get_default_config


def create_env_with_camera(camera_names, config):
    """创建支持离屏渲染的 robosuite 环境"""
    env = suite.make(
        env_name=config.env.env_name,
        robots=config.env.robots,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=240,
        camera_widths=320,
        camera_depths=False,
        horizon=config.env.horizon,
        control_freq=config.env.control_freq,
    )
    return env


def test_camera_rendering(env, camera_names):
    """测试相机渲染是否正常"""
    print("\n  Testing camera rendering...")
    env.reset()
    obs = env._get_observations()
    
    print(f"  Available observation keys: {list(obs.keys())}")
    
    for cam in camera_names:
        img_key = f"{cam}_image"
        if img_key in obs:
            img = obs[img_key]
            print(f"  {img_key}: shape {img.shape}, dtype {img.dtype}, "
                  f"pixel range [{img.min()}, {img.max()}]")
            if img.max() == 0:
                print(f"    Warning: {img_key} is all black!")
        elif cam in obs:
            img = obs[cam]
            print(f"  {cam}: shape {img.shape}, dtype {img.dtype}, "
                  f"pixel range [{img.min()}, {img.max()}]")
        else:
            print(f"  Error: Neither '{img_key}' nor '{cam}' found in observations!")
    print()


def render_demo_aligned(env, actions, initial_state, camera_names):
    """
    渲染图像，与动作一一对应（T 个动作对应 T 个观测）
    
    策略：
    1. 设置初始状态
    2. 获取初始观测（作为第0帧）
    3. 对于每个动作：
       - 执行动作
       - 获取新观测（作为下一帧）
    4. 最终得到与动作数量相同的观测帧
    
    Returns:
        images_dict: {camera_name: array(T, H, W, 3), ...}
    """
    # 重置环境并设置初始状态
    env.reset()
    if initial_state is not None and len(initial_state) > 0:
        if hasattr(env, 'sim') and hasattr(env.sim, 'set_state_from_flattened'):
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
    
    num_steps = len(actions)
    images_dict = {cam: [] for cam in camera_names}
    
    # 获取初始观测（第0帧 - 执行任何动作前）
    obs = env._get_observations()
    for cam in camera_names:
        # robosuite 的图像键名通常带 _image 后缀
        img_key = f"{cam}_image"
        if img_key in obs:
            images_dict[cam].append(obs[img_key].copy())
        elif cam in obs:
            images_dict[cam].append(obs[cam].copy())
    
    # 执行动作序列，每个动作后记录观测
    for step_idx in range(num_steps - 1):  # 只执行前 T-1 个动作
        action = actions[step_idx]
        obs, reward, done, info = env.step(action)
        
        # 记录观测
        for cam in camera_names:
            # robosuite 的图像键名通常带 _image 后缀
            img_key = f"{cam}_image"
            if img_key in obs:
                images_dict[cam].append(obs[img_key].copy())
            elif cam in obs:
                images_dict[cam].append(obs[cam].copy())
        
        if done:
            break
    
    # 填充：确保每个相机都有 num_steps 帧图像
    for cam in camera_names:
        if len(images_dict[cam]) == 0:
            # 如果列表为空（没有采集到任何图像），创建黑色图像序列
            print(f"    Warning: No images captured for {cam}, creating blank images")
            empty_img = np.zeros((240, 320, 3), dtype=np.uint8)
            images_dict[cam] = np.array([empty_img] * num_steps, dtype=np.uint8)
        else:
            # 用最后一帧填充剩余位置
            while len(images_dict[cam]) < num_steps:
                images_dict[cam].append(images_dict[cam][-1].copy())
            images_dict[cam] = np.array(images_dict[cam][:num_steps], dtype=np.uint8)
    
    # 打印调试信息
    for cam in camera_names:
        img_array = images_dict[cam]
        print(f"    {cam}: shape {img_array.shape}, dtype {img_array.dtype}, "
              f"pixel range [{img_array.min()}, {img_array.max()}]")
    
    return images_dict


def render_demo_aligned_v2(env, actions, initial_state, camera_names):
    """
    另一种对齐方式：每个动作对应执行后的观测
    
    这种对齐方式下：
    - action[0] 执行后的观测作为 obs[0]
    - action[1] 执行后的观测作为 obs[1]
    - ...以此类推
    
    这样观测和动作数量完全相同，且语义清晰：
    obs[t] 是执行 action[t] 后的结果
    """
    # 重置环境并设置初始状态
    env.reset()
    if initial_state is not None and len(initial_state) > 0:
        if hasattr(env, 'sim') and hasattr(env.sim, 'set_state_from_flattened'):
            env.sim.set_state_from_flattened(initial_state)
            env.sim.forward()
    
    num_steps = len(actions)
    images_dict = {cam: [] for cam in camera_names}
    
    # 执行每个动作，并记录执行后的观测
    for step_idx in range(num_steps):
        action = actions[step_idx]
        obs, reward, done, info = env.step(action)
        
        # 记录执行后的观测
        for cam in camera_names:
            # robosuite 的图像键名通常带 _image 后缀
            img_key = f"{cam}_image"
            if img_key in obs:
                images_dict[cam].append(obs[img_key].copy())
            elif cam in obs:
                images_dict[cam].append(obs[cam].copy())
        
        if done:
            break
    
    # 填充：确保每个相机都有 num_steps 帧图像
    for cam in camera_names:
        if len(images_dict[cam]) == 0:
            # 如果列表为空（没有采集到任何图像），创建黑色图像序列
            print(f"    Warning: No images captured for {cam}, creating blank images")
            empty_img = np.zeros((240, 320, 3), dtype=np.uint8)
            images_dict[cam] = np.array([empty_img] * num_steps, dtype=np.uint8)
        else:
            # 用最后一帧填充剩余位置
            while len(images_dict[cam]) < num_steps:
                images_dict[cam].append(images_dict[cam][-1].copy())
            images_dict[cam] = np.array(images_dict[cam][:num_steps], dtype=np.uint8)
    
    return images_dict


def process_hdf5_file(input_path, output_path, env, camera_names, align_mode='post'):
    """
    处理单个 HDF5 文件
    
    Args:
        align_mode: 'post' 表示 obs[t] 是 action[t] 执行后的观测（推荐）
                   'pre' 表示 obs[t] 是 action[t] 执行前的观测
    """
    print(f"  Reading: {input_path}")
    
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            data_out = f_out.create_group('data')
            
            for demo_key in f_in['data'].keys():
                if not demo_key.startswith('demo_'):
                    continue
                
                print(f"\n  Processing {demo_key}...")
                demo_in = f_in['data'][demo_key]
                
                # 读取数据
                actions = np.array(demo_in['actions'])
                states = np.array(demo_in['states']) if 'states' in demo_in else None
                initial_state = np.array(demo_in['initial_state']) if 'initial_state' in demo_in else None
                
                print(f"    Actions shape: {actions.shape}")
                if states is not None:
                    print(f"    States shape: {states.shape}")
                
                # 渲染图像
                print(f"    Rendering {len(actions)} steps (align_mode={align_mode})...")
                if align_mode == 'post':
                    images_dict = render_demo_aligned_v2(env, actions, initial_state, camera_names)
                else:
                    images_dict = render_demo_aligned(env, actions, initial_state, camera_names)
                
                # 创建输出 demo 组
                demo_out = data_out.create_group(demo_key)
                
                # 保存原始数据
                demo_out.create_dataset('actions', data=actions)
                if states is not None:
                    demo_out.create_dataset('states', data=states)
                if initial_state is not None:
                    demo_out.create_dataset('initial_state', data=initial_state)
                
                # 创建观测组
                obs_out = demo_out.create_group('observations')
                
                # 保存图像数据
                for cam_name, images in images_dict.items():
                    if images is not None:
                        obs_out.create_dataset(f'{cam_name}_image', data=images)
                        print(f"    Saved {cam_name}_image: {images.shape}")
                
                # 验证数量匹配
                for cam_name, images in images_dict.items():
                    if images is not None:
                        assert len(images) == len(actions), \
                            f"Mismatch: {len(images)} images vs {len(actions)} actions"
    
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Render images from state sequences (aligned with actions)'
    )
    parser.add_argument('--input_dir', type=str, default='expert_data',
                        help='Input directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='expert_data_with_images',
                        help='Output directory for HDF5 files with images')
    parser.add_argument('--cameras', nargs='+', default=['agentview'],
                        choices=['frontview', 'agentview', 'birdview', 'sideview', 'rearview'],
                        help='Camera names to render')
    parser.add_argument('--align_mode', type=str, default='post',
                        choices=['pre', 'post'],
                        help='Alignment mode: pre=obs before action, post=obs after action')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip camera rendering test')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Image Rendering (Aligned with Diffusion Policy)")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cameras: {args.cameras}")
    print(f"Align mode: {args.align_mode}")
    print("=" * 70)
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = get_default_config()
    
    print("\nCreating robosuite environment...")
    try:
        env = create_env_with_camera(args.cameras, config)
        print(f"Environment created: {config.env.env_name}")
        print(f"Action dim: {env.action_dim}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 测试相机渲染
    if not args.skip_test:
        test_camera_rendering(env, args.cameras)
    
    hdf5_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.hdf5')])
    
    if args.max_files:
        hdf5_files = hdf5_files[:args.max_files]
    
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")
    print("-" * 70)
    
    success_count = 0
    error_count = 0
    
    for i, filename in enumerate(hdf5_files, 1):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        print(f"\n[{i}/{len(hdf5_files)}] Processing {filename}")
        print("-" * 70)
        
        try:
            process_hdf5_file(input_path, output_path, env, args.cameras, args.align_mode)
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Rendering Completed!")
    print("=" * 70)
    print(f"Total files: {len(hdf5_files)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
