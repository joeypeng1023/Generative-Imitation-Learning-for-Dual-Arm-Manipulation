"""
计算MMD和平滑度指标
"""
import os
os.environ["ROBOSUITE_VERBOSE"] = "0"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from data_collection import load_demonstrations

# 随机采样函数
def sample_states(states, max_samples=2000):
    if len(states) <= max_samples:
        return states
    idx = np.random.choice(len(states), max_samples, replace=False)
    return states[idx]

# 策略类
class MLPPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim)
        )
    def forward(self, x):
        return self.net(x)

def get_state_vector(obs):
    return obs['robot0_proprio-state']

def collect_states(policy, env, obs_mean, obs_std, num_trajs=20, max_steps=200):
    """收集策略的状态轨迹"""
    states = []
    actions = []
    for _ in range(num_trajs):
        obs_dict = env.reset()
        step = 0
        done = False
        while not done and step < max_steps:
            state = get_state_vector(obs_dict)
            states.append(state.copy())
            
            state_norm = (state - obs_mean) / (obs_std + 1e-8)
            state_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_t).squeeze(0).detach().numpy()
            actions.append(action.copy())
            
            obs_dict, reward, done, info = env.step(action)
            step += 1
    return np.array(states), np.array(actions)

def mmd(samples1, samples2, gamma=0.1):
    """Compute MMD between two sets of samples using RBF kernel."""
    K_xx = rbf_kernel(samples1, samples1, gamma=gamma)
    K_yy = rbf_kernel(samples2, samples2, gamma=gamma)
    K_xy = rbf_kernel(samples1, samples2, gamma=gamma)
    n = samples1.shape[0]
    m = samples2.shape[0]
    return (np.sum(K_xx) / (n*n) + np.sum(K_yy) / (m*m) - 2 * np.sum(K_xy) / (n*m))

def compute_smoothness(actions, control_freq=20):
    """计算轨迹平滑度指标"""
    if len(actions) < 2:
        return 0.0, 0.0, 0.0
    
    # 加速度 (二阶导数)
    velocities = np.diff(actions, axis=0) * control_freq
    accelerations = np.diff(velocities, axis=0) * control_freq
    jerks = np.diff(accelerations, axis=0) * control_freq
    
    # 计算指标
    avg_velocity_magnitude = np.mean(np.linalg.norm(velocities, axis=1))
    avg_acceleration_magnitude = np.mean(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0.0
    avg_jerk_magnitude = np.mean(np.linalg.norm(jerks, axis=1)) if len(jerks) > 0 else 0.0
    
    return avg_velocity_magnitude, avg_acceleration_magnitude, avg_jerk_magnitude

def main():
    print("=" * 60)
    print("计算MMD和平滑度指标")
    print("=" * 60)
    
    try:
        # 1. 加载演示状态（训练数据中的状态）
        print("\n1. 加载演示状态...")
        demos = load_demonstrations(data_dir="expert_data", augment=False, normalize=False, chunk_size=None)
        demo_states = []
        demo_actions = []
        for demo in demos:
            if 'states' in demo['observations']:
                demo_states.append(demo['observations']['states'])
            if 'actions' in demo:
                demo_actions.append(demo['actions'])
        
        demo_states = np.concatenate(demo_states, axis=0)
        demo_actions = np.concatenate(demo_actions, axis=0)
        print(f"专家演示: {len(demo_states)} 个状态, {len(demo_actions)} 个动作")
        print(f"状态维度: {demo_states.shape}, 动作维度: {demo_actions.shape}")
        
        # 2. 加载BC策略并采集状态
        print("\n2. 加载BC策略...")
        bc_policy = MLPPolicy(50, 14)
        bc_policy.load_state_dict(torch.load("bc_policy.pth"))
        bc_mean = np.load("obs_mean.npy")
        bc_std = np.load("obs_std.npy")
        env = TwoArmLift(robots=['Panda','Panda'], has_renderer=False, use_camera_obs=False)
        
        print("收集BC策略轨迹...")
        bc_states, bc_actions = collect_states(bc_policy, env, bc_mean, bc_std, num_trajs=5, max_steps=150)
        print(f"BC策略: {len(bc_states)} 个状态, {len(bc_actions)} 个动作")
        
        # 3. 加载DAgger策略并采集状态
        print("\n3. 加载DAgger策略...")
        dagger_policy = MLPPolicy(50, 14)
        dagger_policy.load_state_dict(torch.load("dagger_policy.pth"))
        dagger_mean = np.load("dagger_obs_mean.npy")
        dagger_std = np.load("dagger_obs_std.npy")
        
        print("收集DAgger策略轨迹...")
        dagger_states, dagger_actions = collect_states(dagger_policy, env, dagger_mean, dagger_std, num_trajs=5, max_steps=150)
        print(f"DAgger策略: {len(dagger_states)} 个状态, {len(dagger_actions)} 个动作")
        
        # 采样（加速计算）
        print("\n4. 采样数据（用于MMD计算）...")
        demo_sample = sample_states(demo_states, max_samples=1000)
        bc_sample = sample_states(bc_states, max_samples=1000)
        dagger_sample = sample_states(dagger_states, max_samples=1000)
        
        # 5. 计算MMD
        print("\n5. 计算MMD...")
        mmd_demo_bc = mmd(demo_sample, bc_sample, gamma=0.1)
        mmd_demo_dagger = mmd(demo_sample, dagger_sample, gamma=0.1)
        mmd_bc_dagger = mmd(bc_sample, dagger_sample, gamma=0.1)
        
        print(f"Expert vs BC MMD:      {mmd_demo_bc:.6f}")
        print(f"Expert vs DAgger MMD:  {mmd_demo_dagger:.6f}")
        print(f"BC vs DAgger MMD:      {mmd_bc_dagger:.6f}")
        
        # 6. 计算平滑度指标
        print("\n6. 计算轨迹平滑度指标...")
        
        # 专家演示平滑度
        expert_vel, expert_acc, expert_jerk = compute_smoothness(demo_actions, control_freq=20)
        print(f"专家演示平滑度:")
        print(f"  平均速度幅度: {expert_vel:.6f}")
        print(f"  平均加速度幅度: {expert_acc:.6f}")
        print(f"  平均抖动幅度: {expert_jerk:.6f}")
        
        # BC策略平滑度
        bc_vel, bc_acc, bc_jerk = compute_smoothness(bc_actions, control_freq=20)
        print(f"\nBC策略平滑度:")
        print(f"  平均速度幅度: {bc_vel:.6f}")
        print(f"  平均加速度幅度: {bc_acc:.6f}")
        print(f"  平均抖动幅度: {bc_jerk:.6f}")
        
        # DAgger策略平滑度
        dagger_vel, dagger_acc, dagger_jerk = compute_smoothness(dagger_actions, control_freq=20)
        print(f"\nDAgger策略平滑度:")
        print(f"  平均速度幅度: {dagger_vel:.6f}")
        print(f"  平均加速度幅度: {dagger_acc:.6f}")
        print(f"  平均抖动幅度: {dagger_jerk:.6f}")
        
        # 7. 成功率评估（简单版本）
        print("\n7. 简单成功率评估...")
        # 注：这里只是简单评估，实际需要更复杂的评估逻辑
        
        # 8. 保存结果
        print("\n8. 保存结果...")
        results = {
            'mmd': {
                'expert_vs_bc': float(mmd_demo_bc),
                'expert_vs_dagger': float(mmd_demo_dagger),
                'bc_vs_dagger': float(mmd_bc_dagger)
            },
            'smoothness_expert': {
                'avg_velocity': float(expert_vel),
                'avg_acceleration': float(expert_acc),
                'avg_jerk': float(expert_jerk)
            },
            'smoothness_bc': {
                'avg_velocity': float(bc_vel),
                'avg_acceleration': float(bc_acc),
                'avg_jerk': float(bc_jerk)
            },
            'smoothness_dagger': {
                'avg_velocity': float(dagger_vel),
                'avg_acceleration': float(dagger_acc),
                'avg_jerk': float(dagger_jerk)
            }
        }
        
        import json
        with open('metrics_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ 指标计算完成！结果已保存到 metrics_results.json")
        
        # 9. 打印报告格式
        print("\n" + "=" * 60)
        print("报告格式建议:")
        print("=" * 60)
        print("\nMMD分布偏移量化:")
        print(f"- Expert vs BC: {mmd_demo_bc:.6f}")
        print(f"- Expert vs DAgger: {mmd_demo_dagger:.6f}")
        print(f"- BC vs DAgger: {mmd_bc_dagger:.6f}")
        
        print("\n轨迹平滑度对比:")
        print(f"专家演示 - 加速度: {expert_acc:.6f}, 抖动: {expert_jerk:.6f}")
        print(f"BC策略 - 加速度: {bc_acc:.6f}, 抖动: {bc_jerk:.6f}")
        print(f"DAgger策略 - 加速度: {dagger_acc:.6f}, 抖动: {dagger_jerk:.6f}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()