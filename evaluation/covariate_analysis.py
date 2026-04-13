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
    states = []
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
            obs_dict, reward, done, info = env.step(action)
            step += 1
    return np.array(states)

def mmd(samples1, samples2, gamma=0.1):
    """Compute MMD between two sets of samples using RBF kernel."""
    K_xx = rbf_kernel(samples1, samples1, gamma=gamma)
    K_yy = rbf_kernel(samples2, samples2, gamma=gamma)
    K_xy = rbf_kernel(samples1, samples2, gamma=gamma)
    n = samples1.shape[0]
    m = samples2.shape[0]
    return (np.sum(K_xx) / (n*n) + np.sum(K_yy) / (m*m) - 2 * np.sum(K_xy) / (n*m))

if __name__ == "__main__":
    # 1. 加载演示状态（训练数据中的状态）
    print("Loading demonstration states...")
    demos = load_demonstrations(data_dir="expert_data", augment=False, normalize=False, chunk_size=None)
    demo_states = []
    for demo in demos:
        obs_traj = demo['observations']['states']   # (T, 50)
        demo_states.append(obs_traj)
    demo_states = np.concatenate(demo_states, axis=0)
    print(f"Demo states shape: {demo_states.shape}")

    # 2. 加载 BC 策略并采集状态
    print("Collecting BC rollout states...")
    bc_policy = MLPPolicy(50, 14)
    bc_policy.load_state_dict(torch.load("bc_policy.pth"))
    bc_mean = np.load("obs_mean.npy")
    bc_std = np.load("obs_std.npy")
    env = TwoArmLift(robots=['Panda','Panda'], has_renderer=False, use_camera_obs=False)
    bc_states = collect_states(bc_policy, env, bc_mean, bc_std, num_trajs=20, max_steps=200)
    print(f"BC rollout states shape: {bc_states.shape}")

    # 3. 加载 DAgger 策略并采集状态
    print("Collecting DAgger rollout states...")
    dagger_policy = MLPPolicy(50, 14)
    dagger_policy.load_state_dict(torch.load("dagger_policy.pth"))
    dagger_mean = np.load("dagger_obs_mean.npy")
    dagger_std = np.load("dagger_obs_std.npy")
    dagger_states = collect_states(dagger_policy, env, dagger_mean, dagger_std, num_trajs=20, max_steps=200)
    print(f"DAgger rollout states shape: {dagger_states.shape}")

    # 采样（加速）
    demo_sample = sample_states(demo_states, max_samples=2000)
    bc_sample = sample_states(bc_states, max_samples=2000)
    dagger_sample = sample_states(dagger_states, max_samples=2000)

    # 4. 计算 MMD（使用采样后数据）
    print("\nComputing MMD...")
    mmd_demo_bc = mmd(demo_sample, bc_sample)
    mmd_demo_dagger = mmd(demo_sample, dagger_sample)
    mmd_bc_dagger = mmd(bc_sample, dagger_sample)
    print(f"Demo vs BC: {mmd_demo_bc:.6f}")
    print(f"Demo vs DAgger: {mmd_demo_dagger:.6f}")
    print(f"BC vs DAgger: {mmd_bc_dagger:.6f}")

    # 5. PCA 降维并绘图（使用采样后数据）
    print("\nPerforming PCA and plotting...")
    all_states = np.vstack([demo_sample, bc_sample, dagger_sample])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_states)

    n_demo = len(demo_sample)
    n_bc = len(bc_sample)
    demo_2d = all_2d[:n_demo]
    bc_2d = all_2d[n_demo:n_demo+n_bc]
    dagger_2d = all_2d[n_demo+n_bc:]

    plt.figure(figsize=(8,6))
    plt.scatter(demo_2d[:,0], demo_2d[:,1], s=2, alpha=0.6, label='Demonstration', c='blue')
    plt.scatter(bc_2d[:,0], bc_2d[:,1], s=2, alpha=0.6, label='BC', c='orange')
    plt.scatter(dagger_2d[:,0], dagger_2d[:,1], s=2, alpha=0.6, label='DAgger', c='green')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('State Distribution PCA (sampled)')
    plt.legend()
    plt.savefig('state_distribution_pca.png', dpi=150)
    plt.show()
    print("Saved figure: state_distribution_pca.png")