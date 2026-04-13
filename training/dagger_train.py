import os
os.environ["ROBOSUITE_VERBOSE"] = "0"   # 屏蔽 robosuite 的 INFO 日志

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import KDTree
from data_collection import load_demonstrations
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift

# ========== 策略网络 ==========
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x):
        return self.net(x)

# ========== 训练函数 ==========
def train_bc(policy, obs_list, act_list, epochs=50, batch_size=64, lr=1e-3):
    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)

    obs_mean = obs_all.mean(axis=0)
    obs_std = obs_all.std(axis=0)

    obs_norm = (obs_all - obs_mean) / (obs_std + 1e-8)

    obs_t = torch.tensor(obs_norm, dtype=torch.float32)
    act_t = torch.tensor(act_all, dtype=torch.float32)

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_act in loader:
            pred = policy(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")

    return obs_mean, obs_std

# ========== 最近邻专家 ==========
class NearestNeighborExpert:
    def __init__(self, obs_list, act_list):
        self.obs_all = np.concatenate(obs_list, axis=0)
        self.act_all = np.concatenate(act_list, axis=0)
        self.tree = KDTree(self.obs_all)
        print(f"NearestNeighborExpert built with {self.obs_all.shape[0]} state-action pairs.")

    def __call__(self, obs):
        _, idx = self.tree.query(obs)
        return self.act_all[idx]

# ========== DAgger 主循环 ==========
def dagger(env, expert, initial_demos, num_iter=2, steps_per_iter=50, train_epochs=50, max_steps_per_episode=150):
    # 提取初始轨迹
    init_obs_list = []
    init_act_list = []
    for demo in initial_demos:
        obs = demo['observations']['states']
        act = demo['actions']
        init_obs_list.append(obs)
        init_act_list.append(act)

    obs_list = init_obs_list[:]
    act_list = init_act_list[:]

    obs_dim = init_obs_list[0].shape[1]
    act_dim = init_act_list[0].shape[1]

    policy = MLPPolicy(obs_dim, act_dim)

    for it in range(num_iter):
        print(f"\n--- DAgger Iteration {it+1} ---")

        # 训练
        print("Training policy...")
        obs_mean, obs_std = train_bc(policy, obs_list, act_list, epochs=train_epochs)

        # 采集
        print("Collecting new trajectories...")
        new_obs_list = []
        new_act_list = []

        for ep in range(steps_per_iter):
            obs_dict = env.reset()
            done = False
            traj_obs = []
            traj_acts = []
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                state = obs_dict['robot0_proprio-state']
                state_norm = (state - obs_mean) / (obs_std + 1e-8)
                state_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_policy = policy(state_t).squeeze(0).detach().numpy()
                action_expert = expert(state)
                traj_obs.append(state.copy())
                traj_acts.append(action_expert.copy())
                obs_dict, reward, done, info = env.step(action_policy)
                step_count += 1
            new_obs_list.append(np.array(traj_obs))
            new_act_list.append(np.array(traj_acts))
            if (ep+1) % 10 == 0:
                print(f"  Collected {ep+1} trajectories")

        # 聚合
        obs_list.extend(new_obs_list)
        act_list.extend(new_act_list)
        total_samples = sum(len(o) for o in obs_list)
        print(f"Dataset size: {len(obs_list)} trajectories, total samples: {total_samples}")

    return policy, obs_mean, obs_std

# ========== 主程序 ==========
if __name__ == "__main__":
    print("Loading demonstrations for expert...")
    demos = load_demonstrations(data_dir="expert_data", augment=False, normalize=False, chunk_size=None)

    obs_list_raw = []
    act_list_raw = []
    for demo in demos:
        obs_list_raw.append(demo['observations']['states'])
        act_list_raw.append(demo['actions'])

    expert = NearestNeighborExpert(obs_list_raw, act_list_raw)

    # 创建环境（移除了 use_gpu 参数）
    env = TwoArmLift(robots=['Panda','Panda'], has_renderer=False, use_camera_obs=False)

    # 减小采集量避免内存不足
    final_policy, final_obs_mean, final_obs_std = dagger(
        env, expert, demos,
        num_iter=2,           # 2 次迭代
        steps_per_iter=30,    # 每次只采集 30 条轨迹
        train_epochs=30,      # 训练 30 个 epoch
        max_steps_per_episode=150
    )

    torch.save(final_policy.state_dict(), "dagger_policy.pth")
    np.save("dagger_obs_mean.npy", final_obs_mean)
    np.save("dagger_obs_std.npy", final_obs_std)
    print("DAgger finished. Model saved as dagger_policy.pth")