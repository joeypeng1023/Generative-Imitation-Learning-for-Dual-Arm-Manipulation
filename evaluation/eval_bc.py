import torch
import numpy as np
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift


# 复制 MLPPolicy 类定义
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
    # 假设训练时使用的是 robot0_proprio-state（50 维）
    return obs['robot0_proprio-state']


def evaluate_bc(policy, env, obs_mean, obs_std, num_episodes=10):
    success = 0
    for ep in range(num_episodes):
        obs_dict = env.reset()
        done = False
        while not done:
            state = get_state_vector(obs_dict)
            state_norm = (state - obs_mean) / (obs_std + 1e-8)
            state_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_t).squeeze(0).numpy()
            obs_dict, reward, done, info = env.step(action)
        if info.get('success', False):
            success += 1
        print(f"Episode {ep + 1}: {'Success' if info.get('success') else 'Fail'}")
    return success / num_episodes


if __name__ == "__main__":
    # 加载模型和归一化参数
    obs_dim = 50  # 训练时的观测维度
    act_dim = 14
    policy = MLPPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load("bc_policy.pth"))
    obs_mean = np.load("obs_mean.npy")
    obs_std = np.load("obs_std.npy")

    # 创建环境（关闭渲染，使用低维观测）
    env = TwoArmLift(robots=['Panda', 'Panda'], has_renderer=False, use_camera_obs=False)

    # 评估
    rate = evaluate_bc(policy, env, obs_mean, obs_std, num_episodes=5)
    print(f"BC 成功率: {rate:.2f}")