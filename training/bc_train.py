import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_collection import load_demonstrations

# 加载演示数据
demos = load_demonstrations(data_dir="expert_data", augment=False, normalize=False, chunk_size=None)

# 提取观测和动作
obs_list = []
act_list = []
for demo in demos:
    # 根据实际键名 'states' 提取观测
    obs = demo['observations']['states']   # shape (T, obs_dim)
    act = demo['actions']                  # shape (T, act_dim)
    obs_list.append(obs)
    act_list.append(act)

# 展平所有轨迹
obs_all = np.concatenate(obs_list, axis=0)
act_all = np.concatenate(act_list, axis=0)

print("观测总样本数：", obs_all.shape[0])
print("观测维度：", obs_all.shape[1])
print("动作维度：", act_all.shape[1])

# 归一化观测
obs_mean = obs_all.mean(axis=0)
obs_std = obs_all.std(axis=0)
obs_all_norm = (obs_all - obs_mean) / (obs_std + 1e-8)

# 转换为 PyTorch 张量
obs_t = torch.tensor(obs_all_norm, dtype=torch.float32)
act_t = torch.tensor(act_all, dtype=torch.float32)

dataset = TensorDataset(obs_t, act_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 策略网络
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

obs_dim = obs_all.shape[1]
act_dim = act_all.shape[1]
policy = MLPPolicy(obs_dim, act_dim)

# 训练
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for batch_obs, batch_act in dataloader:
        pred = policy(batch_obs)
        loss = criterion(pred, batch_act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.6f}")

# 保存模型和归一化参数
torch.save(policy.state_dict(), "bc_policy.pth")
np.save("obs_mean.npy", obs_mean)
np.save("obs_std.npy", obs_std)
print("模型已保存为 bc_policy.pth，归一化参数已保存。")