import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置图表风格
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "./evaluation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dummy_data():
    """
    生成假数据用于测试流水线。
    后期请替换为: return pd.read_csv("real_experiment_results.csv")
    """
    print("正在生成模拟测试数据...")

    # 1. 成功率 vs 演示数据量 (Demonstrations)
    demo_counts = [10, 50, 100, 200, 500]
    results = []
    for count in demo_counts:
        # 模拟不同算法的成功率 (Diffusion > DAgger > BC)
        results.append({"Demos": count, "Algorithm": "BC",
                        "Success_Rate": min(0.4 + count * 0.0005 + np.random.normal(0, 0.05), 1.0)})
        results.append({"Demos": count, "Algorithm": "DAgger",
                        "Success_Rate": min(0.5 + count * 0.0008 + np.random.normal(0, 0.05), 1.0)})
        results.append({"Demos": count, "Algorithm": "Diffusion Policy",
                        "Success_Rate": min(0.6 + count * 0.001 + np.random.normal(0, 0.02), 1.0)})

    df_scaling = pd.DataFrame(results)

    # 2. 扩散策略消融实验数据 (历史长度 vs 噪声步数)
    history_lengths = [1, 2, 4, 8]
    noise_steps = [10, 50, 100]
    ablation_results = []
    for hl in history_lengths:
        for ns in noise_steps:
            # 模拟：适当的历史长度和噪声步数效果最好
            score = 0.8 - abs(hl - 4) * 0.05 - abs(ns - 50) * 0.002 + np.random.normal(0, 0.02)
            ablation_results.append({"History_Length": hl, "Noise_Steps": ns, "Success_Rate": min(max(score, 0), 1.0)})

    df_ablation = pd.DataFrame(ablation_results)

    return df_scaling, df_ablation


def plot_data_scaling(df):
    """绘制：演示数据量 vs 成功率折线图"""
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Demos", y="Success_Rate", hue="Algorithm", marker="o", linewidth=2.5)

    plt.title("Task Success Rate vs. Number of Demonstrations", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Expert Demonstrations", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.ylim(0, 1.1)

    save_path = os.path.join(OUTPUT_DIR, "data_scaling_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


def plot_ablation_heatmap(df):
    """绘制：历史长度与噪声步数的消融实验热力图"""
    plt.figure(figsize=(7, 5))
    # 转换数据格式以适应热力图
    pivot_df = df.pivot(index="History_Length", columns="Noise_Steps", values="Success_Rate")

    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0.4, vmax=1.0)
    plt.title("Ablation: History Length vs. Noise Steps (Success Rate)", fontsize=14, fontweight='bold')
    plt.xlabel("Noise Steps", fontsize=12)
    plt.ylabel("Observation History Length", fontsize=12)
    plt.gca().invert_yaxis()  # 习惯上从小到大排列

    save_path = os.path.join(OUTPUT_DIR, "diffusion_ablation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 1. 获取数据
    df_scaling, df_ablation = generate_dummy_data()

    # 2. 生成可视化图表
    print("正在生成可视化图表...")
    plot_data_scaling(df_scaling)
    plot_ablation_heatmap(df_ablation)
    print("所有可视化流水线执行完毕！")