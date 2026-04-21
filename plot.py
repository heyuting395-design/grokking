import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_results(log_dir):
    csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: No metrics.csv found in {log_dir}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建画布
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 设置网格 (替代 sns.set_style("whitegrid"))
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # === 绘制 Accuracy (左轴, 蓝色) ===
    # 使用 ax.plot 替代 sns.lineplot
    l1 = ax1.plot(df['epoch'], df['train_acc'], label='Train Acc', color='tab:blue', linestyle='--', alpha=0.7)
    l2 = ax1.plot(df['epoch'], df['val_acc'], label='Val Acc', color='tab:blue', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.05, 1.05) # 限制范围 0-1

    # === 绘制 Loss (右轴, 红色) ===
    ax2 = ax1.twinx()
    l3 = ax2.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:red', linestyle='--', alpha=0.5)
    l4 = ax2.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:red', alpha=0.5)
    
    ax2.set_ylabel('Loss (Log Scale)', color='tab:red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_yscale('log') # 对数坐标

    # === 合并图例 (优化点) ===
    # 将左右两轴的线条合并到一个图例中，方便查看
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, shadow=True)

    # 标题和保存
    plt.title(f"Grokking Results ({os.path.basename(log_dir)})")
    plt.tight_layout()
    
    save_path = os.path.join(log_dir, "results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to log directory")
    args = parser.parse_args()
    
    plot_results(args.dir)