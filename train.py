import argparse
import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

# 确保 grok 文件夹下有对应的模块
from grok.utils import seed_everything, save_config, get_device
from grok.data import get_dataloaders
from grok.transformer import Transformer
from grok.optimizer import CustomAdamW
from grok.metrics import compute_measure, norm
# 如果不需要 Sharpness 可以注释掉，或者确保文件存在
from grok.measure import get_sharpness

# === 1. 学习率调度器 (严格复刻 OpenAI) ===
def get_lr_lambda(warmup_steps, max_lr, min_lr, anneal_lr, anneal_lr_steps):
    def lr_lambda(step):
        # 1. Warmup 阶段
        if step <= warmup_steps:
            return float(step) / max(warmup_steps, 1)
        
        # 2. Constant 或 Decay 阶段
        if not anneal_lr:
            return 1.0  # 保持常数学习率 (Grokking 常用)
        else:
            # Cosine Decay
            if step <= anneal_lr_steps + warmup_steps:
                effective_step = step - warmup_steps
                t = effective_step / anneal_lr_steps
                cos = (1 + np.cos(np.pi * t)) / 2
                target_lr = min_lr + (max_lr - min_lr) * cos
                return target_lr / max_lr
            else:
                return min_lr / max_lr
    return lr_lambda

# === 2. 初始化逻辑 (Grokking 核心) ===
def apply_scaled_init(model, d_model, scale=1.0):
    """
    OpenAI Grokking 论文使用的初始化：N(0, 1/sqrt(d_model)).
    这通常比 PyTorch 默认的 Kaiming 初始化(sqrt(1/fan_in))要小。
    较小的初始化有助于延长"记忆阶段"，从而让 Grokking 更明显。
    """
    std = (1.0 / math.sqrt(d_model)) * scale
    print(f"Applying Scaled Initialization: std={std:.5f} (scale={scale})")
    
    for name, p in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(p)
        elif "weight" in name:
            # 只对 2D 权重（Linear, Embedding）使用该初始化
            # LayerNorm (1D) 保持默认
            if p.dim() > 1: 
                nn.init.normal_(p, mean=0.0, std=std)

def main(args):
    # 1. 基础设置
    seed_everything(args.seed)
    device = get_device()
    
    # 实验命名: 包含关键参数以便区分
    exp_name = f"grok_mult_p{args.train_pct}_wd{args.weight_decay}_sd{args.seed}"
    save_dir = os.path.join("logs", exp_name)
    save_config(args, save_dir)
    print(f"=== Experiment Started: {exp_name} ===")
    print(f"Device: {device}")
    
    # 2. 数据准备
    train_loader, val_loader, vocab = get_dataloaders(args)
    vocab_size = len(vocab)
    
    # 3. 初始化模型
    model = Transformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        dropout=args.dropout,
        max_context_len=args.max_context_len,
        vocab_len=vocab_size,
        non_linearity=args.non_linearity,
        weight_noise=args.weight_noise
    ).to(device)
    
    # 应用论文标准的初始化
    apply_scaled_init(model, args.d_model, scale=args.init_std_scale)
    
    # 4. 优化器 (参数严格对齐论文)
    optimizer = CustomAdamW(
        model.parameters(),
        lr=args.max_lr, 
        betas=(0.9, 0.98), # 论文指定
        eps=1e-8,
        weight_decay=args.weight_decay, # 默认为 1.0
        noise_factor=args.noise_factor,
        weight_decay_form=args.weight_decay_kind
    )
    
    # 5. 调度器
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=get_lr_lambda(
            args.warmup_steps, args.max_lr, args.max_lr/10, 
            args.anneal_lr, args.anneal_lr_steps
        )
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 6. 训练循环
    history = []
    global_step = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(range(args.max_epochs), desc="Training", unit="ep")
    
    for epoch in pbar:
        # --- Train ---
        model.train()
        train_loss_accum = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            global_step += 1
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output, _, _ = model(inputs) # Forward
            last_token_logits = output[:, -1, :]  # 形状: [batch_size, vocab_size]
            last_token_target = targets[:, -1]    # 形状: [batch_size]

# 现在的 loss 只由“计算答案”的正确与否决定
            loss = criterion(last_token_logits, last_token_target)
            # Loss 计算 (Flatten)
            loss.backward()
            optimizer.step()
            scheduler.step() # 每个 Step 更新学习率
            
            train_loss_accum += loss.item()
            last_preds = last_token_logits.argmax(dim=-1)  # [B]
            train_correct += (last_preds == last_token_target).sum().item()
            train_total += last_token_target.size(0)
            
        train_acc = train_correct / train_total
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_accum = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output, _, _ = model(inputs)
                
                val_loss = criterion(output[:, -1, :], targets[:, -1])
                val_loss_accum += val_loss.item()
                
                last_preds = output[:, -1, :].argmax(dim=-1)
                val_correct += (last_preds == targets[:, -1]).sum().item()
                val_total += targets.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # --- Logging ---
        # 定期记录指标
        if epoch % args.log_interval == 0 or epoch == args.max_epochs - 1:
            # 计算 Sharpness (为了速度只在记录时计算)
            sharpness = 0.0
            if args.compute_sharpness:
                # 注意：这里会采样一个 batch 来计算
                sharpness = get_sharpness(train_loader, model)
            
            # 计算 L2 Norm
            l2_norm = compute_measure(model, None, norm, "sum", {"p": 2}, p=2)
            
            log_entry = {
                "epoch": epoch,
                "step": global_step,
                "train_acc": train_acc,
                "train_loss": avg_train_loss,
                "val_acc": val_acc,
                "val_loss": avg_val_loss,
                "sharpness": sharpness,
                "l2_norm": l2_norm,
                "lr": optimizer.param_groups[0]['lr']
            }
            history.append(log_entry)
            
            # 实时保存 CSV
            pd.DataFrame(history).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
            
            # 更新进度条后缀信息
            pbar.set_description(
                f"Ep {epoch} | T_Acc {train_acc:.3f} | V_Acc {val_acc:.3f} | V_Loss {avg_val_loss:.3f}"
            )

    # 训练结束，保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print(f"Training Complete. Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokking: Modular Multiplication")
    
    # === 数据参数 ===
    parser.add_argument("--modulus", type=int, default=113)
    # 关键修改：默认 train_pct 设为 0.5。乘法任务如果给 50% 数据太容易，Grokking 不明显。
    parser.add_argument("--train_pct", type=float, default=0.3, help="Fraction of data for training (0.3 recommended for grokking)")
    parser.add_argument("--seed", type=int, default=42)
    
    # === 模型参数 ===
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--max_context_len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--non_linearity", type=str, default="relu")
    parser.add_argument("--weight_noise", type=float, default=0.0)
    
    # === 训练参数 ===
    parser.add_argument("--batch_size", type=int, default=512)
    # 关键修改：Epoch 增加到 15000，保证有足够时间 Grokking
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    
    # === 正则化参数 (核心) ===
    # Grokking 需要较强的 Weight Decay
    parser.add_argument("--weight_decay", type=float, default=0.3)
    parser.add_argument("--weight_decay_kind", type=str, default="to_zero")
    parser.add_argument("--noise_factor", type=float, default=0.0)
    
    # === 初始化参数 ===
    # 默认 1.0 对应 1/sqrt(d)。如果要更难的任务，可以设为 < 1.0
    parser.add_argument("--init_std_scale", type=float, default=1.0)
    
    # === 调度器 & 日志 ===
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--anneal_lr", action="store_true", default=False)
    parser.add_argument("--anneal_lr_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--compute_sharpness", action="store_true", default=True)
    
    args = parser.parse_args()
    main(args)