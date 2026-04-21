import logging
import torch
import numpy as np
import scipy.optimize

def get_loss_and_grads(x, model, data_loader):
    model.eval()
    
    # 将一维的 x 向量还原回模型的参数
    x_start = 0
    for p in model.parameters():
        param_size = p.data.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = x[x_start : x_start + param_idx]
        p.data = torch.Tensor(x_part.reshape(param_size)).to(p.device)
        x_start += param_idx

    batch_losses = []
    batch_grads = []
    
    # 计算整个数据集上的 Loss 和梯度
    # 注意：这里我们简化逻辑，只取一个 batch 以加速计算 (Grokking 实验中常用做法)
    # 如果 data_loader 是全量数据，那就是全量 Sharpness
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        with torch.set_grad_enabled(True):
            # Forward
            output, _, _ = model(inputs)
            # Output shape: [batch, seq, vocab] -> Flatten
            vocab_size = output.size(-1)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
        
        batch_losses.append(loss.item())
        
        # 收集梯度
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1).cpu().numpy())
            else:
                grads.append(np.zeros(p.numel()))
        batch_grads.append(np.concatenate(grads))
        
        # 为了速度，通常只算一个 Batch 就可以代表局部几何特征
        break 

    mean_loss = np.mean(batch_losses)
    mean_grad = np.mean(np.stack(batch_grads), axis=0)

    return mean_loss, mean_grad.astype(np.float64)


def get_weights(model):
    """将模型所有参数展平成一个 numpy 向量"""
    x0 = None
    for p in model.parameters():
        if x0 is None:
            x0 = p.data.view(-1)
        else:
            x0 = torch.cat((x0, p.data.view(-1)))
    return x0.cpu().numpy()


def get_sharpness(data_loader, model, subspace_dim=10, epsilon=1e-3, maxiter=10):
    """
    计算 Keskar et. al. 定义的 Sharpness。
    在参数空间的一个随机子空间内，寻找 Loss 最大的点。
    """
    x0 = get_weights(model)
    f_x0, _ = get_loss_and_grads(x0, model, data_loader)
    
    # 由于 scipy 是 minimize，我们需要 minimize(-loss) 来 maximize loss
    # 目标：找到 max_loss
    
    # 1. 构建随机投影矩阵 A (Subspace)
    # A_plus shape: [subspace_dim, n_params]
    A_plus = np.random.rand(subspace_dim, x0.shape[0]) * 2.0 - 1.0
    A_plus_norm = np.linalg.norm(A_plus, axis=1)
    A_plus = A_plus / np.reshape(A_plus_norm, (subspace_dim, 1))
    A = np.linalg.pinv(A_plus)

    # 2. 定义边界
    abs_bound = epsilon * (np.abs(np.dot(A_plus, x0)) + 1)
    abs_bound = np.reshape(abs_bound, (abs_bound.shape[0], 1))
    bounds = np.concatenate([-abs_bound, abs_bound], 1)

    # 3. 定义优化目标函数
    def func(y):
        # x = x0 + A*y
        f_loss, f_grads = get_loss_and_grads(
            x0 + np.dot(A, y),
            model,
            data_loader,
        )
        # Chain rule: grad_y = A^T * grad_x
        # Minimize negative loss
        return -f_loss, -np.dot(np.transpose(A), f_grads)

    init_guess = np.zeros(subspace_dim)

    # 4. 执行优化
    try:
        minimum_x, min_neg_loss, d = scipy.optimize.fmin_l_bfgs_b(
            func,
            init_guess,
            maxiter=maxiter,
            bounds=bounds,
            disp=0, # 不打印过程
        )
        max_loss = -min_neg_loss
    except Exception as e:
        print(f"Sharpness calc failed: {e}")
        max_loss = f_x0

    # 5. 还原模型参数 (非常重要！否则模型参数就被改乱了)
    x_start = 0
    for p in model.parameters():
        param_size = p.data.size()
        param_idx = 1
        for s in param_size:
            param_idx *= s
        x_part = torch.from_numpy(x0[x_start : x_start + param_idx]).float().to(p.device)
        p.data = x_part.view(param_size)
        x_start += param_idx

    # Sharpness 公式
    phi = (max_loss - f_x0) / (1 + f_x0) * 100
    return phi