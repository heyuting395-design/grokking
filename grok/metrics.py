import torch
import math
import torch.nn as nn
from typing import Callable

def compute_measure(
    model: nn.Module,
    init_model: nn.Module, # 如果不需要对比初始模型，可以传 None
    measure_func: Callable,
    operator: str,
    kwargs: dict = {},
    p: int = 1,
) -> float:
    """
    计算每一层的 measure 并聚合
    """
    measure_value = 0
    # 我们主要关注 Linear 和 Embedding 层的权重
    weight_modules = ["Linear", "Embedding"]

    # 递归遍历子模块
    has_children = False
    for child in model.children():
        has_children = True
        # 对应 init_model 的子模块
        init_child = None
        if init_model is not None:
            init_child = list(init_model.children())[0] # 简化处理，这里只是个占位符逻辑

        measure_value += compute_measure(child, init_child, measure_func, operator, kwargs, p)

    if not has_children:
        module_name = model._get_name()
        if module_name in weight_modules:
            # 计算当前层的 measure
            val = measure_func(model, init_model, **kwargs)
            if operator == "sum":
                return val
            elif operator == "product":
                return val # 简化处理，通常我们只用 sum
    
    return measure_value

def norm(module, init_module, p=2):
    """计算权重的 Lp 范数"""
    if hasattr(module, 'weight'):
        return module.weight.norm(p).item()
    return 0.0

def spectral_norm(module, init_module):
    """计算权重的谱范数 (Spectral Norm)"""
    if hasattr(module, 'weight'):
        w = module.weight.data
        if w.dim() > 2: w = w.view(w.size(0), -1)
        # SVD 计算奇异值，最大奇异值即为谱范数
        try:
            _, S, _ = torch.svd(w)
            return S.max().item()
        except:
            return 0.0
    return 0.0