from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """固定随机种子，减少每次运行结果波动。

    推荐实验里抽样、初始化 embedding、负采样都会引入随机性。
    这个函数只做基础固定，不追求完全逐 bit 可复现。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """选择当前机器最合适的 PyTorch 设备。

    优先级：
    1. CUDA：适合 NVIDIA GPU。
    2. MPS：适合 Apple Silicon / Metal。
    3. CPU：兜底，保证没有 GPU 也能跑。

    本仓库统一用 PyTorch，避免 TensorFlow/TFRS 和 MPS 支持问题。
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
