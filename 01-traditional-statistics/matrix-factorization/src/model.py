from __future__ import annotations

"""PyTorch 矩阵分解模型。

模型意图：
把 userId 和 movieId 映射成 embedding，通过点积预测评分。
同时加入用户偏置、电影偏置和全局均值，让模型能处理用户打分宽松程度
和电影整体受欢迎程度。
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from utils.movielens import make_id_maps
from utils.torch_utils import cleanup_dataloaders, dataloader_kwargs, get_device, seed_everything


class MatrixFactorization(nn.Module):
    """带偏置的矩阵分解模型。

    基础矩阵分解只做 `user_embedding dot movie_embedding`。
    这里额外加入：
    - 全局均值：数据集整体评分水平。
    - 用户偏置：有些用户天生打分更宽松或更严格。
    - 电影偏置：有些电影整体更容易拿高分。
    """

    def __init__(self, n_users: int, n_movies: int, n_factors: int, global_mean: float) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """预测一批用户-电影评分。

        输出是连续分数，用 MSELoss 和真实 1-5 评分做回归。
        """

        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        dot = (user_vec * movie_vec).sum(dim=1)
        user_bias = self.user_bias(user_ids).squeeze(1)
        movie_bias = self.movie_bias(movie_ids).squeeze(1)
        return self.global_mean + user_bias + movie_bias + dot


@dataclass
class MFResult:
    """矩阵分解训练结果。

    训练后除了模型本身，还要保留 ID 映射。
    没有这些映射，报告阶段就无法把 embedding 下标转回原始 movieId。
    """

    model: MatrixFactorization
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: dict[int, int]
    metrics: dict[str, float]
    device_name: str


def _to_tensors(ratings: pd.DataFrame, user_to_index: dict[int, int], movie_to_index: dict[int, int]) -> TensorDataset:
    """把评分 DataFrame 转成 PyTorch TensorDataset。

    这里假设传入 ratings 已经过滤到训练中见过的用户和电影。
    如果直接 map 冷启动 ID，会出现 NaN，不能转成 embedding 下标。
    """

    users = ratings["userId"].map(user_to_index).to_numpy()
    movies = ratings["movieId"].map(movie_to_index).to_numpy()
    y = ratings["rating"].astype("float32").to_numpy()
    return TensorDataset(
        torch.tensor(users, dtype=torch.long),
        torch.tensor(movies, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )


def train_mf(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    n_factors: int = 64,
    max_epochs: int = 1000,
    patience: int = 5,
    batch_size: int = 4096,
    num_workers: int = 8,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    keep_checkpoints: int = 3,
) -> MFResult:
    """训练矩阵分解模型，并用验证集 early stopping。

    这里不自己写优化器细节，直接用 PyTorch 的 Adam 和 DataLoader。
    `max_epochs` 给模型足够训练空间，`patience` 防止无意义地继续训练。

    checkpoint 策略非常克制：
    - best 模型只在训练结束后写一次，避免每次提升都写 SSD。
    - 中间 checkpoint 默认关闭；如果传 `checkpoint_every > 0`，也只保留少量。
    """

    seed_everything(42)
    # 只用训练集建立 ID 映射，避免验证集/测试集信息提前进入模型。
    user_to_index, _ = make_id_maps(train["userId"])
    movie_to_index, index_to_movie = make_id_maps(train["movieId"])

    # 验证集中可能出现训练集没见过的用户或电影。第一版直接过滤掉，
    # 因为 embedding 模型无法给完全没见过的 ID 查向量。
    valid_known = valid[valid["userId"].isin(user_to_index) & valid["movieId"].isin(movie_to_index)].copy()

    device = get_device()
    model = MatrixFactorization(len(user_to_index), len(movie_to_index), n_factors, train["rating"].mean()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # 训练 DataLoader 使用通用性能参数：
    # CUDA 才 pin_memory，MPS/CPU 不 pin；num_workers > 0 才 persistent_workers。
    loader = DataLoader(
        _to_tensors(train, user_to_index, movie_to_index),
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs(device, num_workers),
    )
    best_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    saved_intermediates: list[Path] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for users, movies, ratings in tqdm(loader, desc=f"MF epoch {epoch + 1}/{max_epochs}", unit="batch"):
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad()
            predictions = model(users, movies)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        valid_metrics = evaluate_mf(model, valid_known, user_to_index, movie_to_index, device)
        valid_rmse = valid_metrics["rmse"]
        print(
            f"[MF] epoch={epoch + 1} "
            f"avg_loss={total_loss / max(len(loader), 1):.4f} "
            f"valid_rmse={valid_rmse:.4f}"
        )

        # Early stopping 的意图：只要验证集不再变好，就停止训练。
        # 这样比固定 5 轮更稳，也避免明明已经过拟合还继续跑。
        if valid_rmse < best_rmse - 1e-4:
            best_rmse = valid_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[MF] early stopping: validation RMSE has not improved for {patience} epochs.")
                break

        # 中间 checkpoint 不是主线，默认 checkpoint_every=0 不会写。
        # 如果用户打开，也只保留少量最近文件。
        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "valid_rmse": valid_rmse,
                },
                checkpoint_path,
            )
            saved_intermediates.append(checkpoint_path)
            while len(saved_intermediates) > keep_checkpoints:
                old_path = saved_intermediates.pop(0)
                old_path.unlink(missing_ok=True)

    # 训练循环结束后主动关闭 DataLoader worker。
    # macOS + persistent_workers 有时会导致脚本最后不退出，这里集中清理。
    cleanup_dataloaders(loader)

    model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        # best.pt 只在训练结束后写一次，避免每次验证提升都覆盖磁盘。
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "best_valid_rmse": best_rmse,
                "n_factors": n_factors,
            },
            checkpoint_dir / "best.pt",
        )
    metrics = evaluate_mf(model, valid_known, user_to_index, movie_to_index, device)
    return MFResult(model, user_to_index, movie_to_index, index_to_movie, metrics, str(device))


def evaluate_mf(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    device: torch.device,
) -> dict[str, float]:
    """在测试集上计算 RMSE 和 MAE。

    RMSE 对大误差更敏感，MAE 更直观。
    两个指标一起看，可以避免只看一个数字误判模型表现。
    """

    if ratings.empty:
        return {"rmse": 0.0, "mae": 0.0}

    dataset = _to_tensors(ratings, user_to_index, movie_to_index)
    # 评估阶段不反向传播，也不需要 shuffle。
    # 保持 num_workers=0，让评估逻辑更简单稳定。
    loader = DataLoader(dataset, batch_size=8192, num_workers=0)
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for users, movies, y in tqdm(loader, desc="Evaluating MF", unit="batch"):
            pred = model(users.to(device), movies.to(device)).detach().cpu().numpy()
            preds.append(pred)
            actuals.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def similar_movies(result: MFResult, movie_id: int, top_k: int = 10) -> list[tuple[int, float]]:
    """基于电影 embedding 找相似电影。

    这不是独立的推荐算法，而是帮助初学者理解 embedding：
    如果模型学得合理，相似电影在向量空间里应该更接近。
    """

    if movie_id not in result.movie_to_index:
        return []
    model = result.model
    model.eval()
    with torch.no_grad():
        # 把 embedding 拉回 CPU 做相似度，避免为了报告样例占用 GPU/MPS。
        embeddings = model.movie_embedding.weight.detach().cpu()
        target_index = result.movie_to_index[movie_id]
        target = embeddings[target_index]
        scores = torch.nn.functional.cosine_similarity(target.unsqueeze(0), embeddings).numpy()

    best = np.argsort(-scores)
    output: list[tuple[int, float]] = []
    for index in best:
        candidate = result.index_to_movie[int(index)]
        if candidate == movie_id:
            continue
        output.append((candidate, float(scores[index])))
        if len(output) >= top_k:
            break
    return output
