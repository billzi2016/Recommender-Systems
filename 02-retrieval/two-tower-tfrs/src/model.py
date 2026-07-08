from __future__ import annotations

"""PyTorch 双塔召回模型。

模型意图：
把用户和电影分别编码成同一维度的向量。训练时让一个 batch 内
真实匹配的 user/movie 对角线分数更高，其他电影自然成为批内负样本。

这个实现尽量调用 PyTorch 现成能力，不造复杂训练框架。
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from utils.movielens import make_id_maps
from utils.torch_utils import dataloader_kwargs, get_device, seed_everything


class TwoTower(nn.Module):
    """最小可用双塔模型。

    第一版只使用 userId 和 movieId embedding。
    这样读者能先看懂召回的核心，再考虑加入 genres 或用户历史特征。
    """

    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """输出归一化用户向量，方便用点积当余弦相似度。"""

        return nn.functional.normalize(self.user_embedding(user_ids), dim=1)

    def encode_movies(self, movie_ids: torch.Tensor) -> torch.Tensor:
        """输出归一化电影向量。"""

        return nn.functional.normalize(self.movie_embedding(movie_ids), dim=1)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """返回 batch 内 user 和 movie 的两两相似度矩阵。"""

        user_vec = self.encode_users(user_ids)
        movie_vec = self.encode_movies(movie_ids)
        return user_vec @ movie_vec.T


@dataclass
class TwoTowerResult:
    """双塔训练结果。"""

    model: TwoTower
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_user: dict[int, int]
    index_to_movie: dict[int, int]
    device_name: str
    best_valid_loss: float


def build_positive_interactions(ratings: pd.DataFrame, positive_threshold: float = 4.0) -> pd.DataFrame:
    """把高评分记录转成双塔训练用正样本。

    MovieLens 没有曝光日志，所以这里不把低评分或未评分强行当成真负样本。
    批内其他电影只作为训练近似负样本。
    """

    positives = ratings[ratings["rating"] >= positive_threshold][["userId", "movieId", "timestamp"]].copy()
    return positives.sort_values(["userId", "timestamp"]).reset_index(drop=True)


def _dataset(interactions: pd.DataFrame, user_to_index: dict[int, int], movie_to_index: dict[int, int]) -> TensorDataset:
    users = interactions["userId"].map(user_to_index).to_numpy()
    movies = interactions["movieId"].map(movie_to_index).to_numpy()
    return TensorDataset(torch.tensor(users, dtype=torch.long), torch.tensor(movies, dtype=torch.long))


def train_two_tower(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    embedding_dim: int = 64,
    max_epochs: int = 1000,
    patience: int = 5,
    batch_size: int = 2048,
    num_workers: int = 8,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    keep_checkpoints: int = 3,
) -> TwoTowerResult:
    """训练双塔模型，并用验证集 loss 做 early stopping。

    保存策略和矩阵分解保持一致：
    - 默认可以只保存训练结束后的 best.pt。
    - 中间 checkpoint 只有 `checkpoint_every > 0` 时才写。
    - 最多保留 `keep_checkpoints` 个中间文件。
    """

    seed_everything(42)
    train_pos = build_positive_interactions(train)
    valid_pos = build_positive_interactions(valid)
    user_to_index, index_to_user = make_id_maps(train_pos["userId"])
    movie_to_index, index_to_movie = make_id_maps(train_pos["movieId"])

    valid_known = valid_pos[valid_pos["userId"].isin(user_to_index) & valid_pos["movieId"].isin(movie_to_index)].copy()
    device = get_device()
    model = TwoTower(len(user_to_index), len(movie_to_index), embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        _dataset(train_pos, user_to_index, movie_to_index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs(device, num_workers),
    )
    valid_loader = DataLoader(_dataset(valid_known, user_to_index, movie_to_index), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    best_valid_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    saved_intermediates: list[Path] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for users, movies in tqdm(train_loader, desc=f"TwoTower epoch {epoch + 1}/{max_epochs}", unit="batch"):
            users = users.to(device)
            movies = movies.to(device)
            labels = torch.arange(users.shape[0], device=device)
            optimizer.zero_grad()
            logits = model(users, movies)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        valid_loss = evaluate_loss(model, valid_loader, device)
        print(
            f"[TwoTower] epoch={epoch + 1} "
            f"train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"valid_loss={valid_loss:.4f}"
        )

        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[TwoTower] early stopping: validation loss has not improved for {patience} epochs.")
                break

        if checkpoint_dir is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "valid_loss": valid_loss,
                },
                checkpoint_path,
            )
            saved_intermediates.append(checkpoint_path)
            while len(saved_intermediates) > keep_checkpoints:
                saved_intermediates.pop(0).unlink(missing_ok=True)

    model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "best_valid_loss": best_valid_loss,
                "embedding_dim": embedding_dim,
            },
            checkpoint_dir / "best.pt",
        )

    return TwoTowerResult(model, user_to_index, movie_to_index, index_to_user, index_to_movie, str(device), best_valid_loss)


def evaluate_loss(model: TwoTower, loader: DataLoader, device: torch.device) -> float:
    """计算验证集 batch 内负样本 loss。"""

    if len(loader) == 0:
        return 0.0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for users, movies in loader:
            users = users.to(device)
            movies = movies.to(device)
            labels = torch.arange(users.shape[0], device=device)
            total += float(loss_fn(model(users, movies), labels).detach().cpu())
    return total / len(loader)


def recommend_for_users(result: TwoTowerResult, train: pd.DataFrame, user_ids: list[int], top_k: int = 10) -> dict[int, list[int]]:
    """为一批用户做向量检索。

    这里直接用矩阵乘法检索全部电影。MovieLens 电影数量不大，足够清楚。
    更大规模系统再换 FAISS 或其他 ANN 索引。
    """

    model = result.model
    device = next(model.parameters()).device
    model.eval()
    all_movie_indices = torch.arange(len(result.index_to_movie), dtype=torch.long, device=device)
    seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
    output: dict[int, list[int]] = {}

    with torch.no_grad():
        movie_vectors = model.encode_movies(all_movie_indices)
        for user_id in tqdm(user_ids, desc="Retrieving candidates", unit="user"):
            if user_id not in result.user_to_index:
                continue
            user_index = torch.tensor([result.user_to_index[user_id]], dtype=torch.long, device=device)
            scores = (model.encode_users(user_index) @ movie_vectors.T).squeeze(0).detach().cpu().numpy()
            ranked = np.argsort(-scores)
            seen = seen_by_user.get(user_id, set())
            recs: list[int] = []
            for movie_index in ranked:
                movie_id = result.index_to_movie[int(movie_index)]
                if movie_id in seen:
                    continue
                recs.append(movie_id)
                if len(recs) >= top_k:
                    break
            output[user_id] = recs
    return output
