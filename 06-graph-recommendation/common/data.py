from __future__ import annotations

"""MovieLens 图推荐数据准备。

图推荐把 MovieLens 看成一张用户-电影二部图：
用户是一类节点，电影是一类节点，用户高评分电影之间连边。

这个文件负责把 ratings.csv 转成三类对象：
1. 连续的 user/movie index 映射，供 embedding 查表。
2. 归一化二部图邻接矩阵，供 LightGCN/NGCF 做消息传递。
3. BPR 训练样本 `(user, positive_movie, negative_movie)`。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, diags
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class GraphSpec:
    """图推荐实验需要的 ID 映射和图规模。"""

    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: dict[int, int]
    num_users: int
    num_movies: int


class BPRDataset(Dataset):
    """BPR 训练数据集。

    每条样本包含：
    - user：目标用户。
    - positive：用户真实高评分过的电影。
    - negative：用户没有高评分过的随机电影。

    negative 不等于“用户讨厌”，它只是未交互采样项。
    """

    def __init__(self, user_indices: np.ndarray, positive_indices: np.ndarray, user_positives: dict[int, set[int]], num_movies: int) -> None:
        self.user_indices = user_indices.astype(np.int64, copy=False)
        self.positive_indices = positive_indices.astype(np.int64, copy=False)
        self.user_positives = user_positives
        self.num_movies = num_movies
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.positive_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回一条 BPR 三元组。"""

        user = int(self.user_indices[index])
        positive = int(self.positive_indices[index])
        positives = self.user_positives[user]
        negative = int(self.rng.integers(0, self.num_movies))
        # 负样本如果刚好采到用户正反馈电影，就继续采。
        # MovieLens 电影很多，通常很快能采到未交互项。
        while negative in positives:
            negative = int(self.rng.integers(0, self.num_movies))
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negative, dtype=torch.long),
        )


def build_positive_edges(ratings: pd.DataFrame, positive_threshold: float = 4.0, max_positives_per_user: int = 50) -> pd.DataFrame:
    """从评分表里抽取用户-电影正反馈边。

    默认每个用户最多保留最近 50 条高评分边。
    这不是伪造数据，而是为了让本地全量 MovieLens 图训练可控：
    图传播每轮都要处理边，极长历史用户会明显拖慢训练。
    如果想用全量正反馈边，传 `--max-positives-per-user 0`。
    """

    positives = ratings[ratings["rating"] >= positive_threshold][["userId", "movieId", "timestamp"]].copy()
    positives = positives.sort_values(["userId", "timestamp"])
    if max_positives_per_user > 0:
        positives = positives.groupby("userId", group_keys=False).tail(max_positives_per_user)
    return positives.reset_index(drop=True)


def build_graph_spec(edges: pd.DataFrame) -> GraphSpec:
    """基于正反馈边建立连续 user/movie index。"""

    user_ids = sorted(edges["userId"].unique())
    movie_ids = sorted(edges["movieId"].unique())
    user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
    return GraphSpec(user_to_index, movie_to_index, index_to_movie, len(user_to_index), len(movie_to_index))


def encode_edges(edges: pd.DataFrame, spec: GraphSpec) -> tuple[np.ndarray, np.ndarray, dict[int, set[int]]]:
    """把原始边转成 user/movie index，并记录每个用户的正样本集合。"""

    user_indices = edges["userId"].map(spec.user_to_index).to_numpy(dtype=np.int64)
    movie_indices = edges["movieId"].map(spec.movie_to_index).to_numpy(dtype=np.int64)
    user_positives: dict[int, set[int]] = {}
    for user, movie in zip(user_indices, movie_indices, strict=False):
        user_positives.setdefault(int(user), set()).add(int(movie))
    return user_indices, movie_indices, user_positives


def build_normalized_adjacency(spec: GraphSpec, user_indices: np.ndarray, movie_indices: np.ndarray) -> torch.Tensor:
    """构建对称归一化二部图邻接矩阵。

    节点排列方式：
    - 前 `num_users` 个节点是用户。
    - 后 `num_movies` 个节点是电影。

    LightGCN/NGCF 的传播核心是 `A_norm @ embeddings`。
    这里用 scipy 先构建稀疏矩阵，再转成 PyTorch sparse COO tensor。
    """

    num_nodes = spec.num_users + spec.num_movies
    movie_nodes = movie_indices + spec.num_users
    rows = np.concatenate([user_indices, movie_nodes])
    cols = np.concatenate([movie_nodes, user_indices])
    values = np.ones(len(rows), dtype=np.float32)
    adjacency = coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree_inv_sqrt = np.power(np.maximum(degree, 1.0), -0.5)
    normalized = diags(degree_inv_sqrt) @ adjacency @ diags(degree_inv_sqrt)
    normalized = normalized.tocoo()
    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    vals = torch.tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, size=normalized.shape).coalesce()


def make_bpr_dataset(edges: pd.DataFrame, spec: GraphSpec) -> tuple[BPRDataset, dict[int, set[int]], np.ndarray, np.ndarray]:
    """根据正反馈边构建 BPRDataset。"""

    user_indices, movie_indices, user_positives = encode_edges(edges, spec)
    dataset = BPRDataset(user_indices, movie_indices, user_positives, spec.num_movies)
    return dataset, user_positives, user_indices, movie_indices


def split_edges_by_user_time(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按用户时间切分图边。

    每个用户最后一条正反馈作为测试边，倒数第二条作为验证边，其余作为训练边。
    少于 3 条正反馈的用户只进入训练集，避免构造没有历史的评估样本。
    """

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in tqdm(edges.groupby("userId"), desc="Split graph edges by user", unit="user"):
        group = group.sort_values("timestamp")
        if len(group) < 3:
            train_parts.append(group)
            continue
        train_parts.append(group.iloc[:-2])
        valid_parts.append(group.iloc[-2:-1])
        test_parts.append(group.iloc[-1:])
    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True) if valid_parts else edges.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else edges.iloc[0:0].copy()
    return train, valid, test

