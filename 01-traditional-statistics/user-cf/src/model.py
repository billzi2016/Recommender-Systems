from __future__ import annotations

"""User-CF 的模型实现。

实现意图：
1. 用 MovieLens 评分构建 user-item 稀疏矩阵。
2. 调用 sklearn NearestNeighbors 计算相似用户。
3. 汇总相似用户喜欢、但目标用户没看过的电影。

这里不手写相似度搜索，直接使用 sklearn，避免重复造轮子。
"""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


@dataclass
class UserCFModel:
    """User-CF 训练后的必要对象。

    user_to_row:
        MovieLens 原始 userId 到稀疏矩阵行号的映射。
    movie_to_col:
        MovieLens 原始 movieId 到稀疏矩阵列号的映射。
    col_to_movie:
        反向映射，用于把推荐结果从矩阵列号转回真实 movieId。
    matrix:
        user-item 稀疏矩阵。每一行是一个用户，每一列是一部电影。
    neighbors:
        sklearn 近邻模型，负责找相似用户。
    """

    user_to_row: dict[int, int]
    movie_to_col: dict[int, int]
    col_to_movie: dict[int, int]
    matrix: csr_matrix
    neighbors: NearestNeighbors


def fit_user_cf(ratings: pd.DataFrame, positive_threshold: float = 4.0) -> UserCFModel:
    """训练用户-用户近邻模型。

    这里把高评分当成正反馈，构建 user-item 二值矩阵。
    每一行是一个用户，每一列是一部电影。
    """

    # User-CF 第一版只看正反馈。
    # 用二值矩阵可以让“相似用户”更容易解释：他们喜欢过相似的电影集合。
    positives = ratings[ratings["rating"] >= positive_threshold].copy()
    users = sorted(positives["userId"].unique())
    movies = sorted(positives["movieId"].unique())
    user_to_row = {int(user_id): row for row, user_id in enumerate(users)}
    movie_to_col = {int(movie_id): col for col, movie_id in enumerate(movies)}
    col_to_movie = {col: movie_id for movie_id, col in movie_to_col.items()}

    # 原始 ID 不能直接作为稀疏矩阵下标，所以先映射成连续整数。
    rows = positives["userId"].map(user_to_row).to_numpy()
    cols = positives["movieId"].map(movie_to_col).to_numpy()
    values = [1.0] * len(positives)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(movies)))

    # 用户数量虽然不少，但 MovieLens 本地实验用 brute force 余弦近邻足够清楚。
    # 真正超大规模时可以换 ANN 索引，但那不是第一版重点。
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    return UserCFModel(user_to_row, movie_to_col, col_to_movie, matrix, neighbors)


def recommend_for_user(model: UserCFModel, ratings: pd.DataFrame, user_id: int, top_k: int = 10, neighbors_k: int = 40) -> list[tuple[int, float]]:
    """为单个用户生成 User-CF 推荐。

    推荐逻辑：
    1. 找目标用户的相似用户。
    2. 看这些邻居喜欢过哪些电影。
    3. 用“邻居相似度”给候选电影加权。
    4. 过滤目标用户已经评分过的电影。
    """

    if user_id not in model.user_to_row:
        return []

    # seen 用来过滤已经评分过的电影。
    # 推荐系统通常不应该把用户已经明确看过/评过的物品再次作为新推荐。
    seen = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
    row = model.user_to_row[user_id]
    distances, indices = model.neighbors.kneighbors(model.matrix[row], n_neighbors=min(neighbors_k + 1, model.matrix.shape[0]))

    scores: dict[int, float] = {}
    for distance, neighbor_row in tqdm(list(zip(distances[0], indices[0], strict=False)), desc=f"Scoring neighbors for user {user_id}", leave=False):
        if int(neighbor_row) == row:
            continue
        similarity = 1.0 - float(distance)
        # 邻居喜欢过的电影都会给候选加分。
        # 越相似的邻居，贡献越大；多个邻居都喜欢同一部电影，分数会累加。
        liked_cols = model.matrix[int(neighbor_row)].indices
        for col in liked_cols:
            movie_id = model.col_to_movie[int(col)]
            if movie_id in seen:
                continue
            scores[movie_id] = scores.get(movie_id, 0.0) + similarity

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
