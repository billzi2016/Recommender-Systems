from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


@dataclass
class ItemCFModel:
    """Item-CF 训练后的必要对象。

    user_to_row:
        MovieLens userId 到稀疏矩阵行号的映射。
    movie_to_col:
        MovieLens movieId 到稀疏矩阵列号的映射。
    col_to_movie:
        稀疏矩阵列号到原始 movieId 的反向映射，用于把推荐结果转回电影 ID。
    matrix:
        movie-user 稀疏矩阵。每一行是一部电影，每一列是一个用户。
    neighbors:
        sklearn 的近邻模型，负责按余弦距离找相似电影。
    """

    user_to_row: dict[int, int]
    movie_to_col: dict[int, int]
    col_to_movie: dict[int, int]
    matrix: csr_matrix
    neighbors: NearestNeighbors


def fit_item_cf(ratings: pd.DataFrame, positive_threshold: float = 4.0) -> ItemCFModel:
    """从 MovieLens 正反馈评分中训练 Item-CF。

    Item-CF 的核心是“电影之间相似不相似”。
    这里把评分大于等于 `positive_threshold` 的记录当作喜欢，
    然后构建 movie-user 矩阵，最后用 sklearn 的 NearestNeighbors
    计算电影之间的余弦近邻。
    """

    positives = ratings[ratings["rating"] >= positive_threshold].copy()
    # 重新编号是为了构建稀疏矩阵。原始 ID 不一定连续，不能直接当行列号。
    users = sorted(positives["userId"].unique())
    movies = sorted(positives["movieId"].unique())
    user_to_row = {int(user_id): row for row, user_id in enumerate(users)}
    movie_to_col = {int(movie_id): col for col, movie_id in enumerate(movies)}
    col_to_movie = {col: movie_id for movie_id, col in movie_to_col.items()}

    rows = positives["userId"].map(user_to_row).to_numpy()
    cols = positives["movieId"].map(movie_to_col).to_numpy()
    values = [1.0] * len(positives)

    user_item = csr_matrix((values, (rows, cols)), shape=(len(users), len(movies)))
    item_user = user_item.T.tocsr()

    # sklearn 已经实现了稀疏余弦近邻搜索，这里直接调包，不自己造轮子。
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(item_user)
    return ItemCFModel(user_to_row, movie_to_col, col_to_movie, item_user, neighbors)


def recommend_for_user(model: ItemCFModel, ratings: pd.DataFrame, user_id: int, top_k: int = 10, neighbors_k: int = 30) -> list[tuple[int, float]]:
    """给单个用户生成 Item-CF 推荐。

    逻辑：
    1. 找到用户喜欢过的电影。
    2. 对每部喜欢过的电影，查它的相似电影。
    3. 把相似度累加到候选电影上。
    4. 过滤用户已经评分过的电影。
    5. 返回分数最高的 top_k 个候选。
    """

    liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4.0)]["movieId"].tolist()
    seen = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
    scores: dict[int, float] = {}

    for movie_id in tqdm(liked, desc=f"Scoring candidates for user {user_id}", leave=False):
        if movie_id not in model.movie_to_col:
            continue
        col = model.movie_to_col[movie_id]
        distances, indices = model.neighbors.kneighbors(model.matrix[col], n_neighbors=min(neighbors_k + 1, model.matrix.shape[0]))
        for distance, index in zip(distances[0], indices[0], strict=False):
            candidate = model.col_to_movie[int(index)]
            if candidate == movie_id or candidate in seen:
                continue
            # sklearn 返回的是 cosine distance，Item-CF 推荐分数更适合用 similarity。
            similarity = 1.0 - float(distance)
            scores[candidate] = scores.get(candidate, 0.0) + similarity

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
