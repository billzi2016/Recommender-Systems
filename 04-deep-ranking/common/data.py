from __future__ import annotations

"""深度精排实验的数据准备。

04 组模型的目标不是从全库召回候选，而是给用户-电影组合打更细的分。
这里统一把 MovieLens 评分转成二分类标签：`rating >= 4.0` 表示喜欢。

注意：这里没有把“用户没评分过的电影”全部当负样本。
MovieLens 只有评分日志，没有曝光日志。用户没评分一部电影，可能是不喜欢，
也可能只是没见过。所以这一组实验先用已有评分做二分类精排，避免把数据语义讲歪。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

GENRE_SEPARATOR = "|"
NO_GENRE = "(no genres listed)"
HOUR_BUCKETS = 4


@dataclass(frozen=True)
class RankingSpec:
    """记录 ID 映射和稀疏特征空间大小。

    PyTorch 的 embedding 层需要从 0 开始的连续整数下标。
    MovieLens 原始 userId/movieId 虽然看起来是数字，但不能直接拿来当 embedding 下标：
    - 原始 ID 不保证连续。
    - ID 最大值可能远大于真实数量，直接用会浪费 embedding 表。
    - 训练集没见过的 ID 没有向量，验证/测试时必须过滤。
    """

    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: dict[int, int]
    genre_to_index: dict[str, int]
    movie_genres: dict[int, tuple[int, ...]]
    max_genres: int

    @property
    def num_users(self) -> int:
        return len(self.user_to_index)

    @property
    def num_movies(self) -> int:
        return len(self.movie_to_index)

    @property
    def num_genres(self) -> int:
        return len(self.genre_to_index)


class PairRankingDataset(Dataset):
    """NCF 使用的用户-电影二分类数据集。

    NCF 这一版只关心两个字段：用户是谁、电影是谁。
    它故意不加 genres 和时间段，是为了让它和矩阵分解的差异更清楚：
    同样只看 ID，点积换成 MLP 后会发生什么。
    """

    def __init__(self, users: np.ndarray, movies: np.ndarray, labels: np.ndarray) -> None:
        self.users = users.astype(np.int64, copy=False)
        self.movies = movies.astype(np.int64, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回一条 `(user_index, movie_index, label)`。

        DataLoader 会把很多条样本自动拼成 batch。
        这里每次只构造单条 tensor，避免提前生成一个很大的二维 tensor 占内存。
        """

        return (
            torch.tensor(self.users[index], dtype=torch.long),
            torch.tensor(self.movies[index], dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


class ContextRankingDataset(Dataset):
    """Wide&Deep 和 DCN 使用的上下文精排数据集。

    这两个模型除了 user/movie ID，还会看 genre 和时间段。
    genre 是多值特征，一部电影可能同时属于 `Action|Sci-Fi`。
    为了让 batch 里的每条样本长度一致，需要把 genre 列表补齐到 `max_genres`。
    padding 使用 0，embedding 里也把 0 设成 padding_idx。
    """

    def __init__(
        self,
        users: np.ndarray,
        movies: np.ndarray,
        genres: np.ndarray,
        hours: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.users = users.astype(np.int64, copy=False)
        self.movies = movies.astype(np.int64, copy=False)
        self.genres = genres.astype(np.int64, copy=False)
        self.hours = hours.astype(np.int64, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回一条包含 ID、genre、时间段和标签的样本。"""

        return {
            "user": torch.tensor(self.users[index], dtype=torch.long),
            "movie": torch.tensor(self.movies[index], dtype=torch.long),
            "genres": torch.tensor(self.genres[index], dtype=torch.long),
            "hour": torch.tensor(self.hours[index], dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def _split_genres(value: str) -> list[str]:
    """把 `Action|Sci-Fi` 拆成单独 genre。

    MovieLens 用竖线把多值 genre 放在一个字符串里。
    模型不能直接理解这个字符串，所以要拆成多个离散特征。
    """

    return [genre for genre in str(value).split(GENRE_SEPARATOR) if genre and genre != NO_GENRE]


def build_ranking_spec(train: pd.DataFrame, movies: pd.DataFrame) -> RankingSpec:
    """基于训练集建立连续 ID 和 genre 映射。

    这里刻意只用训练侧出现过的用户和电影建表。
    如果用全量数据先建表，再切训练/验证，就会让模型提前知道测试集中有哪些 ID，
    这虽然不是直接泄露评分，但会让实验边界变模糊。
    """

    user_ids = sorted(train["userId"].unique())
    movie_ids = sorted(train["movieId"].unique())
    movie_rows = movies[movies["movieId"].isin(movie_ids)].copy()
    genre_names = sorted({genre for value in movie_rows["genres"] for genre in _split_genres(value)})

    user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    index_to_movie = {index: int(movie_id) for movie_id, index in movie_to_index.items()}
    # genre 下标从 1 开始，0 留给 padding。
    # 这样电影 genre 数量不一致时，可以把短列表补 0，而 0 不参与有效语义。
    genre_to_index = {genre: index + 1 for index, genre in enumerate(genre_names)}

    movie_genres: dict[int, tuple[int, ...]] = {}
    for row in movie_rows[["movieId", "genres"]].itertuples(index=False):
        movie_index = movie_to_index[int(row.movieId)]
        genre_indices = tuple(genre_to_index[genre] for genre in _split_genres(row.genres))
        movie_genres[movie_index] = genre_indices

    max_genres = max((len(values) for values in movie_genres.values()), default=1)
    return RankingSpec(user_to_index, movie_to_index, index_to_movie, genre_to_index, movie_genres, max_genres)


def _known_frame(ratings: pd.DataFrame, spec: RankingSpec) -> pd.DataFrame:
    """过滤训练集中没见过的用户或电影。

    这是最基础的冷启动处理。
    embedding 模型没法给训练中从未出现的用户/电影查向量。
    真正工业系统会有冷启动策略，这里先过滤，保证实验重点放在模型结构。
    """

    return ratings[ratings["userId"].isin(spec.user_to_index) & ratings["movieId"].isin(spec.movie_to_index)].copy()


def build_pair_dataset(ratings: pd.DataFrame, spec: RankingSpec) -> PairRankingDataset:
    """构建 NCF 的用户-电影数据集。

    标签含义：
    - `1`：rating >= 4.0，近似表示用户喜欢。
    - `0`：rating < 4.0，表示这次评分没有达到喜欢阈值。
    """

    frame = _known_frame(ratings, spec)
    users = frame["userId"].map(spec.user_to_index).to_numpy(dtype=np.int64)
    movies = frame["movieId"].map(spec.movie_to_index).to_numpy(dtype=np.int64)
    labels = (frame["rating"].to_numpy(dtype=np.float32) >= 4.0).astype(np.float32)
    return PairRankingDataset(users, movies, labels)


def build_context_dataset(ratings: pd.DataFrame, spec: RankingSpec) -> ContextRankingDataset:
    """构建带上下文特征的精排数据集。

    这里的“上下文”很克制，只放 MovieLens 自带的字段：
    - movie genres：电影内容侧信息。
    - timestamp hour bucket：评分发生时间的大致时段。

    timestamp 不是观看时间，只能当弱上下文使用。报告里不要把它解释成真实观影时段。
    """

    frame = _known_frame(ratings, spec)
    users = frame["userId"].map(spec.user_to_index).to_numpy(dtype=np.int64)
    movie_indices = frame["movieId"].map(spec.movie_to_index).to_numpy(dtype=np.int64)
    # 每部电影的 genre 数量不同。这里用一个二维矩阵承载：
    # 行是评分样本，列是这部电影的第几个 genre。空位补 0。
    genre_matrix = np.zeros((len(frame), spec.max_genres), dtype=np.int64)
    for row_index, movie_index in enumerate(movie_indices):
        values = spec.movie_genres.get(int(movie_index), ())
        genre_matrix[row_index, : len(values)] = values
    hours = pd.to_datetime(frame["timestamp"], unit="s").dt.hour.to_numpy(dtype=np.int64)
    # 24 小时粗分成 4 段，避免把时间特征切得太碎。
    # 对 MovieLens 来说，时间戳噪声很大，粗一点反而更稳。
    hour_buckets = np.minimum(hours // 6, HOUR_BUCKETS - 1)
    labels = (frame["rating"].to_numpy(dtype=np.float32) >= 4.0).astype(np.float32)
    return ContextRankingDataset(users, movie_indices, genre_matrix, hour_buckets, labels)
