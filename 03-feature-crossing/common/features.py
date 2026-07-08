from __future__ import annotations

"""MovieLens 特征交叉实验的数据准备。

FM、DeepFM、xDeepFM 都不是只看 `(userId, movieId)` 的模型。
它们真正想展示的是：把用户、电影、电影类型、时间段这些离散特征放在一起，
让模型自己学习“用户 42 + 科幻片 + 晚上”这种组合是否更容易得到高评分。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

GENRE_SEPARATOR = "|"
NO_GENRE = "(no genres listed)"
HOUR_BUCKET_NAMES = ["late_night", "morning", "afternoon", "evening"]


@dataclass(frozen=True)
class FeatureSpec:
    """记录稀疏特征空间的编号方式。

    `0` 专门留给 padding。真实特征从 `1` 开始编号，这样每条样本可以补齐到
    同样长度，方便 PyTorch DataLoader 拼 batch。
    """

    num_features: int
    max_fields: int
    genre_names: list[str]


class FeatureCrossingDataset(Dataset):
    """按需组装一条 MovieLens 样本。

    为了避免全量 MovieLens 32M 时生成巨大的二维特征矩阵，这里只保存几列
    numpy 数组。真正喂给模型的定长特征向量在 `__getitem__` 里临时组装。
    """

    def __init__(
        self,
        user_features: np.ndarray,
        movie_features: np.ndarray,
        hour_features: np.ndarray,
        movie_genre_features: dict[int, tuple[int, ...]],
        labels: np.ndarray,
        max_fields: int,
    ) -> None:
        self.user_features = user_features.astype(np.int64, copy=False)
        self.movie_features = movie_features.astype(np.int64, copy=False)
        self.hour_features = hour_features.astype(np.int64, copy=False)
        self.movie_genre_features = movie_genre_features
        self.labels = labels.astype(np.float32, copy=False)
        self.max_fields = max_fields

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回一条定长稀疏特征和二分类标签。

        FM 系列模型吃的是“一组特征编号”，不是已经 one-hot 展开的巨大向量。
        embedding 层会根据这些编号查向量，所以这里返回的是 feature id 列表。
        """

        # 用 0 初始化，代表 padding。真实特征编号都从 1 开始。
        features = np.zeros(self.max_fields, dtype=np.int64)
        movie_feature = int(self.movie_features[index])
        values = [
            int(self.user_features[index]),
            movie_feature,
            *self.movie_genre_features.get(movie_feature, ()),
            int(self.hour_features[index]),
        ]
        # 正常情况下不会超长；这里保留 clipped 是防御式写法，
        # 避免某些异常 genre 数据让样本长度超过固定字段数。
        clipped = values[: self.max_fields]
        features[: len(clipped)] = clipped
        return torch.from_numpy(features), torch.tensor(self.labels[index], dtype=torch.float32)


def _split_genres(value: str) -> list[str]:
    """把 MovieLens 的 `Action|Sci-Fi` 拆成干净的 genre 列表。"""

    genres = [genre for genre in str(value).split(GENRE_SEPARATOR) if genre and genre != NO_GENRE]
    return genres


def build_feature_spec(ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[FeatureSpec, dict[int, int], dict[int, int], dict[int, tuple[int, ...]]]:
    """建立全局稀疏特征编号。

    特征编号按块排列：
    - 用户 ID 特征。
    - 电影 ID 特征。
    - 电影 genre 特征。
    - 时间段特征。
    """

    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(movies["movieId"].unique())
    genre_names = sorted({genre for value in movies["genres"] for genre in _split_genres(value)})

    # 0 留给 padding，所以真实特征从 1 开始。
    next_feature = 1
    user_to_feature = {int(user_id): next_feature + index for index, user_id in enumerate(user_ids)}
    next_feature += len(user_to_feature)
    movie_to_feature = {int(movie_id): next_feature + index for index, movie_id in enumerate(movie_ids)}
    next_feature += len(movie_to_feature)
    genre_to_feature = {genre: next_feature + index for index, genre in enumerate(genre_names)}
    next_feature += len(genre_to_feature)
    hour_features = [next_feature + index for index in range(len(HOUR_BUCKET_NAMES))]
    next_feature += len(hour_features)

    movie_genre_features: dict[int, tuple[int, ...]] = {}
    for row in movies[["movieId", "genres"]].itertuples(index=False):
        movie_feature = movie_to_feature[int(row.movieId)]
        genre_features = tuple(genre_to_feature[genre] for genre in _split_genres(row.genres))
        movie_genre_features[movie_feature] = genre_features

    # max_fields 决定每条样本的固定长度：
    # user + movie + 最多若干 genre + hour bucket。
    max_genres = max((len(values) for values in movie_genre_features.values()), default=0)
    spec = FeatureSpec(
        num_features=next_feature - 1,
        max_fields=1 + 1 + max_genres + 1,
        genre_names=genre_names,
    )
    return spec, user_to_feature, movie_to_feature, movie_genre_features


def build_dataset(
    ratings: pd.DataFrame,
    user_to_feature: dict[int, int],
    movie_to_feature: dict[int, int],
    movie_genre_features: dict[int, tuple[int, ...]],
    spec: FeatureSpec,
) -> FeatureCrossingDataset:
    """把评分表转换成模型可读的数据集。

    标签使用 `rating >= 4.0`。这样 FM 系列模型学的是“这条用户-电影-上下文
    组合是否像一次喜欢”，比直接回归 1-5 分更贴近推荐系统里的排序任务。
    """

    # 验证/测试里可能有训练没见过的用户或电影。
    # embedding 模型无法处理这些冷启动 ID，第一版直接过滤。
    frame = ratings[ratings["userId"].isin(user_to_feature) & ratings["movieId"].isin(movie_to_feature)].copy()
    user_features = frame["userId"].map(user_to_feature).to_numpy(dtype=np.int64)
    movie_features = frame["movieId"].map(movie_to_feature).to_numpy(dtype=np.int64)
    hours = pd.to_datetime(frame["timestamp"], unit="s").dt.hour.to_numpy(dtype=np.int64)
    hour_bucket = np.minimum(hours // 6, len(HOUR_BUCKET_NAMES) - 1)
    # hour 特征在 build_feature_spec 里是最后一块编号。
    # 这里用 offset 把 0/1/2/3 的时间桶转回全局 feature id。
    hour_offset = spec.num_features - len(HOUR_BUCKET_NAMES) + 1
    hour_features = hour_offset + hour_bucket
    labels = (frame["rating"].to_numpy(dtype=np.float32) >= 4.0).astype(np.float32)
    return FeatureCrossingDataset(user_features, movie_features, hour_features, movie_genre_features, labels, spec.max_fields)
