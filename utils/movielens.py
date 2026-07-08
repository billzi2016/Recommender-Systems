from __future__ import annotations

"""MovieLens 数据读取和通用预处理。

所有实验都走这个文件读取数据，避免每个算法目录重复写：
解压、读 CSV、固定抽样、ID 映射、按时间切分。

这样做的好处是实验之间更可比：
如果切分逻辑分散在各处，很容易出现某个算法用随机切分、另一个算法用时间切分，
最后指标就没有可比性。
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .paths import raw_dir


@dataclass(frozen=True)
class MovieLensData:
    """MovieLens 的两个核心表。

    ratings:
        用户评分表，包含 userId、movieId、rating、timestamp。
    movies:
        电影信息表，包含 movieId、title、genres。
    """

    ratings: pd.DataFrame
    movies: pd.DataFrame


def ensure_ml32m_unzipped() -> Path:
    """确保 MovieLens 32M 已经解压，并返回解压目录。

    仓库里本地保留官方 zip，但 zip 不进 git。
    每个算法实验不应该重复写“检查 zip 是否存在、是否解压”的逻辑，
    所以这里统一处理。

    如果已经有 `raw/ml-32m/ratings.csv`，直接返回。
    如果没有，就从 `raw/ml-32m.zip` 解压。
    """

    target = raw_dir() / "ml-32m"
    ratings_csv = target / "ratings.csv"
    if ratings_csv.exists():
        return target

    archive = raw_dir() / "ml-32m.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing {archive}. Run dataset-movie_lens/scripts/download_ml_32m.sh first."
        )

    print(f"[data] 解压 MovieLens 32M: {archive}")
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Unzipping MovieLens", unit="file"):
            zf.extract(member, raw_dir())
    return target


def load_movielens(sample_ratings: int | None = None, min_rating: float = 0.0) -> MovieLensData:
    """读取 MovieLens 的 ratings 和 movies。

    参数说明：
    - `sample_ratings`：抽样评分条数。默认 `None`，表示使用全量数据。
      如果只是想快速检查流程，可以传较小的整数。
    - `min_rating`：最低评分过滤阈值。例如做隐式反馈时，可以只保留
      rating >= 4.0 的记录。

    这里的抽样使用固定 random_state，保证每次运行结果可复现。
    """

    data_dir = ensure_ml32m_unzipped()
    print("[data] 读取 ratings.csv 和 movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")

    if min_rating > 0:
        ratings = ratings[ratings["rating"] >= min_rating].copy()

    if sample_ratings is not None and len(ratings) > sample_ratings:
        # 固定 random_state，保证同一个 sample-ratings 参数每次拿到同一批评分。
        # 这对调参和对比算法很重要。
        ratings = ratings.sample(n=sample_ratings, random_state=42).copy()

    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    return MovieLensData(ratings=ratings, movies=movies)


def make_id_maps(values: pd.Series) -> tuple[dict[int, int], dict[int, int]]:
    """把 MovieLens 原始 ID 映射成从 0 开始的连续编号。

    很多模型，尤其是 PyTorch embedding，需要连续整数下标。
    原始 userId/movieId 不一定连续，所以统一用这个函数做映射。
    """

    # 排序后再编号，保证同一份输入每次得到一致映射。
    unique_values = sorted(values.unique())
    to_index = {int(value): index for index, value in enumerate(unique_values)}
    to_value = {index: int(value) for value, index in to_index.items()}
    return to_index, to_value


def attach_titles(recommendations: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """给推荐结果补上电影标题和 genres，方便报告里人工检查。"""

    return recommendations.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")


def train_test_by_user_time(ratings: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按用户内部时间顺序切分训练集和测试集。

    推荐系统里随机切分容易泄露未来信息。
    比如一个用户 2020 年才开始喜欢动画片，如果随机切分，
    模型可能在训练集中看到 2020 年行为，却去预测 2010 年行为。

    这里对每个用户单独按 timestamp 排序：
    - 前一段进入 train。
    - 后一段进入 test。
    - 只有一条记录的用户留在 train，因为它没法提供有意义的未来测试样本。
    """

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in tqdm(ratings.groupby("userId"), desc="Time split by user", unit="user"):
        group = group.sort_values("timestamp")
        if len(group) < 2:
            # 只有一条记录的用户无法切出“未来行为”测试集。
            # 这类用户保留在训练集，避免无意义地制造空测试样本。
            train_parts.append(group)
            continue
        cut = max(1, int(len(group) * (1 - test_ratio)))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test
