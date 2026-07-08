from __future__ import annotations

"""MovieLens 序列推荐数据准备。

序列推荐和前面几组实验最大的区别是：样本不再只是一个用户-电影二元组，
而是一段按时间排序的电影历史。模型要做的是 next item prediction：
给定前面的电影序列，预测下一部更可能出现的电影。

这里统一做四件事：
1. 只保留高评分电影，避免把明显不喜欢的电影放进“正向兴趣序列”。
2. 按用户内部 timestamp 排序，保证序列顺序是真实时间顺序。
3. 把原始 movieId 映射成从 1 开始的 item index，0 专门留给 padding。
4. 为训练、验证、测试构造前缀-目标样本。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SequenceSpec:
    """序列推荐需要的 ID 映射。

    item index 从 1 开始，0 留给 padding。
    这样短序列可以在左侧补 0，模型用 `padding_idx=0` 忽略这些空位。
    """

    movie_to_index: dict[int, int]
    index_to_movie: dict[int, int]
    max_seq_len: int

    @property
    def num_items(self) -> int:
        """真实电影数量，不包含 padding。"""

        return len(self.movie_to_index)


class NextItemDataset(Dataset):
    """next item prediction 数据集。

    每条样本包含：
    - `sequence`：固定长度历史电影 index，左侧 padding。
    - `target`：下一部电影 index。

    这里提前保存 numpy 数组，而不是在 `__getitem__` 里做 pandas 操作。
    DataLoader 多进程读取 numpy 数组更轻，退出也更容易清理。
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = sequences.astype(np.int64, copy=False)
        self.targets = targets.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回一条 `(history_sequence, next_item)` 样本。"""

        return torch.from_numpy(self.sequences[index]), torch.tensor(self.targets[index], dtype=torch.long)


def _left_pad(values: list[int], max_seq_len: int) -> np.ndarray:
    """把变长历史序列左侧补 0 到固定长度。

    序列模型通常更关心最近行为，所以如果历史超过 `max_seq_len`，
    这里保留最后 `max_seq_len` 个 item。
    """

    clipped = values[-max_seq_len:]
    output = np.zeros(max_seq_len, dtype=np.int64)
    output[-len(clipped) :] = clipped
    return output


def build_sequence_spec(ratings: pd.DataFrame, max_seq_len: int) -> SequenceSpec:
    """根据训练可见数据建立 movieId 到 item index 的映射。

    只根据传入 ratings 建映射，避免提前把测试集独有电影放进 embedding 表。
    """

    movie_ids = sorted(ratings["movieId"].unique())
    movie_to_index = {int(movie_id): index + 1 for index, movie_id in enumerate(movie_ids)}
    index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
    return SequenceSpec(movie_to_index=movie_to_index, index_to_movie=index_to_movie, max_seq_len=max_seq_len)


def build_user_sequences(ratings: pd.DataFrame, positive_threshold: float = 4.0, min_sequence_len: int = 4) -> dict[int, list[int]]:
    """把 MovieLens 评分表转换成用户正反馈电影序列。

    只保留 `rating >= positive_threshold` 的记录。
    至少需要 `min_sequence_len` 个正反馈电影，才能切出 train/valid/test：
    前面若干个做训练，倒数第二个做验证目标，最后一个做测试目标。
    """

    positives = ratings[ratings["rating"] >= positive_threshold].copy()
    user_sequences: dict[int, list[int]] = {}
    for user_id, group in tqdm(positives.groupby("userId"), desc="Build user sequences", unit="user"):
        ordered = group.sort_values("timestamp")["movieId"].astype(int).tolist()
        if len(ordered) >= min_sequence_len:
            user_sequences[int(user_id)] = ordered
    return user_sequences


def _encode_known(sequence: list[int], spec: SequenceSpec) -> list[int]:
    """把原始 movieId 序列转换成 item index，并过滤训练中没见过的电影。"""

    return [spec.movie_to_index[movie_id] for movie_id in sequence if movie_id in spec.movie_to_index]


def build_next_item_datasets(
    user_sequences: dict[int, list[int]],
    spec: SequenceSpec,
    max_examples_per_user: int,
) -> tuple[NextItemDataset, NextItemDataset, NextItemDataset]:
    """构建训练、验证、测试数据集。

    切分方式：
    - 每个用户最后一个电影作为测试目标。
    - 倒数第二个电影作为验证目标。
    - 更早的位置切成训练样本。

    `max_examples_per_user` 只限制每个用户贡献多少条训练前缀，默认保留最近的若干条。
    这样全量 MovieLens 也能在本地训练，不会因为长历史用户生成过多样本。
    """

    train_sequences: list[np.ndarray] = []
    train_targets: list[int] = []
    valid_sequences: list[np.ndarray] = []
    valid_targets: list[int] = []
    test_sequences: list[np.ndarray] = []
    test_targets: list[int] = []

    for raw_sequence in tqdm(user_sequences.values(), desc="Create next-item examples", unit="user"):
        encoded = _encode_known(raw_sequence, spec)
        if len(encoded) < 4:
            continue

        test_sequences.append(_left_pad(encoded[:-1], spec.max_seq_len))
        test_targets.append(encoded[-1])
        valid_sequences.append(_left_pad(encoded[:-2], spec.max_seq_len))
        valid_targets.append(encoded[-2])

        train_end = len(encoded) - 2
        train_positions = range(1, train_end)
        if max_examples_per_user > 0:
            train_positions = list(train_positions)[-max_examples_per_user:]
        for target_position in train_positions:
            history = encoded[:target_position]
            target = encoded[target_position]
            train_sequences.append(_left_pad(history, spec.max_seq_len))
            train_targets.append(target)

    return (
        NextItemDataset(np.stack(train_sequences), np.array(train_targets)),
        NextItemDataset(np.stack(valid_sequences), np.array(valid_targets)),
        NextItemDataset(np.stack(test_sequences), np.array(test_targets)),
    )

