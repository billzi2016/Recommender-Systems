from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """返回仓库根目录。

    每个实验都在多层算法目录下面。如果直接依赖当前工作目录，
    用户从不同目录执行脚本时很容易出错。

    这里用 `utils` 文件本身的位置反推出仓库根目录，
    这样 `run.sh`、`main.py`、测试脚本从哪里启动都能找到数据。
    """

    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    """返回 MovieLens 数据集目录。"""
    return repo_root() / "dataset-movie_lens"


def raw_dir() -> Path:
    """返回原始数据目录，里面放官方 zip 和解压后的 CSV。"""
    return dataset_root() / "raw"


def processed_dir() -> Path:
    """返回处理后数据目录，后续可放切分后的训练集和测试集。"""
    return dataset_root() / "processed"
