from __future__ import annotations

"""SASRec 实验轻量封装。

SASRec 和 GRU4Rec 复用同一套序列数据与训练循环。
差异只在模型结构：SASRec 使用 causal self-attention。
"""

from pathlib import Path

from common.runner import run_sequential_experiment


def run() -> None:
    """运行 SASRec 序列推荐实验。"""

    run_sequential_experiment("sasrec", Path(__file__).resolve().parents[1])

