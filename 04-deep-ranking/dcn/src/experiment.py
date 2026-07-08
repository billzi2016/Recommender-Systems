from __future__ import annotations

"""DCN 实验的轻量封装。

DCN 和 Wide&Deep 使用同一套上下文特征。
区别只在模型结构：DCN 多了 cross layer，用来显式学习特征交叉。
"""

from pathlib import Path

from common.runner import run_deep_ranking_experiment


def run() -> None:
    """运行 DCN 深度精排实验。

    训练、评估和 report 生成都交给公共 runner，避免重复代码。
    """

    run_deep_ranking_experiment("dcn", Path(__file__).resolve().parents[1])
