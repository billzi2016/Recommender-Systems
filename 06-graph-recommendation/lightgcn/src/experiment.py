from __future__ import annotations

"""LightGCN 实验轻量封装。

入口文件保持很薄，真正的图构建、训练、评估都复用 06 组 common 模块。
这样 LightGCN 和 NGCF 的差异集中在模型结构，而不是重复的工程代码。
"""

from pathlib import Path

from common.runner import run_graph_experiment


def run() -> None:
    """运行 LightGCN 图推荐实验。"""

    run_graph_experiment("lightgcn", Path(__file__).resolve().parents[1])

