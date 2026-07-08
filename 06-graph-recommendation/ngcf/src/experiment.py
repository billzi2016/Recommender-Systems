from __future__ import annotations

"""NGCF 实验轻量封装。

NGCF 和 LightGCN 复用同一套二部图数据。
区别在模型：NGCF 在消息传递时加入线性变换、非线性和特征交互。
"""

from pathlib import Path

from common.runner import run_graph_experiment


def run() -> None:
    """运行 NGCF 图推荐实验。"""

    run_graph_experiment("ngcf", Path(__file__).resolve().parents[1])

