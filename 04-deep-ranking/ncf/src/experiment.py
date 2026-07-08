from __future__ import annotations

"""NCF 实验的轻量封装。

`main.py` 只负责作为命令入口。
真正运行逻辑放在这里，是为了保持每个算法目录都有 `src` 模块，
后面如果要加测试或扩展实验，也不用把所有代码塞进 main。
"""

from pathlib import Path

from common.runner import run_deep_ranking_experiment


def run() -> None:
    """运行 NCF 深度精排实验。

    具体的数据读取、训练、报告写入都复用 04 组公共 runner。
    """

    run_deep_ranking_experiment("ncf", Path(__file__).resolve().parents[1])
