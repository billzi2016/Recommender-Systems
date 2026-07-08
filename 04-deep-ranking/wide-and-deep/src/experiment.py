from __future__ import annotations

"""Wide&Deep 实验的轻量封装。

保留这个 `src` 层，是为了让目录结构和其他实验一致。
新手看项目时，可以稳定地知道：`main.py` 是入口，`src` 是模块。
"""

from pathlib import Path

from common.runner import run_deep_ranking_experiment


def run() -> None:
    """运行 Wide&Deep 深度精排实验。

    公共 runner 会为它选择带上下文的输入特征：
    用户 ID、电影 ID、genres 和 timestamp 小时段。
    """

    run_deep_ranking_experiment("wide-and-deep", Path(__file__).resolve().parents[1])
