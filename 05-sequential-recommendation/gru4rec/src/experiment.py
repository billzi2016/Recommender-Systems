from __future__ import annotations

"""GRU4Rec 实验轻量封装。

`main.py` 只做命令入口。
这里保留一个 `src` 层，是为了让所有实验目录结构一致：
入口在 main，算法运行逻辑在 src/common。
"""

from pathlib import Path

from common.runner import run_sequential_experiment


def run() -> None:
    """运行 GRU4Rec 序列推荐实验。"""

    run_sequential_experiment("gru4rec", Path(__file__).resolve().parents[1])

