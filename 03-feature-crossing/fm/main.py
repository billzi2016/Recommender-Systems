from __future__ import annotations

"""FM 特征交叉实验入口。

运行方式：
    ./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

FM 会把用户 ID、电影 ID、电影类型、时间段放进同一个稀疏特征集合，
再用 embedding 内积学习任意两个特征之间的二阶关系。
"""

from pathlib import Path

from common.runner import run_feature_crossing_experiment


def main() -> None:
    """运行 FM 实验，并生成 report.md / report.zh.md。"""

    run_feature_crossing_experiment("fm", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

