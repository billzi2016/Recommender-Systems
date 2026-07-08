from __future__ import annotations

"""DeepFM 特征交叉实验入口。

运行方式：
    ./03-feature-crossing/deepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

DeepFM 一边保留 FM 的二阶交叉，一边把所有稀疏特征 embedding 展平后交给 MLP，
让模型继续学习更高阶、更非线性的组合关系。
"""

from pathlib import Path

from common.runner import run_feature_crossing_experiment


def main() -> None:
    """运行 DeepFM 实验，并生成 report.md / report.zh.md。"""

    run_feature_crossing_experiment("deepfm", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

