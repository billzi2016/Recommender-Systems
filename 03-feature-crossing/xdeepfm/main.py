from __future__ import annotations

"""xDeepFM 特征交叉实验入口。

运行方式：
    ./03-feature-crossing/xdeepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

xDeepFM 在 DeepFM 的思路上增加 CIN：它显式构造多层特征交叉，
让“哪些字段和哪些字段在一起有用”这件事更直接地进入模型。
"""

from pathlib import Path

from common.runner import run_feature_crossing_experiment


def main() -> None:
    """运行 xDeepFM 实验，并生成 report.md / report.zh.md。"""

    run_feature_crossing_experiment("xdeepfm", Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

