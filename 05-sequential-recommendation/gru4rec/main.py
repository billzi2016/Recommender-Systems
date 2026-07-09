from __future__ import annotations

"""GRU4Rec 序列推荐实验入口。

运行方式：
    ./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

GRU4Rec 把用户高评分电影按时间排序，用 GRU 读取历史序列，
再预测下一部更可能出现的电影。
"""

from src.experiment import run


def main() -> None:
    """运行 GRU4Rec 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()
