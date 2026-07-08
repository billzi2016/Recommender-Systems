from __future__ import annotations

"""LightGCN 图推荐实验入口。

运行方式：
    ./06-graph-recommendation/lightgcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

LightGCN 把用户和电影看成二部图，只做 embedding 传播和多层平均，
不在传播过程中加入 MLP 或激活函数。
"""

from src.experiment import run


def main() -> None:
    """运行 LightGCN 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()

