from __future__ import annotations

"""Wide&Deep 深度精排实验入口。

运行方式：
    ./04-deep-ranking/wide-and-deep/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

Wide 侧记住稳定规则，Deep 侧用 embedding 和 MLP 做泛化。

这个入口不直接写训练循环，是为了避免 Wide&Deep、DCN、NCF 三个目录复制
同一套读取数据、early stopping 和写 report 的代码。
"""

from src.experiment import run


def main() -> None:
    """运行 Wide&Deep 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()
