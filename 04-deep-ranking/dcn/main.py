from __future__ import annotations

"""DCN 深度精排实验入口。

运行方式：
    ./04-deep-ranking/dcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

DCN 用 cross layer 显式学习特征交叉，再和 deep network 的非线性表达合并。

这个入口只负责启动 DCN 实验。
模型细节和训练细节拆到 common 模块，是为了让读者能分别看“入口”“模型”“训练循环”。
"""

from src.experiment import run


def main() -> None:
    """运行 DCN 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()
