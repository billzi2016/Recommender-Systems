from __future__ import annotations

"""NGCF 图推荐实验入口。

运行方式：
    ./06-graph-recommendation/ngcf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

NGCF 在用户-电影图上传播 embedding，并在传播时加入神经网络变换。
"""

from src.experiment import run


def main() -> None:
    """运行 NGCF 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()

