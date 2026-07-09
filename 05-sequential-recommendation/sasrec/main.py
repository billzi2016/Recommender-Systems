from __future__ import annotations

"""SASRec 序列推荐实验入口。

运行方式：
    ./05-sequential-recommendation/sasrec/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0

SASRec 使用 causal self-attention 读取用户历史。
causal mask 会阻止模型看到未来电影，避免 next-item 任务泄露答案。
"""

from src.experiment import run


def main() -> None:
    """运行 SASRec 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()
