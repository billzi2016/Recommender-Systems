from __future__ import annotations

"""NCF 深度精排实验入口。

运行方式：
    ./04-deep-ranking/ncf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0

NCF 用 MLP 替代矩阵分解里的点积，让模型自己学习用户和电影 embedding
之间的交互方式。

这个文件故意很薄：
- 参数解析放在 common runner。
- 模型定义放在 common models。
- 这里只保留“从命令行启动 NCF 实验”的职责。
"""

from src.experiment import run


def main() -> None:
    """运行 NCF 实验，并生成 report.md / report.zh.md。"""

    run()


if __name__ == "__main__":
    main()
