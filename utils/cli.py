from __future__ import annotations

"""实验脚本共用的轻量命令行工具。

这里只放多个实验都会用到的参数解析逻辑。
不做复杂 CLI 框架，避免把项目变重。
"""

import argparse


DEFAULT_SAMPLE_RATINGS: int | None = None


def add_sample_ratings_arg(parser: argparse.ArgumentParser) -> None:
    """给实验入口添加统一的采样参数。

    默认使用全量 MovieLens。用户想快速试跑时，再传较小采样值。
    """

    parser.add_argument(
        "--sample-ratings",
        default="none" if DEFAULT_SAMPLE_RATINGS is None else str(DEFAULT_SAMPLE_RATINGS),
        help="Number of ratings to sample. Use 'none' for the full dataset.",
    )


def parse_sample_ratings(value: str) -> int | None:
    """把命令行传入的采样参数转换成程序使用的值。"""

    if value.lower() == "none":
        return None
    return int(value)


def sample_ratings_text(sample_ratings: int | None) -> str:
    """把采样规模转换成适合打印和写报告的文字。"""

    return "全量数据" if sample_ratings is None else f"{sample_ratings:,} 条评分"
