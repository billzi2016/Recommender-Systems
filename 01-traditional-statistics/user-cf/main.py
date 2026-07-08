from __future__ import annotations

"""User-CF 实验入口。

运行方式：
    ./run.sh

这个入口只负责串起一次完整实验：
读取数据、时间切分、训练用户近邻、生成推荐、写 Markdown 报告。
"""

import argparse
from pathlib import Path

import pandas as pd

from src.model import fit_user_cf, recommend_for_user

from utils.metrics import precision_at_k, recall_at_k
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text


def parse_args() -> argparse.Namespace:
    """解析实验参数，只保留当前需要的采样配置。"""

    parser = argparse.ArgumentParser(description="Run User-CF on MovieLens.")
    add_sample_ratings_arg(parser)
    return parser.parse_args()


def main() -> None:
    """运行 User-CF，并生成 report.md / report.zh.md。"""

    args = parse_args()
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)
    print(f"[User-CF] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    train, test = train_test_by_user_time(data.ratings)

    print("[User-CF] 使用 sklearn NearestNeighbors 训练用户-用户相似度。")
    model = fit_user_cf(train)

    user_id = int(test["userId"].value_counts().index[0]) if len(test) else int(train["userId"].iloc[0])
    recommendations = recommend_for_user(model, train, user_id=user_id, top_k=10)
    rec_df = attach_titles(pd.DataFrame(recommendations, columns=["movieId", "score"]), data.movies)

    relevant = set(test[(test["userId"] == user_id) & (test["rating"] >= 4.0)]["movieId"].tolist())
    rec_ids = [movie_id for movie_id, _ in recommendations]
    metrics = {
        "precision@10": precision_at_k(rec_ids, relevant, 10),
        "recall@10": recall_at_k(rec_ids, relevant, 10),
    }

    examples = rec_df[["title", "genres", "score"]].to_markdown(index=False)
    write_report(
        Path(__file__).resolve().parent,
        "User-CF",
        "User-CF",
        [
            "- Loaded a deterministic MovieLens sample.",
            "- Treated ratings >= 4.0 as positive feedback.",
            "- Found similar users with sklearn cosine nearest neighbors.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 将评分大于等于 4.0 的记录当作正反馈。",
            "- 使用 sklearn 的余弦近邻搜索寻找相似用户。",
        ],
        metrics,
        examples,
        examples,
    )
    print("[User-CF] 已生成 report.md 和 report.zh.md")


if __name__ == "__main__":
    main()
