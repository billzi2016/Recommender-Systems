from __future__ import annotations

"""Item-CF 实验入口。

运行方式：
    ./run.sh

这个文件只做实验编排：
1. 读取 MovieLens。
2. 按时间切分训练集和测试集。
3. 训练 Item-CF。
4. 为一个用户生成推荐。
5. 写入 report.md 和 report.zh.md。

算法细节放在 `src/model.py`，公共工具放在根目录 `utils/`。
"""

import argparse
from pathlib import Path

import pandas as pd

from src.model import fit_item_cf, recommend_for_user

from utils.metrics import precision_at_k, recall_at_k
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text


def parse_args() -> argparse.Namespace:
    """解析实验参数。

    这里只提供当前实验真正需要的参数，不做复杂 CLI 框架。
    `--sample-ratings none` 表示使用全量 MovieLens。
    """

    parser = argparse.ArgumentParser(description="Run Item-CF on MovieLens.")
    add_sample_ratings_arg(parser)
    return parser.parse_args()


def main() -> None:
    """运行一次完整 Item-CF 实验，并生成报告。

    这里的报告不是假数据，也不是预写结论。
    用户实际运行脚本后，report 会被当前数据切分、当前参数下的真实结果覆盖。
    """

    args = parse_args()
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)
    print(f"[Item-CF] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    train, test = train_test_by_user_time(data.ratings)

    print("[Item-CF] 使用 sklearn NearestNeighbors 训练电影-电影相似度。")
    model = fit_item_cf(train)

    # 为了让样例更容易读，优先选择测试集中行为较多的用户。
    # 如果测试集为空，就退回训练集第一个用户，保证脚本仍能生成报告。
    user_id = int(test["userId"].value_counts().index[0]) if len(test) else int(train["userId"].iloc[0])
    recommendations = recommend_for_user(model, train, user_id=user_id, top_k=10)
    rec_df = attach_titles(
        pd.DataFrame(recommendations, columns=["movieId", "score"]),
        data.movies,
    )

    # 这里的 relevant 来自测试集里该用户后续的高评分电影。
    # 如果用户在测试集中没有高评分电影，precision/recall 会自然变成 0。
    relevant = set(test[(test["userId"] == user_id) & (test["rating"] >= 4.0)]["movieId"].tolist())
    rec_ids = [movie_id for movie_id, _ in recommendations]
    metrics = {
        "precision@10": precision_at_k(rec_ids, relevant, 10),
        "recall@10": recall_at_k(rec_ids, relevant, 10),
    }

    examples = rec_df[["title", "genres", "score"]].to_markdown(index=False)
    write_report(
        Path(__file__).resolve().parent,
        "Item-CF",
        "Item-CF",
        [
            "- Loaded a deterministic sample from MovieLens 32M.",
            "- Treated ratings >= 4.0 as positive feedback.",
            "- Built item-item cosine neighbors with sklearn.",
        ],
        [
            f"- 从 MovieLens 32M 读取：{sample_text}。",
            "- 将评分大于等于 4.0 的记录当作正反馈。",
            "- 使用 sklearn 的近邻模型计算电影之间的余弦相似度。",
        ],
        metrics,
        examples,
        examples,
    )
    print("[Item-CF] 已生成 report.md 和 report.zh.md")


if __name__ == "__main__":
    main()
