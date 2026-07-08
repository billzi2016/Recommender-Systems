from __future__ import annotations

"""PyTorch 双塔召回实验入口。

运行方式：
    ./02-retrieval/two-tower-tfrs/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 0

这个实验只做召回，不做最终精排。训练目标是让用户向量靠近高评分电影向量。
"""

import argparse
from pathlib import Path

import pandas as pd

from src.model import recommend_for_users, train_two_tower

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.metrics import mean_ranking_metrics
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report


def parse_args() -> argparse.Namespace:
    """解析双塔实验参数。

    参数保持在少数几个真正会调的维度上：
    数据规模、embedding 大小、batch 大小、early stopping、DataLoader worker
    和 checkpoint 策略。这样命令行足够实用，但不会变成一个复杂 CLI 项目。
    """

    parser = argparse.ArgumentParser(description="Run PyTorch two tower retrieval on MovieLens.")
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="User and movie embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size. Batch items also act as in-batch negatives.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for training.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and optional sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs when checkpoint saving is enabled. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    parser.add_argument("--force-train", action="store_true", help="Ignore checkpoints/best.pt and train again.")
    parser.add_argument("--eval-users", type=int, default=200, help="Number of users used for retrieval metric examples.")
    return parser.parse_args()


def report_examples(recommendations: dict[int, list[int]], train: pd.DataFrame, movies: pd.DataFrame) -> tuple[str, str]:
    """生成报告里的推荐样例。

    只展示一个用户，避免报告太长。完整评估指标在 metrics 里。
    """

    if not recommendations:
        return "No recommendations were produced.", "本次没有生成推荐样例。"

    user_id = next(iter(recommendations))
    history = attach_titles(
        train[(train["userId"] == user_id) & (train["rating"] >= 4.0)].tail(8)[["movieId"]].copy(),
        movies,
    )
    rec_df = attach_titles(pd.DataFrame({"movieId": recommendations[user_id]}), movies)
    english = (
        f"Example user: `{user_id}`\n\n"
        "Recent positive history:\n\n"
        + history[["title", "genres"]].to_markdown(index=False)
        + "\n\nRetrieved candidates:\n\n"
        + rec_df[["title", "genres"]].to_markdown(index=False)
    )
    chinese = (
        f"样例用户：`{user_id}`\n\n"
        "最近高评分历史：\n\n"
        + history[["title", "genres"]].to_markdown(index=False)
        + "\n\n召回候选：\n\n"
        + rec_df[["title", "genres"]].to_markdown(index=False)
    )
    return english, chinese


def main() -> None:
    """运行双塔召回实验，并生成 report.md / report.zh.md。

    流程分成四步：
    1. 按用户时间切分，避免用未来行为预测过去。
    2. 训练双塔，让高评分电影和用户向量靠近。
    3. 对测试用户召回 top-k 候选。
    4. 写入真实指标、样例和 checkpoint 大小。
    """

    args = parse_args()
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)
    print(f"[TwoTower] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    # 先切出测试集，再从 train_valid 里切验证集。
    # 这样验证集用于 early stopping，测试集用于最终报告指标。
    train_valid, test = train_test_by_user_time(data.ratings)
    train, valid = train_test_by_user_time(train_valid)

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print("[TwoTower] 不保存 .pt checkpoint。")
    else:
        print(f"[TwoTower] checkpoint 保存目录：{checkpoint_dir}")

    print("[TwoTower] 使用 PyTorch 训练双塔召回模型，并使用 early stopping。")
    result = train_two_tower(
        train,
        valid,
        embedding_dim=args.embedding_dim,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
        force_train=args.force_train,
    )

    relevant_by_user = {
        int(user_id): set(group[group["rating"] >= 4.0]["movieId"].tolist())
        for user_id, group in test.groupby("userId")
    }
    # 只评估训练中见过、且测试集中确实有高评分电影的用户。
    # 否则 recall/NDCG 没有明确意义。
    eval_users = [user_id for user_id in relevant_by_user if user_id in result.user_to_index and relevant_by_user[user_id]][: args.eval_users]
    recommendations = recommend_for_users(result, train, eval_users, top_k=10)
    metrics = mean_ranking_metrics(recommendations, relevant_by_user, k=10)
    metrics["best_valid_loss"] = result.best_valid_loss

    examples, examples_zh = report_examples(recommendations, train, data.movies)
    checkpoint_md, checkpoint_zh_md = checkpoint_size_markdown(checkpoint_dir)
    examples += "\n\n## Checkpoint size\n\n" + checkpoint_md
    examples_zh += "\n\n## Checkpoint 大小\n\n" + checkpoint_zh_md

    write_report(
        Path(__file__).resolve().parent,
        "Two tower retrieval",
        "双塔召回",
        [
            f"- Loaded MovieLens: {sample_text}.",
            "- Trained a PyTorch two tower retrieval model with in-batch negatives.",
            f"- Device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 使用 PyTorch 训练双塔召回模型，batch 内其他电影作为近似负样本。",
            f"- DataLoader 使用 `{args.num_workers}` 个 worker。",
            f"- 本次使用设备：`{result.device_name}`。",
        ],
        metrics,
        examples,
        examples_zh,
    )
    print("[TwoTower] 已生成 report.md 和 report.zh.md")


if __name__ == "__main__":
    main()
