from __future__ import annotations

"""矩阵分解实验入口。

运行方式：
    ./run.sh

这个实验使用 PyTorch。设备会自动选择 cuda、mps 或 cpu。
训练完成后会输出评分预测指标，并给一部样例电影找 embedding 最近邻。
"""

import argparse
from pathlib import Path

import pandas as pd

from src.model import similar_movies, train_mf

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report


def parse_args() -> argparse.Namespace:
    """解析实验参数。

    矩阵分解是第一个 PyTorch 训练实验，所以这里暴露的参数比 CF 多：
    - early stopping 相关参数，控制训练什么时候停。
    - DataLoader worker，控制 batch 准备速度。
    - checkpoint 策略，控制是否写 `.pt` 文件。
    """

    parser = argparse.ArgumentParser(description="Run matrix factorization on MovieLens.")
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for training.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and a few sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs when checkpoint saving is enabled. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    return parser.parse_args()


def main() -> None:
    """运行矩阵分解实验，并生成 report.md / report.zh.md。

    这个入口只做实验编排，不写模型细节：
    1. 读取 MovieLens。
    2. 按用户时间切分 train/valid/test。
    3. 训练带偏置的矩阵分解。
    4. 用电影 embedding 找相似电影。
    5. 写真实指标、样例和 checkpoint 大小。
    """

    args = parse_args()
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)
    print(f"[MF] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    # 先留测试集，再从训练部分切验证集。
    # 验证集用于 early stopping，测试集保留给报告里的最终指标。
    train_valid, test = train_test_by_user_time(data.ratings)
    train, valid = train_test_by_user_time(train_valid)

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print("[MF] 不保存 .pt checkpoint，避免不必要的磁盘写入。")
    else:
        print(f"[MF] checkpoint 保存目录：{checkpoint_dir}")

    print("[MF] 使用 PyTorch 训练带偏置的矩阵分解模型，并使用 early stopping。")
    result = train_mf(
        train,
        valid,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
    )

    # 选择训练集中评分数量较多的一部电影，避免样例电影太冷门导致近邻难看。
    example_movie_id = int(train["movieId"].value_counts().index[0])
    neighbors = similar_movies(result, example_movie_id, top_k=10)
    neighbor_df = attach_titles(pd.DataFrame(neighbors, columns=["movieId", "similarity"]), data.movies)
    example_title = data.movies.loc[data.movies["movieId"] == example_movie_id, "title"].iloc[0]
    checkpoint_md, checkpoint_zh_md = checkpoint_size_markdown(checkpoint_dir)
    examples = (
        f"Example movie: `{example_title}`\n\n"
        + neighbor_df[["title", "genres", "similarity"]].to_markdown(index=False)
        + "\n\n## Checkpoint size\n\n"
        + checkpoint_md
    )
    examples_zh = (
        f"样例电影：`{example_title}`\n\n"
        + neighbor_df[["title", "genres", "similarity"]].to_markdown(index=False)
        + "\n\n## Checkpoint 大小\n\n"
        + checkpoint_zh_md
    )

    write_report(
        Path(__file__).resolve().parent,
        "Matrix factorization",
        "矩阵分解",
        [
            "- Loaded a deterministic MovieLens sample.",
            "- Trained a biased matrix factorization model with PyTorch and early stopping.",
            f"- Device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            f"- 使用 PyTorch 训练带用户偏置和电影偏置的矩阵分解模型，最多 {args.max_epochs} 轮，early stopping patience 为 {args.patience}。",
            f"- DataLoader 使用 `{args.num_workers}` 个 worker。",
            f"- 本次使用设备：`{result.device_name}`。",
            "- 默认保存 `best.pt`，并按间隔最多保留少量中间 checkpoint；如需关闭可传 `--no-save-checkpoints`。",
        ],
        result.metrics,
        examples,
        examples_zh,
    )
    print("[MF] 已生成 report.md 和 report.zh.md")


if __name__ == "__main__":
    main()
