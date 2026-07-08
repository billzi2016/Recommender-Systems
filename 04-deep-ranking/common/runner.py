from __future__ import annotations

"""04 深度精排实验统一 runner。

三个算法的训练流程高度相似：
读取 MovieLens、按用户时间切分、构建数据集、训练、写 report。
runner 把这些重复流程集中起来，让每个算法目录只保留很薄的入口文件。
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from common.data import build_context_dataset, build_pair_dataset, build_ranking_spec
from common.models import DCN, NCF, WideAndDeep
from common.training import train_ranking_model

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report


MODEL_TITLES = {
    "ncf": ("NCF", "NCF"),
    "wide-and-deep": ("Wide and deep", "Wide and deep"),
    "dcn": ("DCN", "DCN"),
}


def parse_args(description: str) -> argparse.Namespace:
    """解析 04 组实验参数。

    这里保持轻量 argparse，不引入复杂 CLI 框架。
    用户主要需要控制数据规模、DataLoader worker、checkpoint 和 early stopping。
    """

    parser = argparse.ArgumentParser(description=description)
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for training.")
    parser.add_argument("--eval-rows", type=int, default=2000, help="Rows used for human-readable examples.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and optional sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    return parser.parse_args()


def _score_examples(kind: str, model, test: pd.DataFrame, movies: pd.DataFrame, spec, eval_rows: int) -> tuple[str, str]:
    """从测试集里展示模型打分最高的几条记录。

    这不是完整线上推荐流程，只是给报告一个可人工检查的窗口。
    指标告诉你模型整体表现，样例告诉你模型有没有明显跑偏。
    """

    sample = test.head(eval_rows).copy()
    dataset = build_pair_dataset(sample, spec) if kind == "ncf" else build_context_dataset(sample, spec)
    device = next(model.parameters()).device
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            if kind == "ncf":
                batch = tuple(value.unsqueeze(0).to(device) for value in item)
            else:
                batch = {key: value.unsqueeze(0).to(device) for key, value in item.items()}
            scores.append(float(torch.sigmoid(model(batch)).item()))
    sample["predicted_like_probability"] = scores
    display = attach_titles(
        sample.sort_values("predicted_like_probability", ascending=False).head(10)[["userId", "movieId", "rating", "predicted_like_probability"]],
        movies,
    )
    display = display[["userId", "title", "genres", "rating", "predicted_like_probability"]]
    english = "Top scored held-out rows:\n\n" + display.to_markdown(index=False)
    chinese = "测试集中模型打分最高的几条记录：\n\n" + display.to_markdown(index=False)
    return english, chinese


def run_deep_ranking_experiment(kind: str, algorithm_dir: Path) -> None:
    """运行一个深度精排实验并写报告。

    NCF 只用 user/movie ID，适合观察“MLP 替代点积”本身。
    Wide&Deep 和 DCN 加入 genres、时间段，适合观察更多特征进入精排后的变化。
    """

    title, title_zh = MODEL_TITLES[kind]
    args = parse_args(f"Run {title} on MovieLens.")
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)

    print(f"[{kind}] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    # 按用户内部时间切分，避免用未来行为预测过去行为。
    train_valid, test = train_test_by_user_time(data.ratings)
    train, valid = train_test_by_user_time(train_valid)
    spec = build_ranking_spec(train_valid, data.movies)

    # NCF 和另外两个模型的输入结构不同：
    # NCF 是 tuple(user, movie, label)，上下文模型是 dict。
    if kind == "ncf":
        train_dataset = build_pair_dataset(train, spec)
        valid_dataset = build_pair_dataset(valid, spec)
        model = NCF(spec.num_users, spec.num_movies, args.embedding_dim)
        feature_text = "user ID and movie ID"
        feature_text_zh = "用户 ID 和电影 ID"
    else:
        train_dataset = build_context_dataset(train, spec)
        valid_dataset = build_context_dataset(valid, spec)
        model = WideAndDeep(spec, args.embedding_dim) if kind == "wide-and-deep" else DCN(spec, args.embedding_dim)
        feature_text = "user ID, movie ID, movie genres, and timestamp hour bucket"
        feature_text_zh = "用户 ID、电影 ID、电影 genres 和 timestamp 小时段"

    checkpoint_dir = algorithm_dir / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print(f"[{kind}] 不保存 .pt checkpoint。")
    else:
        print(f"[{kind}] checkpoint 保存目录：{checkpoint_dir}")

    result = train_ranking_model(
        model,
        train_dataset,
        valid_dataset,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
    )

    examples, examples_zh = _score_examples(kind, result.model, test, data.movies, spec, args.eval_rows)
    checkpoint_md, checkpoint_zh_md = checkpoint_size_markdown(checkpoint_dir)
    examples += "\n\n## Checkpoint size\n\n" + checkpoint_md
    examples_zh += "\n\n## Checkpoint 大小\n\n" + checkpoint_zh_md

    write_report(
        algorithm_dir,
        title,
        title_zh,
        [
            f"- Loaded MovieLens: {sample_text}.",
            "- Converted ratings into binary labels: `rating >= 4.0` means positive.",
            f"- Used {feature_text}.",
            f"- DataLoader workers: `{args.num_workers}`.",
            f"- Device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。",
            f"- 使用特征：{feature_text_zh}。",
            f"- DataLoader 使用 `{args.num_workers}` 个 worker。",
            f"- 本次使用设备：`{result.device_name}`。",
        ],
        {
            "best_valid_logloss": result.best_valid_loss,
            "best_valid_auc": result.best_valid_auc,
            "best_valid_accuracy": result.best_valid_accuracy,
            "epochs_ran": float(result.epochs_ran),
        },
        examples,
        examples_zh,
    )
    print(f"[{kind}] 已生成 report.md 和 report.zh.md")
