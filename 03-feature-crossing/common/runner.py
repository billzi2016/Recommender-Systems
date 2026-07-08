from __future__ import annotations

"""FM 系列实验的统一入口逻辑。

03 组实验的差异主要在模型结构，数据读取、切分、训练、写报告都一样。
把公共流程集中到 runner，可以让 FM/DeepFM/xDeepFM 的 main.py 保持很薄。
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from common.features import build_dataset, build_feature_spec
from common.models import build_model
from common.training import train_feature_crossing_model

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.movielens import attach_titles, load_movielens, train_test_by_user_time
from utils.reports import write_report


MODEL_TITLES = {
    "fm": ("Factorization Machine", "FM 因子分解机"),
    "deepfm": ("DeepFM", "DeepFM"),
    "xdeepfm": ("xDeepFM", "xDeepFM"),
}


def parse_feature_crossing_args(description: str) -> argparse.Namespace:
    """解析 FM 系列实验共用参数。

    这里保留必要参数，不做复杂 CLI：
    数据规模、embedding 维度、batch、worker、early stopping 和 checkpoint。
    """

    parser = argparse.ArgumentParser(description=description)
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Sparse feature embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for training.")
    parser.add_argument("--eval-rows", type=int, default=2000, help="Rows used for human-readable prediction examples.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and optional sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    return parser.parse_args()


def _prediction_examples(model, test: pd.DataFrame, movies: pd.DataFrame, user_to_feature: dict[int, int], movie_to_feature: dict[int, int], movie_genre_features, spec, eval_rows: int) -> tuple[str, str]:
    """从测试集里抽一小段，展示模型认为更可能喜欢的电影。

    这不是完整推荐列表，只是报告里的人工检查窗口。
    如果指标不错但样例很奇怪，说明还要继续查切分、特征或标签。
    """

    sample = test.head(eval_rows).copy()
    dataset = build_dataset(sample, user_to_feature, movie_to_feature, movie_genre_features, spec)
    model.eval()
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            feature_ids, _ = dataset[index]
            score = torch.sigmoid(model(feature_ids.unsqueeze(0).to(next(model.parameters()).device))).item()
            rows.append(score)
    sample["predicted_like_probability"] = rows
    sample = sample.sort_values("predicted_like_probability", ascending=False).head(10)
    display = attach_titles(sample[["userId", "movieId", "rating", "predicted_like_probability"]], movies)
    display = display[["userId", "title", "genres", "rating", "predicted_like_probability"]]
    english = "Top scored held-out rows:\n\n" + display.to_markdown(index=False)
    chinese = "测试集中模型打分最高的几条记录：\n\n" + display.to_markdown(index=False)
    return english, chinese


def run_feature_crossing_experiment(kind: str, algorithm_dir: Path) -> None:
    """运行一个 FM 系列实验并写报告。

    主流程：
    1. 读取 MovieLens。
    2. 按用户时间切分 train/valid/test。
    3. 构建稀疏特征编号。
    4. 训练指定模型。
    5. 写入真实指标、样例和 checkpoint 大小。
    """

    title, title_zh = MODEL_TITLES[kind]
    args = parse_feature_crossing_args(f"Run {title} on MovieLens.")
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)

    print(f"[{kind}] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    # 先留出测试集，再从训练部分切验证集。
    # 验证集只服务 early stopping，测试集用于报告里的最终观察。
    train_valid, test = train_test_by_user_time(data.ratings)
    train, valid = train_test_by_user_time(train_valid)

    print(f"[{kind}] 构建用户、电影、genre、时间段稀疏特征。")
    spec, user_to_feature, movie_to_feature, movie_genre_features = build_feature_spec(train_valid, data.movies)
    train_dataset = build_dataset(train, user_to_feature, movie_to_feature, movie_genre_features, spec)
    valid_dataset = build_dataset(valid, user_to_feature, movie_to_feature, movie_genre_features, spec)
    model = build_model(kind, num_features=spec.num_features, max_fields=spec.max_fields, embedding_dim=args.embedding_dim)

    checkpoint_dir = algorithm_dir / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print(f"[{kind}] 不保存 .pt checkpoint。")
    else:
        print(f"[{kind}] checkpoint 保存目录：{checkpoint_dir}")

    result = train_feature_crossing_model(
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

    examples, examples_zh = _prediction_examples(result.model, test, data.movies, user_to_feature, movie_to_feature, movie_genre_features, spec, args.eval_rows)
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
            "- Used user ID, movie ID, movie genres, and timestamp hour bucket as sparse features.",
            f"- DataLoader workers: `{args.num_workers}`.",
            f"- Device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。",
            "- 使用用户 ID、电影 ID、电影类型、时间段作为稀疏特征。",
            f"- DataLoader 使用 `{args.num_workers}` 个 worker。",
            f"- 本次使用设备：`{result.device_name}`。",
        ],
        {
            "best_valid_logloss": result.best_valid_loss,
            "best_valid_auc": result.best_valid_auc,
            "epochs_ran": float(result.epochs_ran),
        },
        examples,
        examples_zh,
    )
    print(f"[{kind}] 已生成 report.md 和 report.zh.md")
