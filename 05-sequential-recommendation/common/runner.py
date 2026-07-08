from __future__ import annotations

"""05 序列推荐实验统一 runner。

GRU4Rec 和 SASRec 的公共流程完全一致：
读取 MovieLens、构造用户序列、切 next-item 样本、训练模型、写 report。
把这部分放在 runner，可以让两个算法目录只保留很薄的入口。
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from common.data import build_next_item_datasets, build_sequence_spec, build_user_sequences
from common.models import build_sequence_model
from common.training import evaluate_sequence_model, train_sequence_model

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.movielens import attach_titles, load_movielens
from utils.reports import write_report


MODEL_TITLES = {
    "gru4rec": ("GRU4Rec", "GRU4Rec"),
    "sasrec": ("SASRec", "SASRec"),
}


def parse_args(description: str) -> argparse.Namespace:
    """解析序列推荐实验参数。

    除了通用训练参数，序列模型额外需要：
    - `max-seq-len`：最多看多长的历史。
    - `max-examples-per-user`：每个用户最多贡献多少条训练前缀。
    """

    parser = argparse.ArgumentParser(description=description)
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Item embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for training.")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Maximum history length kept for each example.")
    parser.add_argument("--max-examples-per-user", type=int, default=20, help="Maximum training prefixes kept per user. Use 0 for all possible prefixes.")
    parser.add_argument("--top-k", type=int, default=10, help="Recall/NDCG cutoff.")
    parser.add_argument("--eval-users", type=int, default=5, help="Human-readable examples written to the report.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and optional sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    return parser.parse_args()


def _examples_markdown(model, test_dataset, spec, movies: pd.DataFrame, top_k: int, eval_users: int) -> tuple[str, str]:
    """生成报告里的序列推荐样例。

    展示历史序列、真实下一部电影和模型 top-k。
    这能帮助检查模型是否只推荐热门电影，或者是否真的跟历史顺序有关。
    """

    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(min(eval_users, len(test_dataset))):
            sequence, target = test_dataset[index]
            logits = model(sequence.unsqueeze(0).to(device))
            logits[:, 0] = -1e9
            top_items = torch.topk(logits, k=min(top_k, logits.shape[1])).indices.squeeze(0).cpu().tolist()
            history_ids = [spec.index_to_movie[int(item)] for item in sequence.tolist() if int(item) in spec.index_to_movie][-8:]
            rows.append(
                {
                    "history": " -> ".join(_titles(history_ids, movies)),
                    "true_next": _titles([spec.index_to_movie[int(target)]], movies)[0],
                    "top_recommendations": " | ".join(_titles([spec.index_to_movie[int(item)] for item in top_items if int(item) in spec.index_to_movie], movies)),
                }
            )
    if not rows:
        return "No examples were produced.", "本次没有生成样例。"
    frame = pd.DataFrame(rows)
    return frame.to_markdown(index=False), frame.to_markdown(index=False)


def _titles(movie_ids: list[int], movies: pd.DataFrame) -> list[str]:
    """把 movieId 列表转成标题列表，供 report 样例展示。"""

    if not movie_ids:
        return []
    frame = attach_titles(pd.DataFrame({"movieId": movie_ids}), movies)
    return frame["title"].fillna("<unknown>").tolist()


def run_sequential_experiment(kind: str, algorithm_dir: Path) -> None:
    """运行一个序列推荐实验并写报告。"""

    title, title_zh = MODEL_TITLES[kind]
    args = parse_args(f"Run {title} on MovieLens.")
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)

    print(f"[{kind}] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    user_sequences = build_user_sequences(data.ratings)
    # 用所有可见正反馈电影建 item 表。每个用户最后两步只作为 valid/test 目标，
    # 但电影本身仍来自 MovieLens 全部正反馈，避免测试目标因为不在词表而无法评估。
    spec = build_sequence_spec(data.ratings[data.ratings["rating"] >= 4.0], max_seq_len=args.max_seq_len)
    train_dataset, valid_dataset, test_dataset = build_next_item_datasets(user_sequences, spec, args.max_examples_per_user)
    model = build_sequence_model(kind, num_items=spec.num_items, max_seq_len=args.max_seq_len, embedding_dim=args.embedding_dim)

    checkpoint_dir = algorithm_dir / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print(f"[{kind}] 不保存 .pt checkpoint。")
    else:
        print(f"[{kind}] checkpoint 保存目录：{checkpoint_dir}")

    result = train_sequence_model(
        model,
        train_dataset,
        valid_dataset,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        top_k=args.top_k,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loss, test_recall, test_ndcg = evaluate_sequence_model(result.model, test_loader, next(result.model.parameters()).device, args.top_k)
    examples, examples_zh = _examples_markdown(result.model, test_dataset, spec, data.movies, args.top_k, args.eval_users)
    checkpoint_md, checkpoint_zh_md = checkpoint_size_markdown(checkpoint_dir)
    examples += "\n\n## Checkpoint size\n\n" + checkpoint_md
    examples_zh += "\n\n## Checkpoint 大小\n\n" + checkpoint_zh_md

    write_report(
        algorithm_dir,
        title,
        title_zh,
        [
            f"- Loaded MovieLens: {sample_text}.",
            "- Kept high-rating movies as positive user sequences.",
            f"- Used max sequence length `{args.max_seq_len}` and max `{args.max_examples_per_user}` training examples per user.",
            f"- DataLoader workers: `{args.num_workers}`.",
            f"- Device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 只保留高评分电影，按用户 timestamp 构造正反馈序列。",
            f"- 最大序列长度 `{args.max_seq_len}`，每个用户最多 `{args.max_examples_per_user}` 条训练前缀。",
            f"- DataLoader 使用 `{args.num_workers}` 个 worker。",
            f"- 本次使用设备：`{result.device_name}`。",
        ],
        {
            "best_valid_loss": result.best_valid_loss,
            f"best_valid_recall@{args.top_k}": result.best_valid_recall,
            f"best_valid_ndcg@{args.top_k}": result.best_valid_ndcg,
            "test_loss": test_loss,
            f"test_recall@{args.top_k}": test_recall,
            f"test_ndcg@{args.top_k}": test_ndcg,
            "epochs_ran": float(result.epochs_ran),
        },
        examples,
        examples_zh,
    )
    print(f"[{kind}] 已生成 report.md 和 report.zh.md")

