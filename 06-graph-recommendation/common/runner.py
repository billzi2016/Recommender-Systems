from __future__ import annotations

"""06 图推荐实验统一 runner。"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from common.data import build_graph_spec, build_normalized_adjacency, build_positive_edges, encode_edges, make_bpr_dataset, split_edges_by_user_time
from common.models import build_graph_model
from common.training import evaluate_graph_model, train_graph_model

from utils.checkpoints import checkpoint_size_markdown
from utils.cli import add_sample_ratings_arg, parse_sample_ratings, sample_ratings_text
from utils.movielens import attach_titles, load_movielens
from utils.reports import write_report


MODEL_TITLES = {
    "lightgcn": ("LightGCN", "LightGCN"),
    "ngcf": ("NGCF", "NGCF"),
}


def parse_args(description: str) -> argparse.Namespace:
    """解析图推荐实验参数。"""

    parser = argparse.ArgumentParser(description=description)
    add_sample_ratings_arg(parser)
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience measured in epochs.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="User and movie embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Graph propagation layers.")
    parser.add_argument("--batch-size", type=int, default=4096, help="BPR training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--num-workers", type=int, default=8, help="PyTorch DataLoader workers for BPR triples.")
    parser.add_argument("--max-positives-per-user", type=int, default=50, help="Keep at most this many recent positive edges per user. Use 0 for all positives.")
    parser.add_argument("--top-k", type=int, default=10, help="Recall/NDCG cutoff.")
    parser.add_argument("--eval-users", type=int, default=5, help="Human-readable examples written to the report.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", default=True, help="Save best.pt and optional sparse intermediate checkpoints.")
    checkpoint_group.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Do not write any .pt checkpoint files.")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save one intermediate checkpoint every N epochs. Use 0 to keep only best.pt.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Keep at most this many intermediate checkpoints.")
    parser.add_argument("--force-train", action="store_true", help="Ignore checkpoints/best.pt and train again.")
    return parser.parse_args()


def _positives_by_user(edges: pd.DataFrame, spec) -> dict[int, set[int]]:
    """把验证/测试边转换成 user index -> movie index 集合。"""

    users, movies, positives = encode_edges(edges[edges["userId"].isin(spec.user_to_index) & edges["movieId"].isin(spec.movie_to_index)], spec)
    return positives


def _example_markdown(model, adjacency: torch.Tensor, train_user_positives: dict[int, set[int]], test_user_positives: dict[int, set[int]], spec, movies: pd.DataFrame, top_k: int, eval_users: int) -> tuple[str, str]:
    """生成 report 样例：用户测试正样本和模型 top-k 推荐。"""

    device = next(model.parameters()).device
    adjacency = adjacency.to(device)
    rows = []
    model.eval()
    with torch.no_grad():
        user_embeddings, movie_embeddings = model.propagate(adjacency)
        for user, positives in list(test_user_positives.items())[:eval_users]:
            scores = user_embeddings[user] @ movie_embeddings.T
            seen = train_user_positives.get(user, set())
            if seen:
                scores[list(seen)] = -1e9
            top_items = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.cpu().tolist()
            true_movie_ids = [spec.index_to_movie[item] for item in positives if item in spec.index_to_movie]
            rec_movie_ids = [spec.index_to_movie[item] for item in top_items if item in spec.index_to_movie]
            rows.append(
                {
                    "test_positive": " | ".join(_titles(true_movie_ids, movies)),
                    "top_recommendations": " | ".join(_titles(rec_movie_ids, movies)),
                }
            )
    if not rows:
        return "No examples were produced.", "本次没有生成样例。"
    frame = pd.DataFrame(rows)
    return frame.to_markdown(index=False), frame.to_markdown(index=False)


def _titles(movie_ids: list[int], movies: pd.DataFrame) -> list[str]:
    """把 movieId 列表转成电影标题。"""

    if not movie_ids:
        return []
    frame = attach_titles(pd.DataFrame({"movieId": movie_ids}), movies)
    return frame["title"].fillna("<unknown>").tolist()


def run_graph_experiment(kind: str, algorithm_dir: Path) -> None:
    """运行一个图推荐实验并写报告。"""

    title, title_zh = MODEL_TITLES[kind]
    args = parse_args(f"Run {title} on MovieLens.")
    sample_ratings = parse_sample_ratings(args.sample_ratings)
    sample_text = sample_ratings_text(sample_ratings)

    print(f"[{kind}] 读取 MovieLens：{sample_text}。")
    data = load_movielens(sample_ratings=sample_ratings)
    edges = build_positive_edges(data.ratings, max_positives_per_user=args.max_positives_per_user)
    train_edges, valid_edges, test_edges = split_edges_by_user_time(edges)
    spec = build_graph_spec(train_edges)
    train_dataset, train_user_positives, user_indices, movie_indices = make_bpr_dataset(train_edges, spec)
    adjacency = build_normalized_adjacency(spec, user_indices, movie_indices)
    valid_user_positives = _positives_by_user(valid_edges, spec)
    test_user_positives = _positives_by_user(test_edges, spec)
    model = build_graph_model(kind, spec.num_users, spec.num_movies, args.embedding_dim, args.num_layers)

    checkpoint_dir = algorithm_dir / "checkpoints" if args.save_checkpoints else None
    if checkpoint_dir is None:
        print(f"[{kind}] 不保存 .pt checkpoint。")
    else:
        print(f"[{kind}] checkpoint 保存目录：{checkpoint_dir}")

    result = train_graph_model(
        model,
        adjacency,
        train_dataset,
        train_user_positives,
        valid_user_positives,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        top_k=args.top_k,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
        force_train=args.force_train,
    )

    test_loss, test_recall, test_ndcg = evaluate_graph_model(result.model, adjacency.to(next(result.model.parameters()).device), test_user_positives, train_user_positives, args.top_k)
    examples, examples_zh = _example_markdown(result.model, adjacency, train_user_positives, test_user_positives, spec, data.movies, args.top_k, args.eval_users)
    checkpoint_md, checkpoint_zh_md = checkpoint_size_markdown(checkpoint_dir)
    examples += "\n\n## Checkpoint size\n\n" + checkpoint_md
    examples_zh += "\n\n## Checkpoint 大小\n\n" + checkpoint_zh_md

    write_report(
        algorithm_dir,
        title,
        title_zh,
        [
            f"- Loaded MovieLens: {sample_text}.",
            "- Built a user-movie bipartite graph from ratings >= 4.0.",
            f"- Kept at most `{args.max_positives_per_user}` recent positive edges per user; use `0` for all positives.",
            f"- Graph device used: `{result.device_name}`.",
        ],
        [
            f"- 读取 MovieLens：{sample_text}。",
            "- 用 rating >= 4.0 构建用户-电影二部图。",
            f"- 每个用户最多保留 `{args.max_positives_per_user}` 条最近正反馈边；传 `0` 可使用全部正反馈。",
            f"- 图训练设备：`{result.device_name}`。",
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
