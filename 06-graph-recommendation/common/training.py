from __future__ import annotations

"""图推荐模型训练循环。

LightGCN/NGCF 都使用 BPR loss：
同一个用户的正样本电影分数应该高于采样负样本电影。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.torch_utils import cleanup_dataloaders, dataloader_kwargs, get_device, seed_everything


@dataclass
class GraphTrainResult:
    """训练结束后 report 需要展示的信息。"""

    model: nn.Module
    best_valid_loss: float
    best_valid_recall: float
    best_valid_ndcg: float
    epochs_ran: int
    device_name: str


def bpr_loss(user_vectors: torch.Tensor, positive_vectors: torch.Tensor, negative_vectors: torch.Tensor) -> torch.Tensor:
    """计算 BPR loss。

    目标是让 `score(user, positive)` 大于 `score(user, negative)`。
    softplus 写法比直接 `-log(sigmoid(diff))` 更数值稳定。
    """

    positive_scores = (user_vectors * positive_vectors).sum(dim=1)
    negative_scores = (user_vectors * negative_vectors).sum(dim=1)
    return torch.nn.functional.softplus(negative_scores - positive_scores).mean()


def evaluate_graph_model(model: nn.Module, adjacency: torch.Tensor, valid_user_positives: dict[int, set[int]], train_user_positives: dict[int, set[int]], top_k: int, max_eval_users: int = 1000) -> tuple[float, float, float]:
    """评估 BPR 验证 loss、Recall@K、NDCG@K。

    为了避免评估过慢，默认最多评估 1000 个用户。
    这不是训练采样，只是本地 report 的评估窗口。
    """

    model.eval()
    losses: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    with torch.no_grad():
        user_embeddings, movie_embeddings = model.propagate(adjacency)
        for count, (user, positives) in enumerate(valid_user_positives.items()):
            if count >= max_eval_users:
                break
            seen = train_user_positives.get(user, set())
            candidates = torch.arange(movie_embeddings.shape[0], device=movie_embeddings.device)
            scores = user_embeddings[user] @ movie_embeddings.T
            if seen:
                scores[list(seen)] = -1e9
            top_items = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.cpu().tolist()
            hit_positions = [rank for rank, item in enumerate(top_items, start=1) if item in positives]
            recalls.append(1.0 if hit_positions else 0.0)
            ndcgs.append(1.0 / np.log2(hit_positions[0] + 1) if hit_positions else 0.0)

            positive = next(iter(positives))
            negative = next((item for item in top_items if item not in positives), top_items[-1])
            loss = bpr_loss(user_embeddings[user : user + 1], movie_embeddings[positive : positive + 1], movie_embeddings[negative : negative + 1])
            losses.append(float(loss.item()))
    if not recalls:
        return 0.0, 0.0, 0.0
    return sum(losses) / len(losses), sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, float]) -> None:
    """保存 checkpoint；默认主线只写 best.pt。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics}, path)


def train_graph_model(
    model: nn.Module,
    adjacency: torch.Tensor,
    train_dataset,
    train_user_positives: dict[int, set[int]],
    valid_user_positives: dict[int, set[int]],
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    top_k: int,
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    keep_checkpoints: int,
    force_train: bool = False,
) -> GraphTrainResult:
    """训练图推荐模型，并用验证集 BPR loss 做 early stopping。"""

    seed_everything(42)
    # torch sparse mm 在 MPS 上支持不稳定，这里图模型默认优先 CUDA，否则 CPU。
    # 这比在 Apple Silicon 上半路报 sparse kernel 错误更可控。
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    adjacency = adjacency.to(device)
    best_path = checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs(device, num_workers),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if best_path is not None and best_path.exists() and not force_train:
        # 06 组图模型的 best.pt 只在训练结束后写入。
        # 检测到它时直接加载，避免重复做图传播和 BPR 训练。
        print(f"[graph] 检测到已有 best checkpoint，跳过训练并加载：{best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        valid_loss, valid_recall, valid_ndcg = evaluate_graph_model(model, adjacency, valid_user_positives, train_user_positives, top_k)
        cleanup_dataloaders(loader)
        return GraphTrainResult(model, valid_loss, valid_recall, valid_ndcg, int(checkpoint.get("epoch", 0)), str(device))

    best_loss = float("inf")
    best_recall = 0.0
    best_ndcg = 0.0
    best_state = None
    stale_epochs = 0
    epochs_ran = 0
    intermediate_paths: list[Path] = []

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()
        progress = tqdm(loader, desc=f"Graph epoch {epoch}", unit="batch")
        for users, positives, negatives in progress:
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            optimizer.zero_grad(set_to_none=True)
            user_embeddings, movie_embeddings = model.propagate(adjacency)
            loss = bpr_loss(user_embeddings[users], movie_embeddings[positives], movie_embeddings[negatives])
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss, valid_recall, valid_ndcg = evaluate_graph_model(model, adjacency, valid_user_positives, train_user_positives, top_k)
        print(f"[graph] epoch={epoch} valid_loss={valid_loss:.4f} recall@{top_k}={valid_recall:.4f} ndcg@{top_k}={valid_ndcg:.4f}")
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_recall = valid_recall
            best_ndcg = valid_ndcg
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if checkpoint_dir is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch-{epoch:04d}.pt"
            _save_checkpoint(checkpoint_path, model, epoch, {"valid_loss": valid_loss, f"valid_recall@{top_k}": valid_recall, f"valid_ndcg@{top_k}": valid_ndcg})
            intermediate_paths.append(checkpoint_path)
            while len(intermediate_paths) > keep_checkpoints:
                old_path = intermediate_paths.pop(0)
                if old_path.exists():
                    old_path.unlink()

        if stale_epochs >= patience:
            print(f"[graph] early stopping at epoch {epoch}.")
            break

    cleanup_dataloaders(loader)
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        _save_checkpoint(checkpoint_dir / "best.pt", model, epochs_ran, {"valid_loss": best_loss, f"valid_recall@{top_k}": best_recall, f"valid_ndcg@{top_k}": best_ndcg})
    return GraphTrainResult(model, best_loss, best_recall, best_ndcg, epochs_ran, str(device))
