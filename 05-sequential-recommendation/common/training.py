from __future__ import annotations

"""序列推荐模型的训练、评估和 checkpoint 收尾。

GRU4Rec 和 SASRec 都是 next item prediction：
输入历史序列，输出下一部电影的分类 logits。
因此训练循环可以共用，模型差异只体现在 forward 内部。
"""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.torch_utils import cleanup_dataloaders, dataloader_kwargs, get_device, seed_everything


@dataclass
class SequentialTrainResult:
    """训练结束后 report 需要展示的结果。"""

    model: nn.Module
    best_valid_loss: float
    best_valid_recall: float
    best_valid_ndcg: float
    epochs_ran: int
    device_name: str


def _ranking_metrics(logits: torch.Tensor, targets: torch.Tensor, k: int) -> tuple[float, float]:
    """计算一个 batch 的 Recall@K 和 NDCG@K。

    每条样本只有一个真实下一部电影，所以：
    - 命中 top-k 则 Recall@K 为 1，否则为 0。
    - NDCG@K 根据命中位置折扣，排第 1 名得分最高。
    """

    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    hits = topk.eq(targets.unsqueeze(1))
    recall = hits.any(dim=1).float().mean().item()
    discounts = torch.zeros(targets.shape[0], device=logits.device)
    if hits.any():
        hit_rows, hit_cols = torch.where(hits)
        discounts[hit_rows] = 1.0 / torch.log2(hit_cols.float() + 2.0)
    ndcg = discounts.mean().item()
    return recall, ndcg


def _mask_padding_for_ranking(logits: torch.Tensor) -> torch.Tensor:
    """只在排序指标里屏蔽 padding item。

    0 号 item 是 padding，不是真实电影，所以 Recall/NDCG 的 top-k 里不能出现它。
    但 CrossEntropyLoss 不需要提前把 0 号 logit 改成极大负数：
    真实 target 永远不是 0，原始 logits 已经可以正常训练。

    之前把 `logits[:, 0] = -1e9` 放在 loss 前面，MPS 上大词表 softmax
    偶尔会把验证 loss 算成 NaN。这里 clone 一份只给 top-k 用，避免污染 loss 路径。
    """

    ranking_logits = logits.clone()
    ranking_logits[:, 0] = torch.finfo(ranking_logits.dtype).min
    return ranking_logits


def evaluate_sequence_model(model: nn.Module, loader: DataLoader, device: torch.device, k: int) -> tuple[float, float, float]:
    """评估验证/测试集 loss、Recall@K、NDCG@K。"""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = model(sequences)
            losses.append(float(criterion(logits, targets).item()))
            # loss 用原始 logits；排序指标才屏蔽 padding item。
            recall, ndcg = _ranking_metrics(_mask_padding_for_ranking(logits), targets, k)
            recalls.append(recall)
            ndcgs.append(ndcg)
    if not losses:
        return 0.0, 0.0, 0.0
    return sum(losses) / len(losses), sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, float]) -> None:
    """保存一个 checkpoint。默认主线只写 best.pt。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics}, path)


def _checkpoint_metrics_are_finite(checkpoint: dict) -> bool:
    """判断 checkpoint 里的指标是否可复用。

    如果一次训练已经产生 NaN/inf，再把这样的 best.pt 当作“已完成训练”复用，
    后面只会反复生成坏报告。这里读取保存的 metrics 做一道轻量防线：
    指标必须存在，并且每个值都是有限数字，才允许跳过训练。
    """

    metrics = checkpoint.get("metrics", {})
    if not metrics:
        return False
    return all(math.isfinite(float(value)) for value in metrics.values())


def train_sequence_model(
    model: nn.Module,
    train_dataset,
    valid_dataset,
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
) -> SequentialTrainResult:
    """训练序列推荐模型，并用验证集 loss 做 early stopping。

    训练目标是多分类 CrossEntropy：在所有电影里预测真实下一部电影。
    这比手写负采样更直接，也更接近“next token prediction”的直觉。
    """

    seed_everything(42)
    device = get_device()
    model = model.to(device)
    best_path = checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs(device, num_workers),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    if best_path is not None and best_path.exists() and not force_train:
        # 05 组 best.pt 只在训练结束后保存。
        # 如果 report 或测试阶段失败，下一次运行可以直接复用 best.pt，不必重训序列模型。
        checkpoint = torch.load(best_path, map_location=device)
        if _checkpoint_metrics_are_finite(checkpoint):
            print(f"[sequential] 检测到已有 best checkpoint，跳过训练并加载：{best_path}")
            model.load_state_dict(checkpoint["model_state"])
            valid_loss, valid_recall, valid_ndcg = evaluate_sequence_model(model, valid_loader, device, top_k)
            cleanup_dataloaders(train_loader, valid_loader)
            return SequentialTrainResult(model, valid_loss, valid_recall, valid_ndcg, int(checkpoint.get("epoch", 0)), str(device))
        print(f"[sequential] 检测到已有 checkpoint 指标不是有限数字，将忽略并重新训练：{best_path}")

    best_loss = float("inf")
    best_recall = 0.0
    best_ndcg = 0.0
    best_state = None
    stale_epochs = 0
    epochs_ran = 0
    intermediate_paths: list[Path] = []

    try:
        for epoch in range(1, max_epochs + 1):
            epochs_ran = epoch
            model.train()
            progress = tqdm(train_loader, desc=f"Sequential epoch {epoch}", unit="batch")
            for sequences, targets in progress:
                sequences = sequences.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(sequences)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("训练 loss 出现 NaN/inf，已停止以避免保存坏 checkpoint。")
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            valid_loss, valid_recall, valid_ndcg = evaluate_sequence_model(model, valid_loader, device, top_k)
            print(f"[sequential] epoch={epoch} valid_loss={valid_loss:.4f} recall@{top_k}={valid_recall:.4f} ndcg@{top_k}={valid_ndcg:.4f}")
            if not math.isfinite(valid_loss):
                raise RuntimeError("验证 loss 出现 NaN/inf，已停止以避免保存坏 checkpoint 和坏报告。")

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
                print(f"[sequential] early stopping at epoch {epoch}.")
                break
    except KeyboardInterrupt:
        # macOS + 多 worker + persistent_workers 时，中断训练最容易留下 semaphore。
        # 这里先给用户一个明确提示，再把 KeyboardInterrupt 原样抛出，让 shell 仍然知道本次被中断。
        print("[sequential] 收到中断信号，正在清理 DataLoader worker。")
        raise
    finally:
        # 无论是正常结束、early stopping，还是 Ctrl-C，都尽量主动关闭 DataLoader worker。
        # 这正是之前 SASRec 中断后退出慢、resource_tracker 报 leaked semaphore 的关键修复。
        cleanup_dataloaders(train_loader, valid_loader)

    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("没有得到有限的最佳验证指标，本次不保存 best.pt。")
    model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        _save_checkpoint(checkpoint_dir / "best.pt", model, epochs_ran, {"valid_loss": best_loss, f"valid_recall@{top_k}": best_recall, f"valid_ndcg@{top_k}": best_ndcg})
    return SequentialTrainResult(model, best_loss, best_recall, best_ndcg, epochs_ran, str(device))
