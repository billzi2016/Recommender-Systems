from __future__ import annotations

"""04 组深度精排模型的训练循环。

NCF 的 batch 是 tuple，Wide&Deep/DCN 的 batch 是 dict。
为了不写三套几乎一样的训练循环，这里用 `_move_batch` 和 `_labels_from_batch`
把两种 batch 结构统一处理。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.torch_utils import cleanup_dataloaders, dataloader_kwargs, get_device, seed_everything


@dataclass
class RankingTrainResult:
    """训练结束后报告需要的信息。

    report 只写真实训练得到的指标。
    如果用户还没运行实验，算法目录里的 report 会保持“未生成”的占位文本。
    """

    model: nn.Module
    best_valid_loss: float
    best_valid_auc: float
    best_valid_accuracy: float
    epochs_ran: int
    device_name: str


def _move_batch(batch, device: torch.device):
    """把 tuple 或 dict batch 移到目标设备。

    PyTorch 模型和输入 tensor 必须在同一个设备上。
    这里兼容两类数据集，避免在训练循环里写一堆 `if kind == ...`。
    """

    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return tuple(value.to(device) for value in batch)


def _labels_from_batch(batch) -> torch.Tensor:
    """从不同 batch 结构里取出标签。

    NCF 的 label 在 tuple 第三个位置。
    带上下文的模型把 label 放在 dict 的 `label` 字段。
    """

    if isinstance(batch, dict):
        return batch["label"]
    return batch[2]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """评估 logloss、AUC 和 accuracy。

    - logloss：优化目标，适合 early stopping。
    - AUC：看模型能不能把正样本排在负样本前面。
    - accuracy：直观但受正负样本比例影响，所以只作为辅助指标。
    """

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            targets = _labels_from_batch(batch)
            logits = model(batch)
            losses.append(float(criterion(logits, targets).item()))
            labels.append(targets.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    if not labels:
        return 0.0, 0.0, 0.0
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    avg_loss = float(np.mean(losses))
    valid_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    valid_accuracy = float(((y_prob >= 0.5) == y_true).mean())
    try:
        valid_logloss = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        valid_logloss = avg_loss
    return valid_logloss, valid_auc, valid_accuracy


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, float]) -> None:
    """保存 checkpoint；默认只写 best.pt，减少 SSD 写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics}, path)


def train_ranking_model(
    model: nn.Module,
    train_dataset,
    valid_dataset,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    keep_checkpoints: int,
) -> RankingTrainResult:
    """训练深度精排模型，并用验证集 early stopping。

    默认 `max_epochs=1000` 不是说一定要跑 1000 轮。
    真正控制训练长度的是 early stopping：验证集 logloss 连续多轮不提升就停止。

    checkpoint 策略也保持克制：
    - 默认最终只写一次 `best.pt`。
    - `checkpoint_every > 0` 时才写中间文件。
    - 中间文件最多保留 `keep_checkpoints` 个，避免长时间训练把磁盘写爆。
    """

    seed_everything(42)
    device = get_device()
    model = model.to(device)
    # 训练集使用多 worker，提高 batch 准备速度。
    # `dataloader_kwargs` 会在 CUDA 上开启 pin_memory，在 MPS/CPU 上不开。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs(device, num_workers),
    )
    # 验证集不 shuffle，也不需要 persistent worker。
    # 这里保持 num_workers=0，减少评估阶段的多进程复杂度。
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_auc = 0.0
    best_accuracy = 0.0
    best_state = None
    stale_epochs = 0
    epochs_ran = 0
    intermediate_paths: list[Path] = []

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()
        progress = tqdm(train_loader, desc=f"Ranking epoch {epoch}", unit="batch")
        for raw_batch in progress:
            batch = _move_batch(raw_batch, device)
            targets = _labels_from_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss, valid_auc, valid_accuracy = _evaluate(model, valid_loader, device)
        print(f"[ranking] epoch={epoch} valid_logloss={valid_loss:.4f} valid_auc={valid_auc:.4f} valid_accuracy={valid_accuracy:.4f}")

        # 用很小的阈值避免浮点数抖动导致“假提升”。
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_auc = valid_auc
            best_accuracy = valid_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        # 中间 checkpoint 默认关闭。用户显式传 checkpoint_every 时才保存。
        if checkpoint_dir is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch-{epoch:04d}.pt"
            _save_checkpoint(checkpoint_path, model, epoch, {"valid_logloss": valid_loss, "valid_auc": valid_auc, "valid_accuracy": valid_accuracy})
            intermediate_paths.append(checkpoint_path)
            while len(intermediate_paths) > keep_checkpoints:
                old_path = intermediate_paths.pop(0)
                if old_path.exists():
                    old_path.unlink()

        if stale_epochs >= patience:
            print(f"[ranking] early stopping at epoch {epoch}.")
            break

    # 训练结束后主动关闭 DataLoader worker。
    # 这样可以保留 persistent_workers 的训练速度，同时降低脚本结束时卡住的概率。
    cleanup_dataloaders(train_loader, valid_loader)

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        _save_checkpoint(checkpoint_dir / "best.pt", model, epochs_ran, {"valid_logloss": best_loss, "valid_auc": best_auc, "valid_accuracy": best_accuracy})

    return RankingTrainResult(model, best_loss, best_auc, best_accuracy, epochs_ran, str(device))
