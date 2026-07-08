from __future__ import annotations

"""特征交叉模型的训练和评估循环。

FM、DeepFM、xDeepFM 的输入形式一致，训练目标也一致。
因此这里共用一套训练循环，避免三个算法目录复制大段代码。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.torch_utils import dataloader_kwargs, get_device, seed_everything


@dataclass
class TrainingResult:
    """训练结束后报告需要用到的信息。

    这里只保存 report 需要展示的最小信息：
    最佳验证 loss、AUC、实际训练轮数和设备名称。
    """

    model: nn.Module
    best_valid_loss: float
    best_valid_auc: float
    device_name: str
    epochs_ran: int


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """计算验证集 logloss、AUC 和 accuracy。

    logloss 用来 early stopping。
    AUC 更接近排序视角：正样本是否排在负样本前面。
    accuracy 直观但会受正负样本比例影响，只作为辅助观察。
    """

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for feature_ids, targets in loader:
            feature_ids = feature_ids.to(device)
            targets = targets.to(device)
            logits = model(feature_ids)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    valid_loss = float(np.mean(losses)) if losses else 0.0
    valid_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    valid_accuracy = float(((y_prob >= 0.5) == y_true).mean())
    try:
        valid_logloss = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        valid_logloss = valid_loss
    return valid_logloss, valid_auc, valid_accuracy


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, float]) -> None:
    """保存一个小而完整的 PyTorch checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics}, path)


def train_feature_crossing_model(
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
) -> TrainingResult:
    """训练 FM 系列模型。

    默认最多跑 1000 轮，但实际由 early stopping 控制。验证集 loss 连续多轮
    没有变好时会提前停止，避免为了“轮数很多”而浪费时间。
    """

    seed_everything(42)
    device = get_device()
    model = model.to(device)
    # 训练集用多 worker 准备 batch；pin_memory 只会在 CUDA 上开启。
    # MPS/CPU 不强行 pin，避免增加内存压力。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs(device, num_workers),
    )
    # 验证集不 shuffle，且不需要 persistent worker。
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_auc = 0.0
    best_state = None
    stale_epochs = 0
    intermediate_paths: list[Path] = []
    epochs_ran = 0

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}", unit="batch")
        for feature_ids, targets in progress:
            feature_ids = feature_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_ids)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_loss, valid_auc, valid_accuracy = _evaluate(model, valid_loader, device)
        print(f"[feature-crossing] epoch={epoch} valid_logloss={valid_loss:.4f} valid_auc={valid_auc:.4f} valid_accuracy={valid_accuracy:.4f}")

        # FM 系列 loss 可能有轻微抖动，这里用严格小于作为“变好”的判断。
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_auc = valid_auc
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        # 中间 checkpoint 默认关闭；用户显式设置 checkpoint_every 才会保存。
        if checkpoint_dir is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch-{epoch:04d}.pt"
            _save_checkpoint(checkpoint_path, model, epoch, {"valid_logloss": valid_loss, "valid_auc": valid_auc, "valid_accuracy": valid_accuracy})
            intermediate_paths.append(checkpoint_path)
            while len(intermediate_paths) > keep_checkpoints:
                old_path = intermediate_paths.pop(0)
                if old_path.exists():
                    old_path.unlink()

        if stale_epochs >= patience:
            print(f"[feature-crossing] early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        # best.pt 只在训练结束后写一次，避免频繁覆盖 SSD。
        _save_checkpoint(checkpoint_dir / "best.pt", model, epochs_ran, {"valid_logloss": best_loss, "valid_auc": best_auc})

    return TrainingResult(model=model, best_valid_loss=best_loss, best_valid_auc=best_auc, device_name=str(device), epochs_ran=epochs_ran)
