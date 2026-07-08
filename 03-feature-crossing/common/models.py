from __future__ import annotations

"""FM、DeepFM、xDeepFM 的 PyTorch 实现。

这些模型共享一个输入形式：每条样本是一组稀疏特征编号。
FM 学二阶交叉，DeepFM 在 FM 旁边加 DNN，xDeepFM 再加一个显式高阶交叉模块。
"""

import torch
from torch import nn


class FactorizationMachine(nn.Module):
    """标准二阶 FM，用 embedding 内积表示任意两个特征的关系。"""

    def __init__(self, num_features: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear = nn.Embedding(num_features + 1, 1, padding_idx=0)
        self.embedding = nn.Embedding(num_features + 1, embedding_dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        linear_term = self.linear(feature_ids).sum(dim=1).squeeze(-1)
        vectors = self.embedding(feature_ids)
        summed = vectors.sum(dim=1)
        squared_sum = summed * summed
        sum_squared = (vectors * vectors).sum(dim=1)
        interactions = 0.5 * (squared_sum - sum_squared).sum(dim=1)
        return self.bias + linear_term + interactions


class DeepFM(nn.Module):
    """FM + MLP。

    FM 负责记住稳定的二阶组合，MLP 负责从所有 embedding 里继续学非线性组合。
    """

    def __init__(self, num_features: int, max_fields: int, embedding_dim: int, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        self.fm = FactorizationMachine(num_features, embedding_dim)
        self.deep_embedding = nn.Embedding(num_features + 1, embedding_dim, padding_idx=0)
        layers: list[nn.Module] = []
        input_dim = max_fields * embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        fm_logits = self.fm(feature_ids)
        deep_input = self.deep_embedding(feature_ids).flatten(start_dim=1)
        deep_logits = self.mlp(deep_input).squeeze(-1)
        return fm_logits + deep_logits


class CINLayer(nn.Module):
    """xDeepFM 里的压缩交叉层。

    它显式计算上一层特征图和原始字段之间的外积，再用 1x1 卷积压缩通道数。
    这里保留核心思想，不把实现扩展成工业级配置系统。
    """

    def __init__(self, previous_fields: int, original_fields: int, output_fields: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(previous_fields * original_fields, output_fields, kernel_size=1)

    def forward(self, x0: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:
        interactions = torch.einsum("bhd,bmd->bhmd", xk, x0)
        batch_size, previous_fields, original_fields, embedding_dim = interactions.shape
        interactions = interactions.reshape(batch_size, previous_fields * original_fields, embedding_dim)
        return torch.relu(self.conv(interactions))


class XDeepFM(nn.Module):
    """简化但保留核心机制的 xDeepFM。"""

    def __init__(
        self,
        num_features: int,
        max_fields: int,
        embedding_dim: int,
        cin_dims: tuple[int, ...] = (16, 16),
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.linear = nn.Embedding(num_features + 1, 1, padding_idx=0)
        self.embedding = nn.Embedding(num_features + 1, embedding_dim, padding_idx=0)
        cin_layers: list[CINLayer] = []
        previous_fields = max_fields
        for output_fields in cin_dims:
            cin_layers.append(CINLayer(previous_fields, max_fields, output_fields))
            previous_fields = output_fields
        self.cin_layers = nn.ModuleList(cin_layers)
        self.cin_output = nn.Linear(sum(cin_dims), 1)

        layers: list[nn.Module] = []
        input_dim = max_fields * embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        linear_logits = self.linear(feature_ids).sum(dim=1).squeeze(-1)
        x0 = self.embedding(feature_ids)
        xk = x0
        cin_outputs = []
        for layer in self.cin_layers:
            xk = layer(x0, xk)
            cin_outputs.append(xk.sum(dim=2))
        cin_logits = self.cin_output(torch.cat(cin_outputs, dim=1)).squeeze(-1)
        deep_logits = self.deep(x0.flatten(start_dim=1)).squeeze(-1)
        return linear_logits + cin_logits + deep_logits


def build_model(kind: str, num_features: int, max_fields: int, embedding_dim: int) -> nn.Module:
    """根据实验名称构建对应模型。"""

    if kind == "fm":
        return FactorizationMachine(num_features=num_features, embedding_dim=embedding_dim)
    if kind == "deepfm":
        return DeepFM(num_features=num_features, max_fields=max_fields, embedding_dim=embedding_dim)
    if kind == "xdeepfm":
        return XDeepFM(num_features=num_features, max_fields=max_fields, embedding_dim=embedding_dim)
    raise ValueError(f"Unknown feature crossing model: {kind}")

