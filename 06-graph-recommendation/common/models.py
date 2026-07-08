from __future__ import annotations

"""LightGCN 和 NGCF 的 PyTorch 模型。

这两个模型都在用户-电影二部图上传播 embedding。
区别是：
- LightGCN 只做线性邻居聚合，并平均多层 embedding。
- NGCF 在传播时加入线性变换、非线性和特征交互项。
"""

import torch
from torch import nn


class LightGCN(nn.Module):
    """LightGCN 图推荐模型。"""

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.num_layers = num_layers
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """在二部图上做多层 embedding 传播。

        LightGCN 不加 MLP，不加激活函数。
        每一层只是把邻居 embedding 按归一化邻接矩阵聚合过来。
        """

        all_embeddings = torch.cat([self.user_embedding.weight, self.movie_embedding.weight], dim=0)
        layer_outputs = [all_embeddings]
        current = all_embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(adjacency, current)
            layer_outputs.append(current)
        final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return final[: self.user_embedding.num_embeddings], final[self.user_embedding.num_embeddings :]


class NGCF(nn.Module):
    """NGCF 图推荐模型。

    NGCF 比 LightGCN 多了两类变换：
    - 邻居消息的线性变换。
    - 当前节点和邻居消息的逐元素交互。
    """

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.message_layers = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])
        self.bi_layers = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 NGCF 消息传递。"""

        all_embeddings = torch.cat([self.user_embedding.weight, self.movie_embedding.weight], dim=0)
        outputs = [all_embeddings]
        current = all_embeddings
        for message_layer, bi_layer in zip(self.message_layers, self.bi_layers, strict=False):
            neighbor = torch.sparse.mm(adjacency, current)
            message = message_layer(neighbor)
            interaction = bi_layer(current * neighbor)
            current = self.dropout(torch.nn.functional.leaky_relu(message + interaction, negative_slope=0.2))
            outputs.append(torch.nn.functional.normalize(current, dim=1))
        final = torch.stack(outputs, dim=0).mean(dim=0)
        return final[: self.user_embedding.num_embeddings], final[self.user_embedding.num_embeddings :]


def build_graph_model(kind: str, num_users: int, num_movies: int, embedding_dim: int, num_layers: int) -> nn.Module:
    """根据实验名称创建图推荐模型。"""

    if kind == "lightgcn":
        return LightGCN(num_users, num_movies, embedding_dim, num_layers)
    if kind == "ngcf":
        return NGCF(num_users, num_movies, embedding_dim, num_layers)
    raise ValueError(f"Unknown graph model: {kind}")

