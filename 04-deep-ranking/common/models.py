from __future__ import annotations

"""04 组深度精排模型。

这一组模型都输出一个 logit，也就是还没过 sigmoid 的分数。
训练时用 `BCEWithLogitsLoss`，它会在内部处理 sigmoid，比手动 sigmoid 后再算
二分类 loss 更稳定。
"""

import torch
from torch import nn

from common.data import HOUR_BUCKETS, RankingSpec


class NCF(nn.Module):
    """NCF：用 MLP 学用户和电影 embedding 的交互。

    矩阵分解把用户向量和电影向量做点积。
    NCF 则把两个向量拼起来交给 MLP，让网络自己学习“怎么组合才算匹配”。
    这更灵活，但也更容易过拟合，所以 MLP 不故意做得很深。
    """

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        """计算一批用户-电影组合的喜欢概率 logit。"""

        users, movies = batch[0], batch[1]
        x = torch.cat([self.user_embedding(users), self.movie_embedding(movies)], dim=1)
        return self.mlp(x).squeeze(-1)


class WideAndDeep(nn.Module):
    """Wide&Deep：wide 侧记简单规则，deep 侧学 embedding 泛化。

    Wide 侧本质上是线性模型：用户、电影、genre、时间段各给一个权重。
    Deep 侧把这些字段变成 embedding 后交给 MLP。
    两边相加得到最终 logit：一边偏“记忆”，一边偏“泛化”。
    """

    def __init__(self, spec: RankingSpec, embedding_dim: int) -> None:
        super().__init__()
        # Wide 分支：每个离散特征直接学习一个标量权重。
        # 这相当于让模型记住“某用户整体更容易高分”“某电影整体更受欢迎”。
        self.user_wide = nn.Embedding(spec.num_users, 1)
        self.movie_wide = nn.Embedding(spec.num_movies, 1)
        self.genre_wide = nn.Embedding(spec.num_genres + 1, 1, padding_idx=0)
        self.hour_wide = nn.Embedding(HOUR_BUCKETS, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # Deep 分支：每个字段变成向量，再拼接给 MLP。
        # 这部分学习的是更软的组合关系，而不是单个字段的固定加分。
        self.user_embedding = nn.Embedding(spec.num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(spec.num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(spec.num_genres + 1, embedding_dim, padding_idx=0)
        self.hour_embedding = nn.Embedding(HOUR_BUCKETS, embedding_dim)
        self.deep = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """把 wide 分数和 deep 分数相加，得到最终 logit。"""

        genre_wide = self.genre_wide(batch["genres"]).sum(dim=1).squeeze(-1)
        wide = (
            self.bias
            + self.user_wide(batch["user"]).squeeze(-1)
            + self.movie_wide(batch["movie"]).squeeze(-1)
            + genre_wide
            + self.hour_wide(batch["hour"]).squeeze(-1)
        )
        # 一个电影可能有多个 genre。这里用平均池化得到一个 genre 向量。
        # 这是简单稳妥的做法，避免为了第一版引入注意力等额外复杂度。
        genre_vec = self.genre_embedding(batch["genres"]).mean(dim=1)
        deep_input = torch.cat(
            [
                self.user_embedding(batch["user"]),
                self.movie_embedding(batch["movie"]),
                genre_vec,
                self.hour_embedding(batch["hour"]),
            ],
            dim=1,
        )
        return wide + self.deep(deep_input).squeeze(-1)


class CrossLayer(nn.Module):
    """DCN 的单层 cross layer。

    Cross layer 的目的不是普通“加深网络”，而是反复把原始输入 x0 和当前表示 x
    做显式混合。这样模型能更直接地表达字段之间的乘性交叉。
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x_{l+1} = x0 * (x_l w_l) + b_l + x_l。"""

        cross = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * cross + self.bias + x


class DCN(nn.Module):
    """Deep & Cross Network：显式 cross network + deep network。

    Cross 分支负责结构化特征交叉，Deep 分支负责普通非线性表达。
    最后把两边拼起来输出一个精排分数。
    """

    def __init__(self, spec: RankingSpec, embedding_dim: int, num_cross_layers: int = 3) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(spec.num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(spec.num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(spec.num_genres + 1, embedding_dim, padding_idx=0)
        self.hour_embedding = nn.Embedding(HOUR_BUCKETS, embedding_dim)
        input_dim = embedding_dim * 4
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim + 64, 1)

    def _dense_input(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """把离散字段统一变成一个 dense 向量。

        DCN 的 cross layer 处理的是连续向量，所以要先完成 embedding 查表。
        """

        genre_vec = self.genre_embedding(batch["genres"]).mean(dim=1)
        return torch.cat(
            [
                self.user_embedding(batch["user"]),
                self.movie_embedding(batch["movie"]),
                genre_vec,
                self.hour_embedding(batch["hour"]),
            ],
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """计算 DCN 的最终 logit。"""

        x0 = self._dense_input(batch)
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(-1)
