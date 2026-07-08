from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """计算 Precision@K。

    含义：推荐列表前 K 个里，有多少比例是真正相关的物品。
    它关心“推荐出去的东西准不准”。
    """

    items = list(recommended)[:k]
    if not items:
        return 0.0
    return sum(1 for item in items if item in relevant) / len(items)


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """计算 Recall@K。

    含义：用户真实相关的物品里，有多少被前 K 个推荐结果覆盖到了。
    它关心“用户可能喜欢的东西有没有被漏掉”。
    """

    if not relevant:
        return 0.0
    items = list(recommended)[:k]
    return sum(1 for item in items if item in relevant) / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """计算 NDCG@K。

    NDCG 不只看命中数量，还看命中的位置。
    同样命中一个相关电影，排第 1 名比排第 10 名更好。
    """

    items = list(recommended)[:k]
    dcg = 0.0
    for rank, item in enumerate(items, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
