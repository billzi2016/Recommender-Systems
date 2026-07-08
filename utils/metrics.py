from __future__ import annotations

"""推荐排序指标工具。

这里放所有实验都会复用的轻量指标函数。
指标函数只接收 Python 列表和集合，不依赖模型框架，
这样传统 CF、PyTorch 召回、深度精排都能复用同一套计算方式。
"""

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


def mean_ranking_metrics(recommendations: dict[int, list[int]], relevant_by_user: dict[int, set[int]], k: int) -> dict[str, float]:
    """计算多用户平均排序指标。

    `recommendations` 是 userId -> 推荐 movieId 列表。
    `relevant_by_user` 是 userId -> 测试集中真实相关 movieId 集合。
    """

    if not recommendations:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0}

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for user_id, recs in recommendations.items():
        # 如果某个用户没有 relevant 集合，三个单用户指标都会自然变成 0。
        # 调用方仍应尽量筛掉无意义用户，否则平均值会被这些用户拉低。
        relevant = relevant_by_user.get(user_id, set())
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    n = len(recommendations)
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
    }
