# User-CF

User-CF 根据相似用户喜欢过的电影来推荐。

它的直觉很像问一个口味接近的朋友。如果两个用户给很多电影打分都差不多，那么其中一个用户喜欢的电影，就可能推荐给另一个用户。

```mermaid
flowchart LR
  U[目标用户] --> N[找相似用户]
  N --> L[收集邻居喜欢的电影]
  L --> F[过滤目标用户已看过]
  F --> R[得到推荐列表]
```

在 MovieLens 上，可以把每个用户表示成稀疏评分向量。然后用余弦相似度或 Pearson 相关系数计算用户相似度。对目标用户，先找最近邻用户，再收集他们喜欢的电影，最后用邻居相似度加权排序。

第一版代码要控制规模，因为完整的用户-用户相似度矩阵会比较大。可以先抽样，或者只在有共同评分电影的用户之间算相似度。

User-CF 适合用来理解“邻居推荐”的基本想法，但它很怕用户评分太少。这个缺点也能解释为什么后来 Item-CF 和 embedding 方法会更常见。

## 一个很小的用户相似度例子

假设只有三部电影：

| 用户 | The Matrix | Inception | Toy Story |
| --- | --- | --- | --- |
| 用户 A | 5 | 4 | ? |
| 用户 B | 5 | 5 | 2 |
| 用户 C | 1 | 2 | 5 |

目标是给用户 A 推荐电影。用户 A 没看过 Toy Story。

先看口味：

- 用户 B 也喜欢 The Matrix 和 Inception。
- 用户 C 不喜欢 The Matrix 和 Inception，但很喜欢 Toy Story。

所以用户 B 比用户 C 更像用户 A。User-CF 会更相信用户 B 的意见。因为用户 B 给 Toy Story 打了 2 分，系统可能不会把 Toy Story 推荐给用户 A。

如果换一张表：

| 用户 | The Matrix | Inception | Interstellar |
| --- | --- | --- | --- |
| 用户 A | 5 | 4 | ? |
| 用户 B | 5 | 5 | 5 |
| 用户 C | 1 | 2 | 3 |

这次用户 B 和用户 A 相似，而且用户 B 很喜欢 Interstellar，那么 Interstellar 就会成为用户 A 的推荐候选。

## 为什么它容易不稳定

User-CF 依赖用户之间的共同评分。如果两个用户只共同看过一部电影，而且都给了 5 分，你很难说他们口味真的接近。他们可能只是都看过一部特别热门的电影。

所以第一版实现时，可以加一个限制：两个用户至少有一定数量的共同评分电影，才计算相似度。比如共同评分少于 5 部，就先不认为他们是可靠邻居。

## 第一版代码先做什么

1. 读取评分。
2. 过滤评分太少的用户。
3. 构建用户-电影评分矩阵。
4. 对目标用户，找有共同评分的候选邻居。
5. 计算相似度。
6. 取 top K 邻居。
7. 汇总邻居喜欢但目标用户没看过的电影。

## 运行方式

从仓库根目录运行：

```bash
./01-traditional-statistics/user-cf/run.sh --sample-ratings 2000000
```

需要更大样本或全量数据时：

```bash
./01-traditional-statistics/user-cf/run.sh --sample-ratings 5000000
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
```

运行后会在本目录生成 `report.md` 和 `report.zh.md`。

## 读完应该能回答

- User-CF 为什么像“找口味接近的朋友”？
- 为什么共同评分太少会导致相似度不可靠？
- 为什么 User-CF 在线计算压力通常比 Item-CF 大？
