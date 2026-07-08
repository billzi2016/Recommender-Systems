# FM

FM，也就是因子分解机，用来学习稀疏特征里的二阶交叉。

协同过滤主要用用户 ID 和电影 ID。FM 保留这个信号，同时还能加入电影类型、时间段、用户属性等特征。每个特征都有一个 embedding。模型预测时，一部分来自线性项，另一部分来自任意两个特征 embedding 的点积。

你可以把 FM 理解成一个更会“联想”的线性模型。普通线性模型只能学到“用户 42 本身有什么倾向”“电影 99 本身有什么倾向”。FM 还能学到“用户 42 和科幻类型放在一起时有什么倾向”“电影 99 和晚上这个时间段放在一起时有什么倾向”。

```mermaid
flowchart LR
  X[稀疏特征<br/>userId movieId genres time] --> E[每个特征查 embedding]
  E --> Pair[两两特征做点积]
  Pair --> Y[预测评分或点击概率]
```

在 MovieLens 上，一条训练样本可以包含：

- 用户 ID
- 电影 ID
- 电影 genres
- 评分时间段

FM 适合宽而稀疏的特征。它能学到某个用户和某类电影之间的关系，也能学到某部电影和某个时间段之间的模式，而不用手写所有交叉规则。

第一版建议和矩阵分解用同一套切分做对比，看加了 genres 以后有没有改善。

## 一条 MovieLens 样本怎么变成特征

原始数据可能是：

| 字段 | 值 |
| --- | --- |
| userId | 42 |
| movieId | 2571 |
| title | The Matrix |
| genres | Action, Sci-Fi |
| rating | 5.0 |
| timestamp | 2009-01-07 21:30 |

FM 看到的不是标题文本，而是一组离散特征：

```text
user=42
movie=2571
genre=Action
genre=Sci-Fi
hour=night
```

这些特征会被放进一个很长的稀疏向量。大部分位置都是 0，只有对应特征的位置是 1。

## FM 学到的交叉长什么样

FM 会考虑任意两个特征之间的关系，比如：

| 特征 1 | 特征 2 | 可能学到的含义 |
| --- | --- | --- |
| user=42 | genre=Sci-Fi | 用户 42 是否偏爱科幻 |
| user=42 | movie=2571 | 用户 42 是否特别喜欢 The Matrix |
| movie=2571 | hour=night | The Matrix 是否常在晚上被高分观看 |
| genre=Action | genre=Sci-Fi | 动作加科幻这个组合是否容易高分 |

如果你手写这些交叉，会很快爆炸。用户有几十万，电影有几万，类型和时间也不少。FM 的做法是给每个特征一个 embedding，用 embedding 点积来表示两个特征的交互强度。

## 和矩阵分解有什么关系

如果 FM 只使用 `userId` 和 `movieId`，它就很像矩阵分解。因为它主要学的是用户 embedding 和电影 embedding 的交互。

当你加入 genres、时间段、用户属性后，FM 就比普通矩阵分解更灵活。它不只记住“用户喜欢这部电影”，还能学到“用户喜欢这类电影”。

## 第一版实验怎么设计

建议做三组对比：

1. 只用 userId 和 movieId。
2. 加 movie genres。
3. 再加 timestamp 转出来的 hour 或 weekday。

如果第二组比第一组好，说明 genres 有帮助。如果第三组没提升，也不奇怪。MovieLens 的时间戳是评分时间，不一定等于真实观看时间。

## 运行

默认全量运行：

```bash
./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

想先快速试跑：

```bash
./03-feature-crossing/fm/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

默认命令只保存 `checkpoints/best.pt`。报告会写入验证指标、测试集预测样例和 checkpoint 大小。

## 常见坑

不要把 title 直接当成普通类别特征乱塞进去。电影标题几乎等价于 movieId，第一版意义不大。

不要一次性加太多特征。特征越多，越难判断到底是谁带来了提升。

不要忘记处理多值 genres。一部电影可能同时是 `Action|Sci-Fi`，这应该拆成两个 genre 特征。
