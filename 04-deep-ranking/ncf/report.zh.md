# NCF 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。
- 使用特征：用户 ID 和电影 ID。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_logloss`：`0.5308`
- `best_valid_auc`：`0.8091`
- `best_valid_accuracy`：`0.7302`
- `epochs_ran`：`16.0000`

## 推荐样例

测试集中模型打分最高的几条记录：

|   userId | title                                                                     | genres                      |   rating |   predicted_like_probability |
|---------:|:--------------------------------------------------------------------------|:----------------------------|---------:|-----------------------------:|
|       14 | Crackerjack (2002)                                                        | Comedy                      |      5   |                     0.991771 |
|       50 | Cool Hand Luke (1967)                                                     | Drama                       |      4.5 |                     0.98562  |
|       40 | Seven Samurai (Shichinin no samurai) (1954)                               | Action|Adventure|Drama      |      5   |                     0.983853 |
|       14 | One Crazy Summer (1986)                                                   | Comedy                      |      5   |                     0.981065 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001)                 | Adventure|Fantasy           |      3   |                     0.979518 |
|       50 | Clockwork Orange, A (1971)                                                | Crime|Drama|Sci-Fi|Thriller |      4   |                     0.97814  |
|       50 | Diving Bell and the Butterfly, The (Scaphandre et le papillon, Le) (2007) | Drama                       |      4.5 |                     0.977615 |
|       50 | Trainspotting (1996)                                                      | Comedy|Crime|Drama          |      4   |                     0.976853 |
|       14 | Perfectly Normal (1990)                                                   | Comedy                      |      5   |                     0.976074 |
|       50 | Sleuth (1972)                                                             | Comedy|Mystery|Thriller     |      4.5 |                     0.975508 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 66.59 |
