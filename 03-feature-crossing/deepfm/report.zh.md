# DeepFM 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。
- 使用用户 ID、电影 ID、电影类型、时间段作为稀疏特征。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_logloss`：`0.6096`
- `best_valid_auc`：`0.7784`
- `epochs_ran`：`14.0000`

## 推荐样例

测试集中模型打分最高的几条记录：

|   userId | title                                                     | genres                   |   rating |   predicted_like_probability |
|---------:|:----------------------------------------------------------|:-------------------------|---------:|-----------------------------:|
|       14 | Stars and Bars (1988)                                     | Action|Comedy|Romance    |      4.5 |                     0.999995 |
|       61 | Destiny Turns on the Radio (1995)                         | Comedy                   |      2   |                     0.999961 |
|       46 | TT3D: Closer to the Edge (2011)                           | Documentary              |      5   |                     0.999904 |
|       35 | Jodorowsky's Dune (2013)                                  | Documentary|Sci-Fi       |      4.5 |                     0.999605 |
|       14 | Crackerjack (2002)                                        | Comedy                   |      5   |                     0.999587 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001) | Adventure|Fantasy        |      3   |                     0.99946  |
|        6 | Star Wars: Episode I - The Phantom Menace (1999)          | Action|Adventure|Sci-Fi  |      5   |                     0.998691 |
|        6 | Indiana Jones and the Temple of Doom (1984)               | Action|Adventure|Fantasy |      4.5 |                     0.998679 |
|       44 | ARQ (2016)                                                | Sci-Fi|Thriller          |      3   |                     0.998497 |
|       44 | Paper Planes (2014)                                       | Children                 |      3   |                     0.998333 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 71.79 |
