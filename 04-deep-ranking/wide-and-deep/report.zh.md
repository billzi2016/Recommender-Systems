# Wide and deep 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。
- 使用特征：用户 ID、电影 ID、电影 genres 和 timestamp 小时段。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_logloss`：`0.5374`
- `best_valid_auc`：`0.8049`
- `best_valid_accuracy`：`0.7268`
- `epochs_ran`：`16.0000`

## 推荐样例

测试集中模型打分最高的几条记录：

|   userId | title                                                     | genres                   |   rating |   predicted_like_probability |
|---------:|:----------------------------------------------------------|:-------------------------|---------:|-----------------------------:|
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001) | Adventure|Fantasy        |      3   |                     0.99902  |
|        6 | Indiana Jones and the Temple of Doom (1984)               | Action|Adventure|Fantasy |      4.5 |                     0.998954 |
|       14 | Crackerjack (2002)                                        | Comedy                   |      5   |                     0.998788 |
|        6 | Dances with Wolves (1990)                                 | Adventure|Drama|Western  |      4   |                     0.998334 |
|        6 | Election (1999)                                           | Comedy                   |      3   |                     0.997379 |
|        6 | Planet of the Apes (1968)                                 | Action|Drama|Sci-Fi      |      3.5 |                     0.997202 |
|        6 | Star Wars: Episode I - The Phantom Menace (1999)          | Action|Adventure|Sci-Fi  |      5   |                     0.997063 |
|       14 | Brewster's Millions (1985)                                | Comedy                   |      5   |                     0.99518  |
|       14 | One Crazy Summer (1986)                                   | Comedy                   |      5   |                     0.994031 |
|       14 | Stars and Bars (1988)                                     | Action|Comedy|Romance    |      4.5 |                     0.991364 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 67.70 |
