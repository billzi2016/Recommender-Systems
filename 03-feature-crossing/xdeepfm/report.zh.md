# xDeepFM 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。
- 使用用户 ID、电影 ID、电影类型、时间段作为稀疏特征。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_logloss`：`0.5483`
- `best_valid_auc`：`0.7956`
- `epochs_ran`：`8.0000`

## 推荐样例

测试集中模型打分最高的几条记录：

|   userId | title                                                                     | genres                                |   rating |   predicted_like_probability |
|---------:|:--------------------------------------------------------------------------|:--------------------------------------|---------:|-----------------------------:|
|       14 | Crackerjack (2002)                                                        | Comedy                                |      5   |                     0.989885 |
|       14 | One Crazy Summer (1986)                                                   | Comedy                                |      5   |                     0.988509 |
|       14 | Perfectly Normal (1990)                                                   | Comedy                                |      5   |                     0.981847 |
|       45 | Star Wars: Episode V - The Empire Strikes Back (1980)                     | Action|Adventure|Sci-Fi               |      4   |                     0.980945 |
|       63 | Matrix, The (1999)                                                        | Action|Sci-Fi|Thriller                |      4   |                     0.979752 |
|       33 | Lives of Others, The (Das leben der Anderen) (2006)                       | Drama|Romance|Thriller                |      5   |                     0.979576 |
|       33 | Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) | Action|Adventure|Western              |      5   |                     0.977838 |
|       63 | City of God (Cidade de Deus) (2002)                                       | Action|Adventure|Crime|Drama|Thriller |      4.5 |                     0.977327 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001)                 | Adventure|Fantasy                     |      3   |                     0.976965 |
|       14 | Brewster's Millions (1985)                                                | Comedy                                |      5   |                     0.976499 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 36.59 |
