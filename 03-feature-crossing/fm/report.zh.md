# FM 因子分解机 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 把评分转成二分类标签：`rating >= 4.0` 表示喜欢。
- 使用用户 ID、电影 ID、电影类型、时间段作为稀疏特征。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_logloss`：`0.6081`
- `best_valid_auc`：`0.7810`
- `epochs_ran`：`15.0000`

## 推荐样例

测试集中模型打分最高的几条记录：

|   userId | title                                     | genres                   |   rating |   predicted_like_probability |
|---------:|:------------------------------------------|:-------------------------|---------:|-----------------------------:|
|       35 | Wrinkles (Arrugas) (2011)                 | Animation|Drama          |      4   |                     0.99996  |
|       63 | Running Scared (2006)                     | Action|Crime|Thriller    |      5   |                     0.999911 |
|       33 | Anvil! The Story of Anvil (2008)          | Documentary|Musical      |      4.5 |                     0.999707 |
|       35 | Capricious Summer (Rozmarné léto) (1968)  | Comedy                   |      4.5 |                     0.999692 |
|       65 | Under Fire (1983)                         | Drama|Thriller|War       |      4   |                     0.999205 |
|       35 | Chaser, The (Chugyeogja) (2008)           | Crime|Drama|Thriller     |      3   |                     0.998928 |
|       33 | Rififi (Du rififi chez les hommes) (1955) | Crime|Film-Noir|Thriller |      5   |                     0.998568 |
|       35 | The Red Turtle (2016)                     | Animation                |      3.5 |                     0.998208 |
|       50 | Man Who Would Be King, The (1975)         | Adventure|Drama          |      4.5 |                     0.998194 |
|       33 | Killing, The (1956)                       | Crime|Film-Noir          |      4.5 |                     0.998186 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 36.33 |
