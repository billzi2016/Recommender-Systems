# 双塔召回 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 使用 PyTorch 训练双塔召回模型，batch 内其他电影作为近似负样本。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `precision@10`：`0.0210`
- `recall@10`：`0.0254`
- `ndcg@10`：`0.0257`
- `best_valid_loss`：`7.2703`

## 推荐样例

样例用户：`1`

最近高评分历史：

| title                           | genres                         |
|:--------------------------------|:-------------------------------|
| Living in Oblivion (1995)       | Comedy                         |
| Opposite of Sex, The (1998)     | Comedy|Drama|Romance           |
| Crimes and Misdemeanors (1989)  | Comedy|Crime|Drama             |
| To Catch a Thief (1955)         | Crime|Mystery|Romance|Thriller |
| Back to the Future (1985)       | Adventure|Comedy|Sci-Fi        |
| Stand by Me (1986)              | Adventure|Drama                |
| Sabrina (1954)                  | Comedy|Romance                 |
| Welcome to the Dollhouse (1995) | Comedy|Drama                   |

召回候选：

| title                            | genres                   |
|:---------------------------------|:-------------------------|
| Sleeper (1973)                   | Comedy|Sci-Fi            |
| Raising Arizona (1987)           | Comedy                   |
| M*A*S*H (a.k.a. MASH) (1970)     | Comedy|Drama|War         |
| Grateful Dead (1995)             | Documentary              |
| Snowriders (1996)                | Documentary              |
| Deliverance (1972)               | Adventure|Drama|Thriller |
| Manchurian Candidate, The (1962) | Crime|Thriller|War       |
| Being There (1979)               | Comedy|Drama             |
| Take the Money and Run (1969)    | Comedy|Crime             |
| Cape Fear (1962)                 | Crime|Drama|Thriller     |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 58.95 |
