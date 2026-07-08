# Item-CF 实验报告

## 本次运行做了什么

- 从 MovieLens 32M 读取：全量数据。
- 将评分大于等于 4.0 的记录当作正反馈。
- 使用 sklearn 的近邻模型计算电影之间的余弦相似度。

## 指标

- `precision@10`：`0.0000`
- `recall@10`：`0.0000`

## 推荐样例

| title                                             | genres               |    score |
|:--------------------------------------------------|:---------------------|---------:|
| You Killed Me First (1985)                        | Drama                | 13.8645  |
| Strip-tease (1963)                                | Drama                | 13.6366  |
| Failure (2013)                                    | Comedy               | 13.2233  |
| A Boring Story (1983)                             | (no genres listed)   |  8.08139 |
| Dialogues with Solzhenitsyn (Uzel) (1999)         | Documentary          |  7.68901 |
| The Third Annual 'On Cinema' Oscar Special (2015) | Comedy               |  6.81117 |
| Life of Her Own, A (1950)                         | Drama                |  6.36396 |
| Fatty Drives the Bus (1999)                       | Comedy               |  6.36396 |
| Lost on Journey (2010)                            | Comedy               |  6.24919 |
| Special 26 (2013)                                 | Crime|Drama|Thriller |  6.16595 |
