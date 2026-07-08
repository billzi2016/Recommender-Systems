# User-CF 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 将评分大于等于 4.0 的记录当作正反馈。
- 使用 sklearn 的余弦近邻搜索寻找相似用户。

## 指标

- `precision@10`：`0.1000`
- `recall@10`：`0.0019`

## 推荐样例

| title                                                  | genres                              |   score |
|:-------------------------------------------------------|:------------------------------------|--------:|
| Wallace & Gromit: The Best of Aardman Animation (1996) | Adventure|Animation|Comedy          | 4.13275 |
| Enron: The Smartest Guys in the Room (2005)            | Documentary                         | 2.89688 |
| Parasite (2019)                                        | Comedy|Drama                        | 2.62419 |
| Muppets Take Manhattan, The (1984)                     | Children|Comedy|Musical             | 2.14223 |
| People Will Talk (1951)                                | Comedy|Romance                      | 2.11905 |
| Run Silent Run Deep (1958)                             | War                                 | 2.10309 |
| The Irishman (2019)                                    | Crime|Drama                         | 2.09153 |
| Murder on the Orient Express (1974)                    | Crime|Mystery|Thriller              | 1.86089 |
| Knives Out (2019)                                      | Comedy|Crime|Drama|Mystery|Thriller | 1.84716 |
| Christmas Carol, A (1938)                              | Children|Drama|Fantasy              | 1.80194 |
