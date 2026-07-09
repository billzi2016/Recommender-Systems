# NGCF 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 用 rating >= 4.0 构建用户-电影二部图。
- 每个用户最多保留 `50` 条最近正反馈边；传 `0` 可使用全部正反馈。
- 图训练设备：`cpu`。

## 指标

- `best_valid_loss`：`1.5886`
- `best_valid_recall@10`：`0.0500`
- `best_valid_ndcg@10`：`0.0251`
- `test_loss`：`1.6412`
- `test_recall@10`：`0.0420`
- `test_ndcg@10`：`0.0233`
- `epochs_ran`：`6.0000`

## 推荐样例

| test_positive                          | top_recommendations                                                                                                                                                                                                                                                                                                                 |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ever After: A Cinderella Story (1998)  | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Shawshank Redemption, The (1994) | Forrest Gump (1994) | Matrix, The (1999) | Terminator 2: Judgment Day (1991) | Seven (a.k.a. Se7en) (1995) | Braveheart (1995) | Lord of the Rings: The Two Towers, The (2002) | Usual Suspects, The (1995)                             |
| Hunchback of Notre Dame, The (1996)    | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Seven (a.k.a. Se7en) (1995) | Matrix, The (1999) | Lord of the Rings: The Two Towers, The (2002) | Terminator 2: Judgment Day (1991) | Braveheart (1995) | Fight Club (1999) | Dark Knight, The (2008) | Fargo (1996)                                                      |
| Sister Act 2: Back in the Habit (1993) | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Shawshank Redemption, The (1994) | Matrix, The (1999) | Forrest Gump (1994) | Seven (a.k.a. Se7en) (1995) | Terminator 2: Judgment Day (1991) | Lord of the Rings: The Two Towers, The (2002) | Dark Knight, The (2008) | Braveheart (1995)                                |
| Stir of Echoes (1999)                  | Shawshank Redemption, The (1994) | Silence of the Lambs, The (1991) | Pulp Fiction (1994) | Forrest Gump (1994) | Seven (a.k.a. Se7en) (1995) | Matrix, The (1999) | Terminator 2: Judgment Day (1991) | Lord of the Rings: The Two Towers, The (2002) | Usual Suspects, The (1995) | Dark Knight, The (2008)                       |
| Lion King, The (1994)                  | Shawshank Redemption, The (1994) | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Matrix, The (1999) | Lord of the Rings: The Two Towers, The (2002) | Dark Knight, The (2008) | Terminator 2: Judgment Day (1991) | Usual Suspects, The (1995) | Seven (a.k.a. Se7en) (1995) | Star Wars: Episode IV - A New Hope (1977) |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 58.11 |
