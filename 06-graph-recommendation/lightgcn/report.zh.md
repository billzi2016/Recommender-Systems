# LightGCN 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 用 rating >= 4.0 构建用户-电影二部图。
- 每个用户最多保留 `50` 条最近正反馈边；传 `0` 可使用全部正反馈。
- 图训练设备：`cpu`。

## 指标

- `best_valid_loss`：`4.9253`
- `best_valid_recall@10`：`0.0700`
- `best_valid_ndcg@10`：`0.0320`
- `test_loss`：`5.1417`
- `test_recall@10`：`0.0560`
- `test_ndcg@10`：`0.0287`
- `epochs_ran`：`15.0000`

## 推荐样例

| test_positive                          | top_recommendations                                                                                                                                                                                                                                                                                                                                                                                   |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ever After: A Cinderella Story (1998)  | Fargo (1996) | Postman, The (Postino, Il) (1994) | Dead Man Walking (1995) | Annie Hall (1977) | Crying Game, The (1992) | Remains of the Day, The (1993) | Like Water for Chocolate (Como agua para chocolate) (1992) | Maltese Falcon, The (1941) | Casablanca (1942) | L.A. Confidential (1997)                                                                                                    |
| Hunchback of Notre Dame, The (1996)    | Apollo 13 (1995) | Dances with Wolves (1990) | Clear and Present Danger (1994) | Braveheart (1995) | Jurassic Park (1993) | Crimson Tide (1995) | Mr. Holland's Opus (1995) | Silence of the Lambs, The (1991) | Dave (1993) | Outbreak (1995)                                                                                                                                                        |
| Sister Act 2: Back in the Habit (1993) | Men in Black (a.k.a. MIB) (1997) | Star Wars: Episode VI - Return of the Jedi (1983) | Forrest Gump (1994) | Shrek (2001) | Sixth Sense, The (1999) | Lord of the Rings: The Fellowship of the Ring, The (2001) | American Pie (1999) | Star Wars: Episode IV - A New Hope (1977) | Back to the Future (1985) | 10 Things I Hate About You (1999)                                                     |
| Stir of Echoes (1999)                  | Star Wars: Episode IV - A New Hope (1977) | Shawshank Redemption, The (1994) | Pulp Fiction (1994) | Star Wars: Episode V - The Empire Strikes Back (1980) | Matrix, The (1999) | Silence of the Lambs, The (1991) | Godfather, The (1972) | Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) | Star Wars: Episode VI - Return of the Jedi (1983) | Forrest Gump (1994) |
| Lion King, The (1994)                  | Apollo 13 (1995) | Dances with Wolves (1990) | Silence of the Lambs, The (1991) | Pulp Fiction (1994) | Jurassic Park (1993) | Shawshank Redemption, The (1994) | Outbreak (1995) | Aladdin (1992) | Speed (1994) | Beauty and the Beast (1991)                                                                                                                                                       |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 58.05 |
