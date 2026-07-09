# SASRec 实验报告

## 本次运行做了什么

- 读取 MovieLens：2,000,000 条评分。
- 只保留高评分电影，按用户 timestamp 构造正反馈序列。
- 最大序列长度 `50`，每个用户最多 `20` 条训练前缀。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_loss`：`7.7595`
- `best_valid_recall@10`：`0.0307`
- `best_valid_ndcg@10`：`0.0143`
- `test_loss`：`7.9499`
- `test_recall@10`：`0.0260`
- `test_ndcg@10`：`0.0121`
- `epochs_ran`：`11.0000`

## 推荐样例

| history                                                                                                                                                                                                                                             | true_next          | top_recommendations                                                                                                                                                                                                                                                                        |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Emma (1996) -> Living in Oblivion (1995) -> Titanic (1997)                                                                                                                                                                                          | Raging Bull (1980) | Schindler's List (1993) | Pulp Fiction (1994) | Forrest Gump (1994) | Monty Python and the Holy Grail (1975) | Terminator 2: Judgment Day (1991) | Seven (a.k.a. Se7en) (1995) | Toy Story (1995) | Jurassic Park (1993) | Usual Suspects, The (1995) | Die Hard (1988)                    |
| Star Wars: Episode IV - A New Hope (1977) -> Schindler's List (1993) -> While You Were Sleeping (1995) -> Ten Commandments, The (1956) -> Superman III (1983) -> Superman IV: The Quest for Peace (1987) -> Matrix Reloaded, The (2003)             | Shadowlands (1993) | X-Men (2000) | Fifth Element, The (1997) | Terminator 2: Judgment Day (1991) | Sixth Sense, The (1999) | Die Hard (1988) | Shrek (2001) | Kill Bill: Vol. 2 (2004) | Star Wars: Episode VI - Return of the Jedi (1983) | Seven (a.k.a. Se7en) (1995) | Gladiator (2000)                    |
| Alien (1979) -> Blues Brothers, The (1980) -> The Butterfly Effect (2004)                                                                                                                                                                           | Serenity (2005)    | Die Hard (1988) | Fifth Element, The (1997) | Stand by Me (1986) | Monty Python and the Holy Grail (1975) | Star Wars: Episode VI - Return of the Jedi (1983) | Forrest Gump (1994) | Seven (a.k.a. Se7en) (1995) | Toy Story (1995) | Big Lebowski, The (1998) | Kill Bill: Vol. 1 (2003) |
| Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) -> Beauty and the Beast (1991) -> Lion King, The (1994) -> Mulan (1998)                                                                                              | Birds, The (1963)  | Lion King, The (1994) | Fugitive, The (1993) | Forrest Gump (1994) | Braveheart (1995) | Jurassic Park (1993) | Toy Story (1995) | Mask, The (1994) | Shawshank Redemption, The (1994) | While You Were Sleeping (1995) | Pretty Woman (1990)                                              |
| Icarus (2017) -> Jodorowsky's Dune (2013) -> Guard, The (2011) -> Bo Burnham: Inside (2021) -> Laputa: Castle in the Sky (Tenkû no shiro Rapyuta) (1986) -> Ponyo (Gake no ue no Ponyo) (2008) -> News of the World (2020) -> Licorice Pizza (2021) | RRR (2022)         | Get Out (2017) | Machete (2010) | Dune (2021) | A-Team, The (2010) | Karate Kid, The (1984) | Monsters vs. Aliens (2009) | Band of Brothers (2001) | Guardians of the Galaxy 2 (2017) | Darkest Hour (2017) | Green Book (2018)                                                            |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 10.98 |
