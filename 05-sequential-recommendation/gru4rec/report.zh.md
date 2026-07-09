# GRU4Rec 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 只保留高评分电影，按用户 timestamp 构造正反馈序列。
- 最大序列长度 `50`，每个用户最多 `20` 条训练前缀。
- DataLoader 使用 `8` 个 worker。
- 本次使用设备：`mps`。

## 指标

- `best_valid_loss`：`6.3296`
- `best_valid_recall@10`：`0.1616`
- `best_valid_ndcg@10`：`0.0860`
- `test_loss`：`6.5610`
- `test_recall@10`：`0.1441`
- `test_ndcg@10`：`0.0769`
- `epochs_ran`：`35.0000`

## 推荐样例

| history                                                                                                                                                                                                                                                         | true_next                              | top_recommendations                                                                                                                                                                                                                                                                       |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Three Colors: Blue (Trois couleurs: Bleu) (1993) -> American History X (1998) -> Queen Margot (Reine Margot, La) (1994) -> Nelly & Monsieur Arnaud (1995) -> Midnight Cowboy (1969) -> Citizen Ruth (1996) -> Dangerous Liaisons (1988) -> All About Eve (1950) | Ever After: A Cinderella Story (1998)  | Secrets & Lies (1996) | Graduate, The (1967) | Cinema Paradiso (Nuovo cinema Paradiso) (1989) | Room with a View, A (1986) | Shine (1996) | Ice Storm, The (1997) | Crying Game, The (1992) | Dangerous Liaisons (1988) | My Left Foot (1989) | Amadeus (1984)                            |
| Circle of Friends (1995) -> Three Musketeers, The (1993) -> Walk in the Clouds, A (1995) -> Billy Madison (1995) -> Snow White and the Seven Dwarfs (1937) -> Robin Hood: Men in Tights (1993) -> Aristocats, The (1970) -> Jungle Book, The (1994)             | Hunchback of Notre Dame, The (1996)    | Pinocchio (1940) | Aristocats, The (1970) | Secret Garden, The (1993) | Man Without a Face, The (1993) | Jungle Book, The (1994) | Free Willy (1993) | Beverly Hillbillies, The (1993) | Indian in the Cupboard, The (1995) | Paper, The (1994) | Free Willy 2: The Adventure Home (1995) |
| Enemy of the State (1998) -> Mr. Holland's Opus (1995) -> Misérables, Les (1998) -> Importance of Being Earnest, The (2002) -> Othello (1995) -> Gods Must Be Crazy, The (1980) -> Shadowlands (1993) -> Chariots of Fire (1981)                                | Sister Act 2: Back in the Habit (1993) | Ever After: A Cinderella Story (1998) | Jerry Maguire (1996) | Sliding Doors (1998) | Fried Green Tomatoes (1991) | Titanic (1997) | Driving Miss Daisy (1989) | Forrest Gump (1994) | October Sky (1999) | My Big Fat Greek Wedding (2002) | Few Good Men, A (1992)                      |
| Patton (1970) -> Indiana Jones and the Temple of Doom (1984) -> Bowfinger (1999) -> Clerks (1994) -> Galaxy Quest (1999)                                                                                                                                        | Stir of Echoes (1999)                  | Fight Club (1999) | Election (1999) | Maltese Falcon, The (1941) | Galaxy Quest (1999) | Elizabeth (1998) | Kingpin (1996) | Ghostbusters (a.k.a. Ghost Busters) (1984) | Office Space (1999) | Gladiator (2000) | Shawshank Redemption, The (1994)                                       |
| Fugitive, The (1993) -> Crimson Tide (1995) -> Cliffhanger (1993) -> GoldenEye (1995) -> Braveheart (1995) -> Forrest Gump (1994) -> Firm, The (1993) -> Jurassic Park (1993)                                                                                   | Lion King, The (1994)                  | Terminator 2: Judgment Day (1991) | Speed (1994) | Seven (a.k.a. Se7en) (1995) | Lion King, The (1994) | Firm, The (1993) | Jurassic Park (1993) | Braveheart (1995) | Mask, The (1994) | Disclosure (1994) | Babe (1995)                                                                 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 27.25 |
