# DCN report

## What ran

- Loaded MovieLens: 全量数据.
- Converted ratings into binary labels: `rating >= 4.0` means positive.
- Used user ID, movie ID, movie genres, and timestamp hour bucket.
- DataLoader workers: `8`.
- Device used: `mps`.

## Metrics

- `best_valid_logloss`: `0.5325`
- `best_valid_auc`: `0.8083`
- `best_valid_accuracy`: `0.7293`
- `epochs_ran`: `13.0000`

## Examples

Top scored held-out rows:

|   userId | title                                                     | genres                      |   rating |   predicted_like_probability |
|---------:|:----------------------------------------------------------|:----------------------------|---------:|-----------------------------:|
|       14 | Crackerjack (2002)                                        | Comedy                      |      5   |                     0.999593 |
|       14 | One Crazy Summer (1986)                                   | Comedy                      |      5   |                     0.999573 |
|       14 | Brewster's Millions (1985)                                | Comedy                      |      5   |                     0.998891 |
|       14 | Love Happy (1949)                                         | Comedy                      |      5   |                     0.998311 |
|       14 | Stars and Bars (1988)                                     | Action|Comedy|Romance       |      4.5 |                     0.998215 |
|       14 | Perfectly Normal (1990)                                   | Comedy                      |      5   |                     0.995119 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001) | Adventure|Fantasy           |      3   |                     0.992206 |
|        6 | Dances with Wolves (1990)                                 | Adventure|Drama|Western     |      4   |                     0.99001  |
|        6 | Election (1999)                                           | Comedy                      |      3   |                     0.986681 |
|       50 | Clockwork Orange, A (1971)                                | Crime|Drama|Sci-Fi|Thriller |      4   |                     0.984737 |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 66.66 |
