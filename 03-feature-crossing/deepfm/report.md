# DeepFM report

## What ran

- Loaded MovieLens: 全量数据.
- Converted ratings into binary labels: `rating >= 4.0` means positive.
- Used user ID, movie ID, movie genres, and timestamp hour bucket as sparse features.
- DataLoader workers: `8`.
- Device used: `mps`.

## Metrics

- `best_valid_logloss`: `0.6096`
- `best_valid_auc`: `0.7784`
- `epochs_ran`: `14.0000`

## Examples

Top scored held-out rows:

|   userId | title                                                     | genres                   |   rating |   predicted_like_probability |
|---------:|:----------------------------------------------------------|:-------------------------|---------:|-----------------------------:|
|       14 | Stars and Bars (1988)                                     | Action|Comedy|Romance    |      4.5 |                     0.999995 |
|       61 | Destiny Turns on the Radio (1995)                         | Comedy                   |      2   |                     0.999961 |
|       46 | TT3D: Closer to the Edge (2011)                           | Documentary              |      5   |                     0.999904 |
|       35 | Jodorowsky's Dune (2013)                                  | Documentary|Sci-Fi       |      4.5 |                     0.999605 |
|       14 | Crackerjack (2002)                                        | Comedy                   |      5   |                     0.999587 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001) | Adventure|Fantasy        |      3   |                     0.99946  |
|        6 | Star Wars: Episode I - The Phantom Menace (1999)          | Action|Adventure|Sci-Fi  |      5   |                     0.998691 |
|        6 | Indiana Jones and the Temple of Doom (1984)               | Action|Adventure|Fantasy |      4.5 |                     0.998679 |
|       44 | ARQ (2016)                                                | Sci-Fi|Thriller          |      3   |                     0.998497 |
|       44 | Paper Planes (2014)                                       | Children                 |      3   |                     0.998333 |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 71.79 |
