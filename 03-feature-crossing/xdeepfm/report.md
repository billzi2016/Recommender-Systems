# xDeepFM report

## What ran

- Loaded MovieLens: 全量数据.
- Converted ratings into binary labels: `rating >= 4.0` means positive.
- Used user ID, movie ID, movie genres, and timestamp hour bucket as sparse features.
- DataLoader workers: `8`.
- Device used: `mps`.

## Metrics

- `best_valid_logloss`: `0.5483`
- `best_valid_auc`: `0.7956`
- `epochs_ran`: `8.0000`

## Examples

Top scored held-out rows:

|   userId | title                                                                     | genres                                |   rating |   predicted_like_probability |
|---------:|:--------------------------------------------------------------------------|:--------------------------------------|---------:|-----------------------------:|
|       14 | Crackerjack (2002)                                                        | Comedy                                |      5   |                     0.989885 |
|       14 | One Crazy Summer (1986)                                                   | Comedy                                |      5   |                     0.988509 |
|       14 | Perfectly Normal (1990)                                                   | Comedy                                |      5   |                     0.981847 |
|       45 | Star Wars: Episode V - The Empire Strikes Back (1980)                     | Action|Adventure|Sci-Fi               |      4   |                     0.980945 |
|       63 | Matrix, The (1999)                                                        | Action|Sci-Fi|Thriller                |      4   |                     0.979752 |
|       33 | Lives of Others, The (Das leben der Anderen) (2006)                       | Drama|Romance|Thriller                |      5   |                     0.979576 |
|       33 | Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966) | Action|Adventure|Western              |      5   |                     0.977838 |
|       63 | City of God (Cidade de Deus) (2002)                                       | Action|Adventure|Crime|Drama|Thriller |      4.5 |                     0.977327 |
|        6 | Lord of the Rings: The Fellowship of the Ring, The (2001)                 | Adventure|Fantasy                     |      3   |                     0.976965 |
|       14 | Brewster's Millions (1985)                                                | Comedy                                |      5   |                     0.976499 |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 36.59 |
