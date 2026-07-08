# Factorization Machine report

## What ran

- Loaded MovieLens: 全量数据.
- Converted ratings into binary labels: `rating >= 4.0` means positive.
- Used user ID, movie ID, movie genres, and timestamp hour bucket as sparse features.
- DataLoader workers: `8`.
- Device used: `mps`.

## Metrics

- `best_valid_logloss`: `0.6081`
- `best_valid_auc`: `0.7810`
- `epochs_ran`: `15.0000`

## Examples

Top scored held-out rows:

|   userId | title                                     | genres                   |   rating |   predicted_like_probability |
|---------:|:------------------------------------------|:-------------------------|---------:|-----------------------------:|
|       35 | Wrinkles (Arrugas) (2011)                 | Animation|Drama          |      4   |                     0.99996  |
|       63 | Running Scared (2006)                     | Action|Crime|Thriller    |      5   |                     0.999911 |
|       33 | Anvil! The Story of Anvil (2008)          | Documentary|Musical      |      4.5 |                     0.999707 |
|       35 | Capricious Summer (Rozmarné léto) (1968)  | Comedy                   |      4.5 |                     0.999692 |
|       65 | Under Fire (1983)                         | Drama|Thriller|War       |      4   |                     0.999205 |
|       35 | Chaser, The (Chugyeogja) (2008)           | Crime|Drama|Thriller     |      3   |                     0.998928 |
|       33 | Rififi (Du rififi chez les hommes) (1955) | Crime|Film-Noir|Thriller |      5   |                     0.998568 |
|       35 | The Red Turtle (2016)                     | Animation                |      3.5 |                     0.998208 |
|       50 | Man Who Would Be King, The (1975)         | Adventure|Drama          |      4.5 |                     0.998194 |
|       33 | Killing, The (1956)                       | Crime|Film-Noir          |      4.5 |                     0.998186 |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 36.33 |
