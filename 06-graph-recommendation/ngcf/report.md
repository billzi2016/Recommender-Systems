# NGCF report

## What ran

- Loaded MovieLens: 全量数据.
- Built a user-movie bipartite graph from ratings >= 4.0.
- Kept at most `50` recent positive edges per user; use `0` for all positives.
- Graph device used: `cpu`.

## Metrics

- `best_valid_loss`: `1.5886`
- `best_valid_recall@10`: `0.0500`
- `best_valid_ndcg@10`: `0.0251`
- `test_loss`: `1.6412`
- `test_recall@10`: `0.0420`
- `test_ndcg@10`: `0.0233`
- `epochs_ran`: `6.0000`

## Examples

| test_positive                          | top_recommendations                                                                                                                                                                                                                                                                                                                 |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ever After: A Cinderella Story (1998)  | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Shawshank Redemption, The (1994) | Forrest Gump (1994) | Matrix, The (1999) | Terminator 2: Judgment Day (1991) | Seven (a.k.a. Se7en) (1995) | Braveheart (1995) | Lord of the Rings: The Two Towers, The (2002) | Usual Suspects, The (1995)                             |
| Hunchback of Notre Dame, The (1996)    | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Seven (a.k.a. Se7en) (1995) | Matrix, The (1999) | Lord of the Rings: The Two Towers, The (2002) | Terminator 2: Judgment Day (1991) | Braveheart (1995) | Fight Club (1999) | Dark Knight, The (2008) | Fargo (1996)                                                      |
| Sister Act 2: Back in the Habit (1993) | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Shawshank Redemption, The (1994) | Matrix, The (1999) | Forrest Gump (1994) | Seven (a.k.a. Se7en) (1995) | Terminator 2: Judgment Day (1991) | Lord of the Rings: The Two Towers, The (2002) | Dark Knight, The (2008) | Braveheart (1995)                                |
| Stir of Echoes (1999)                  | Shawshank Redemption, The (1994) | Silence of the Lambs, The (1991) | Pulp Fiction (1994) | Forrest Gump (1994) | Seven (a.k.a. Se7en) (1995) | Matrix, The (1999) | Terminator 2: Judgment Day (1991) | Lord of the Rings: The Two Towers, The (2002) | Usual Suspects, The (1995) | Dark Knight, The (2008)                       |
| Lion King, The (1994)                  | Shawshank Redemption, The (1994) | Pulp Fiction (1994) | Silence of the Lambs, The (1991) | Matrix, The (1999) | Lord of the Rings: The Two Towers, The (2002) | Dark Knight, The (2008) | Terminator 2: Judgment Day (1991) | Usual Suspects, The (1995) | Seven (a.k.a. Se7en) (1995) | Star Wars: Episode IV - A New Hope (1977) |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 58.11 |
