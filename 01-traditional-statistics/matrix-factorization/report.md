# Matrix factorization report

## What ran

- Loaded a deterministic MovieLens sample.
- Trained a biased matrix factorization model with PyTorch and early stopping.
- Device used: `mps`.

## Metrics

- `rmse`: `0.8743`
- `mae`: `0.6647`

## Examples

Example movie: `Shawshank Redemption, The (1994)`

| title                                           | genres             |   similarity |
|:------------------------------------------------|:-------------------|-------------:|
| Guardians (2016)                                | (no genres listed) |     0.848366 |
| Squeeze (1997)                                  | Drama              |     0.781397 |
| Under the Domim Tree (Etz Hadomim Tafus) (1994) | Drama              |     0.745735 |
| David Lynch: The Art Life (2016)                | (no genres listed) |     0.744699 |
| Paterno (2018)                                  | Drama              |     0.732966 |
| Braveheart (1995)                               | Action|Drama|War   |     0.732349 |
| Day The Earth Froze, The (Sampo) (1959)         | Adventure|Fantasy  |     0.726027 |
| Regret to Inform (1998)                         | Documentary        |     0.719597 |
| Random Harvest (1942)                           | Drama|Romance      |     0.692478 |
| Hitler: The Rise of Evil (2003)                 | Drama              |     0.685111 |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 65.26 |
