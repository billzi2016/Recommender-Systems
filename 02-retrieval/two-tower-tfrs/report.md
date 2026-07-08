# Two tower retrieval report

## What ran

- Loaded MovieLens: 全量数据.
- Trained a PyTorch two tower retrieval model with in-batch negatives.
- Device used: `mps`.

## Metrics

- `precision@10`: `0.0210`
- `recall@10`: `0.0254`
- `ndcg@10`: `0.0257`
- `best_valid_loss`: `7.2703`

## Examples

Example user: `1`

Recent positive history:

| title                           | genres                         |
|:--------------------------------|:-------------------------------|
| Living in Oblivion (1995)       | Comedy                         |
| Opposite of Sex, The (1998)     | Comedy|Drama|Romance           |
| Crimes and Misdemeanors (1989)  | Comedy|Crime|Drama             |
| To Catch a Thief (1955)         | Crime|Mystery|Romance|Thriller |
| Back to the Future (1985)       | Adventure|Comedy|Sci-Fi        |
| Stand by Me (1986)              | Adventure|Drama                |
| Sabrina (1954)                  | Comedy|Romance                 |
| Welcome to the Dollhouse (1995) | Comedy|Drama                   |

Retrieved candidates:

| title                            | genres                   |
|:---------------------------------|:-------------------------|
| Sleeper (1973)                   | Comedy|Sci-Fi            |
| Raising Arizona (1987)           | Comedy                   |
| M*A*S*H (a.k.a. MASH) (1970)     | Comedy|Drama|War         |
| Grateful Dead (1995)             | Documentary              |
| Snowriders (1996)                | Documentary              |
| Deliverance (1972)               | Adventure|Drama|Thriller |
| Manchurian Candidate, The (1962) | Crime|Thriller|War       |
| Being There (1979)               | Comedy|Drama             |
| Take the Money and Run (1969)    | Comedy|Crime             |
| Cape Fear (1962)                 | Crime|Drama|Thriller     |

## Checkpoint size

| file | size MB |
| --- | ---: |
| `best.pt` | 58.95 |
