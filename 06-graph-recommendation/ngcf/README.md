# NGCF

NGCF applies graph neural networks to collaborative filtering.

The basic graph idea is natural for MovieLens: users connect to movies through ratings. A user's representation should be influenced by movies they interacted with, and a movie's representation should be influenced by users who interacted with it.

NGCF propagates information across the user-item graph and uses neural transformations during message passing. Compared with plain matrix factorization, it explicitly uses higher order neighbors: users connected through movies, movies connected through users, and so on.

The first implementation should keep the graph small or sampled. Full graph training can become heavy. Compare it with LightGCN afterward. That comparison is useful because LightGCN asks whether the extra neural components in NGCF are really needed.

## Run

Default full-dataset run:

```bash
./06-graph-recommendation/ngcf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./06-graph-recommendation/ngcf/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The report records validation metrics, test metrics, recommendation examples, and checkpoint size.
