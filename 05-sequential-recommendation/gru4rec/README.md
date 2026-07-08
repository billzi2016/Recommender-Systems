# GRU4Rec

GRU4Rec treats a user's history as a sequence.

Older collaborative filtering methods usually ignore order. GRU4Rec changes the question from "what does this user generally like" to "given the recent sequence, what comes next". It uses a recurrent neural network, usually GRU, to update a hidden state after each item.

On MovieLens, sort each user's ratings by timestamp. The input is a prefix of watched or liked movies. The target is the next movie. You can train with sampled negatives because predicting over every movie is expensive.

The first implementation should use short sequences and a small movie vocabulary if needed. Print examples like: history movies, true next movie, top recommended movies. Sequence models are much easier to debug when you can see the order.

## Run

Default full-dataset run:

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The report records validation metrics, test metrics, sequence examples, and checkpoint size.
