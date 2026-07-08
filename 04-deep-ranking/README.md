# Deep ranking

Deep ranking models score candidates after retrieval.

The candidate set is already small, so the model can spend more computation on each user-item pair. These methods usually combine memorization, embeddings, and nonlinear interaction layers to predict a rating, click, or ranking score.

In this project, the 04 experiments use MovieLens ratings as a binary ranking-style task: `rating >= 4.0` means positive. NCF starts with only user and movie IDs. Wide&Deep and DCN then add genres and timestamp hour buckets so you can see how richer ranking features change the model.

## Run

Default full-dataset runs:

```bash
./04-deep-ranking/ncf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/wide-and-deep/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/dcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./04-deep-ranking/ncf/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The generated report records validation metrics, held-out prediction examples, and checkpoint size.
