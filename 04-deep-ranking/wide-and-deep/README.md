# Wide and deep

Wide and deep combines memorization with generalization.

The wide part is a linear model over raw or crossed features. It can memorize patterns that happen often in the data. The deep part uses embeddings and dense layers, so it can generalize to combinations that were not seen exactly the same way before.

On MovieLens, the wide side can use user ID, movie ID, and simple crossed features. The deep side can use embeddings for IDs and genres. The output is one score for the user-movie pair.

The first implementation should keep the wide crosses understandable. If you cannot explain why a crossed feature is there, leave it out. The model is most useful when the wide side captures known rules and the deep side handles softer similarity.

## Run

Default full-dataset run:

```bash
./04-deep-ranking/wide-and-deep/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./04-deep-ranking/wide-and-deep/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The report records validation metrics, held-out prediction examples, and checkpoint size.
