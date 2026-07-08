# Graph recommendation

Graph recommendation treats users and movies as a graph.

In MovieLens, users connect to movies through ratings. If many similar users connect to the same movies, that structure is useful even without text features. GNN methods pass information along those edges so user and movie embeddings absorb neighborhood signals.

In this project, the 06 experiments build a user-movie bipartite graph from `rating >= 4.0` interactions and train with BPR loss. The default run keeps at most 50 recent positive edges per user so local graph propagation stays manageable. Use `--max-positives-per-user 0` if you want every positive edge.

## Run

Default full-dataset runs:

```bash
./06-graph-recommendation/lightgcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./06-graph-recommendation/ngcf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./06-graph-recommendation/lightgcn/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The generated report records validation metrics, test metrics, recommendation examples, and checkpoint size.
