# Retrieval

Retrieval models choose a small candidate set from a large item pool.

The point is speed. A ranking model can spend more time on a few hundred movies, but it cannot score millions of movies for every request. Two tower models solve this by learning a user vector and an item vector in the same space, then using nearest neighbor search.

## Run

From the repository root:

```bash
./02-retrieval/two-tower-tfrs/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 0
```

For a faster trial run:

```bash
./02-retrieval/two-tower-tfrs/run.sh --sample-ratings 2000000 --save-checkpoints --checkpoint-every 0
```
