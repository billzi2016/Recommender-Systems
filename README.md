# Recommender systems

This project is a hands-on map of recommender systems, using MovieLens as the shared dataset.

Documentation site: https://billzi2016.github.io/Recommender-Systems/

MovieLens is useful because it gives you the pieces most recommendation algorithms need: user IDs, movie IDs, ratings, timestamps, and movie genres. With those columns, you can start from simple neighbors, move to embeddings, add feature crossing, and then try sequence and graph models.

The goal here is not to collect buzzwords. Each section answers four questions:

- What problem was this method trying to solve?
- What is the core idea?
- What does it do with MovieLens?
- What should a beginner implement first?

Start with [MovieLens](https://billzi2016.github.io/Recommender-Systems/datasets/movie_lens/), then read [getting started](https://billzi2016.github.io/Recommender-Systems/getting-started/).

## Run experiments

Install the root dependencies first:

```bash
pip install -r requirements.txt
```

### Experiment 01: traditional statistics

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings none
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: `none` uses the full MovieLens 32M dataset. For a faster trial run, use a smaller sample:

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings 2000000
./01-traditional-statistics/item-cf/run.sh --sample-ratings 5000000
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The matrix factorization command above saves only `checkpoints/best.pt`. Its report includes the `.pt` file size.

Non-main path: if you want a few intermediate checkpoints too, use:

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 20 --keep-checkpoints 3
```

Use `--no-save-checkpoints` if you do not want any `.pt` writes.

PyTorch experiments default to `--num-workers 8` for DataLoader. Lower it if your machine feels overloaded.

### Experiment 02: retrieval

```bash
./02-retrieval/two-tower-tfrs/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### Experiment 03: feature crossing

```bash
./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/deepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/xdeepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### Experiment 04: deep ranking

```bash
./04-deep-ranking/ncf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/wide-and-deep/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/dcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### Experiment 05: sequential recommendation

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./05-sequential-recommendation/sasrec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Each experiment writes:

- `report.md`: English report
- `report.zh.md`: Chinese report

The reports are also linked into the MkDocs site.
