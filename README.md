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

The completed `01-traditional-statistics` experiments can be run with:

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings none
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 0
```

`none` uses the full MovieLens 32M dataset. For a faster trial run, use a smaller sample:

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings 2000000
./01-traditional-statistics/item-cf/run.sh --sample-ratings 5000000
```

The matrix factorization command above saves only `checkpoints/best.pt`. Its report includes the `.pt` file size.

If you want a few intermediate checkpoints too, use:

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 20 --keep-checkpoints 3
```

Use `--no-save-checkpoints` if you do not want any `.pt` writes.

Each experiment writes:

- `report.md`: English report
- `report.zh.md`: Chinese report

The reports are also linked into the MkDocs site.
