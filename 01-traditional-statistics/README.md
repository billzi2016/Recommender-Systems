# Traditional statistics

This group starts with the oldest useful idea in recommendation: similar behavior is a signal.

These methods do not need neural networks. They need a user-item table, a similarity rule, and a way to rank candidates. That makes them a good first stop because every later model is still trying to solve the same basic problem: from sparse feedback, guess what a user may like next.

Start with Item-CF, then compare it with User-CF, then move to matrix factorization.

## Run

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the three experiments:

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings none
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 0
```

`none` uses the full MovieLens 32M dataset. For a faster trial run, pass a smaller sample:

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings 2000000
./01-traditional-statistics/item-cf/run.sh --sample-ratings 5000000
```

The matrix factorization command above saves only `checkpoints/best.pt`, and the report records its file size. To keep a few intermediate checkpoints too:

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 20 --keep-checkpoints 3
```

Use `--no-save-checkpoints` to disable `.pt` writes.

Each experiment writes `report.md` and `report.zh.md` in its own directory.
