# Sequential recommendation

Sequential recommendation uses order.

A user who watched three science fiction movies this week may want something different from a user who watched the same three movies across five years. Timestamp order lets the model learn short term intent, not only long term taste.

In this project, the 05 experiments keep high-rating movies as positive user sequences and train next-item prediction models. The default run uses the full MovieLens data, a maximum sequence length of 50, and at most 20 recent training prefixes per user so local training stays manageable.

## Run

Default full-dataset runs:

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./05-sequential-recommendation/sasrec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: for a faster trial run:

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The generated report records validation metrics, test metrics, sequence examples, and checkpoint size.
