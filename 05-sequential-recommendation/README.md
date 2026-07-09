# Sequential recommendation

Sequential recommendation uses order.

A user who watched three science fiction movies this week may want something different from a user who watched the same three movies across five years. Timestamp order lets the model learn short term intent, not only long term taste.

In this project, the 05 experiments keep high-rating movies as positive user sequences and train next-item prediction models. The GRU4Rec main command still uses the full MovieLens data. SASRec is heavier because it uses full softmax, so its main command uses a 2,000,000-rating sample. Both use a maximum sequence length of 50 and at most 20 recent training prefixes per user by default.

## Run

Main runs:

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./05-sequential-recommendation/sasrec/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

Non-main path: use the full SASRec MovieLens 32M run only when you intentionally want it:

```bash
./05-sequential-recommendation/sasrec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The generated report records validation metrics, test metrics, sequence examples, and checkpoint size.
