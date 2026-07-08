# Feature crossing

Feature crossing models ask a practical question: what if IDs are not enough?

Movie genres, user attributes, time buckets, and context fields can all carry signal. FM style models turn sparse features into embeddings and learn how pairs of features interact. DeepFM and xDeepFM keep that idea, then add neural networks for more complex interactions.

In this project, the 03 experiments turn MovieLens ratings into a binary ranking-style task: `rating >= 4.0` means the user probably liked the movie. Each row uses:

- user ID
- movie ID
- movie genres
- timestamp hour bucket

The three experiments share the same data pipeline so the comparison stays fair.

## Run

Default full-dataset runs:

```bash
./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/deepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/xdeepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

For a faster trial run:

```bash
./03-feature-crossing/fm/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

The default command saves only `checkpoints/best.pt`. The generated report records the `.pt` file size.
