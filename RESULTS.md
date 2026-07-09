# Experiment Results Summary

This file puts the generated experiment reports side by side. It is not a single absolute leaderboard because the experiments optimize different tasks:

- Rating prediction uses `RMSE / MAE`; lower is better.
- Binary ranking uses `AUC / logloss / accuracy`; higher AUC and lower logloss are better.
- Retrieval, sequential, and graph recommendation use `Recall@10 / NDCG@10`; higher is better.
- `best.pt` size shows the saved parameter footprint, not model quality.

SASRec is currently the only experiment run on a `2,000,000`-rating sample. Most other generated reports use the full MovieLens 32M data. For now, the SASRec effectiveness metrics should not be compared directly with the full-data runs. Its checkpoint size is still listed as the actual saved file size; it is driven mainly by the model architecture and item vocabulary, not directly by the number of training rows.

## Overview

| Group | Method | Data scale | Main metrics | checkpoint |
| --- | --- | --- | --- | ---: |
| 01 Traditional statistics | Item-CF | Full | precision@10 `0.0000`, recall@10 `0.0000` | None |
| 01 Traditional statistics | User-CF | Full | precision@10 `0.1000`, recall@10 `0.0019` | None |
| 01 Traditional statistics | Matrix Factorization | Full | RMSE `0.8743`, MAE `0.6647` | `65.26 MB` |
| 02 Retrieval | Two Tower | Full | precision@10 `0.0210`, recall@10 `0.0254`, NDCG@10 `0.0257` | `58.95 MB` |
| 03 Feature crossing | FM | Full | logloss `0.6081`, AUC `0.7810` | `36.33 MB` |
| 03 Feature crossing | DeepFM | Full | logloss `0.6096`, AUC `0.7784` | `71.79 MB` |
| 03 Feature crossing | xDeepFM | Full | logloss `0.5483`, AUC `0.7956` | `36.59 MB` |
| 04 Deep ranking | NCF | Full | logloss `0.5308`, AUC `0.8091`, accuracy `0.7302` | `66.59 MB` |
| 04 Deep ranking | Wide & Deep | Full | logloss `0.5374`, AUC `0.8049`, accuracy `0.7268` | `67.70 MB` |
| 04 Deep ranking | DCN | Full | logloss `0.5325`, AUC `0.8083`, accuracy `0.7293` | `66.66 MB` |
| 05 Sequential recommendation | GRU4Rec | Full | test recall@10 `0.1441`, test NDCG@10 `0.0769` | `27.25 MB` |
| 05 Sequential recommendation | SASRec | 2,000,000 ratings | test recall@10 `0.0260`, test NDCG@10 `0.0121` | `10.98 MB` |
| 06 Graph recommendation | LightGCN | Full | test recall@10 `0.0560`, test NDCG@10 `0.0287` | `58.05 MB` |
| 06 Graph recommendation | NGCF | Full | test recall@10 `0.0420`, test NDCG@10 `0.0233` | `58.11 MB` |

## How to Read the Results

The traditional collaborative filtering numbers are weak, and that is expected. Item-CF and User-CF do not learn parameters or use rich negative sampling. Their value is that they make neighbor-based recommendation easy to understand before moving into embedding models.

Matrix factorization solves rating prediction, so its `RMSE / MAE` should not be compared directly with `Recall@10`. Its role is conceptual: it shows how user IDs and movie IDs become vectors. Two-tower retrieval, NCF, and graph recommenders all inherit that vector idea in different forms.

Feature crossing and deep ranking are easier to compare with each other. In the current reports, the 04 deep ranking models sit around `0.80` AUC and outperform the 03 feature-crossing baselines. AUC measures positive/negative ordering quality, not the quality of a final recommendation list after candidate generation.

For sequential recommendation, GRU4Rec is much stronger than SASRec in the current table, but this is not a fair comparison. GRU4Rec used the full data, while SASRec used a 2M-rating sample. The SASRec run mainly proves that the MPS, masking, full-softmax, checkpoint, and report pipeline now work.

For graph recommendation, LightGCN currently beats NGCF. That matches a common lesson in recommender systems: a simpler graph propagation model can outperform a heavier graph neural network when the core signal is collaborative filtering.

## Which Model Is Best Right Now

Based only on the generated reports, there is no single winner across every task because the experiments measure different things. A better reading is to split the question by task:

| Question | Current best | Evidence |
| --- | --- | --- |
| Rating prediction | Matrix Factorization | It is the current explicit-rating experiment, with RMSE `0.8743` and MAE `0.6647`. |
| Binary ranking | NCF | AUC `0.8091`, logloss `0.5308`, accuracy `0.7302`; slightly ahead within the 04 group. |
| Top-K recommendation list | GRU4Rec | test recall@10 `0.1441`, test NDCG@10 `0.0769`; clearly strongest among the current Top-K reports. |
| Graph recommendation baseline | LightGCN | test recall@10 `0.0560`, test NDCG@10 `0.0287`; ahead of NGCF. |

So if the question is “which current run has the strongest recommendation-list metrics?”, the answer is **GRU4Rec**. If the question is “which ranking model is strongest?”, the answer is **NCF**. If the question is “which model is best for understanding the vector idea behind recommender systems?”, the answer is still **Matrix Factorization**.

One important caveat: SASRec currently uses a `2,000,000`-rating sample, while GRU4Rec uses the full MovieLens 32M data. This does not prove that the SASRec algorithm is worse than GRU4Rec. It only says that, under the current implementation and run scale, GRU4Rec has the strongest full-data Top-K result and the SASRec small-scale run is now working.

## Useful Takeaways

| Observation | Interpretation |
| --- | --- |
| NCF, DCN, and Wide & Deep have close AUC values | They solve a similar binary ranking task with similar inputs, so the gap is not huge. |
| xDeepFM beats FM and DeepFM here | Explicit high-order feature crossing helps the current MovieLens feature setup. |
| GRU4Rec has strong Recall@10 | Full-data sequence training captures short-term intent and popular transition patterns. |
| LightGCN beats NGCF | Simpler graph propagation can be more stable than adding heavier transformations. |
| Checkpoint size does not equal model quality | DeepFM is much larger than xDeepFM here, but its AUC is lower. |

## For a Fairer Comparison

A stricter comparison should unify:

- The same train / validation / test split.
- The same positive and negative sampling protocol.
- The same data scale, either all full data or all sampled data.
- The same candidate set and top-k evaluation protocol.
- Multiple random seeds with mean and variance.

This summary is best read as a learning map: it shows what each family is doing, what metric it reports, and how the current implementation behaves.
