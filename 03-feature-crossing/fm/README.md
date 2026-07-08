# FM

Factorization Machines learn pairwise feature interactions in sparse data.

Collaborative filtering mainly learns from user and item IDs. FM keeps that signal but can also use features such as genre, time bucket, or user metadata. Each feature gets an embedding. The model predicts by combining a linear term with pairwise dot products between feature embeddings.

On MovieLens, one training row can include:

- user ID
- movie ID
- movie genres
- rating time bucket

FM is useful when the feature vector is wide and sparse. It can learn that a particular user and a genre interact, or that a movie and a time bucket have a pattern, without manually writing every crossing rule.

The first implementation should compare FM with matrix factorization using the same train and test split.

## What one MovieLens row becomes

Original fields:

| Field | Value |
| --- | --- |
| userId | 42 |
| movieId | 2571 |
| title | The Matrix |
| genres | Action, Sci-Fi |
| rating | 5.0 |
| timestamp | 2009-01-07 21:30 |

FM sees sparse categorical features:

```text
user=42
movie=2571
genre=Action
genre=Sci-Fi
hour=night
```

```mermaid
flowchart LR
  X[Sparse features<br/>userId movieId genres time] --> E[Look up feature embeddings]
  E --> Pair[Pairwise dot products]
  Pair --> Y[Predicted rating or probability]
```

## What feature crossing means

FM can learn interactions such as:

| Feature 1 | Feature 2 | Possible meaning |
| --- | --- | --- |
| user=42 | genre=Sci-Fi | Does user 42 like sci-fi? |
| user=42 | movie=2571 | Does user 42 especially like The Matrix? |
| movie=2571 | hour=night | Is this movie rated highly at night? |
| genre=Action | genre=Sci-Fi | Is this genre combination strong? |

Writing all crosses by hand would explode. FM gives each feature an embedding and uses embedding dot products to estimate pairwise interaction strength.

## Relationship to matrix factorization

If FM only uses `userId` and `movieId`, it is close to matrix factorization. Once you add genres or time buckets, it becomes more flexible because it can learn user-genre and movie-context signals.

## First experiment

Use the same split for three runs:

1. userId and movieId only.
2. add movie genres.
3. add a simple timestamp bucket such as hour or weekday.

If genres help, the model is using side information. If time does not help, that is also plausible because MovieLens timestamp is rating time, not necessarily viewing time.

## Run

Runnable FM code will follow:

```bash
./03-feature-crossing/fm/run.sh --sample-ratings 2000000
```

## Common mistakes

Do not dump movie title as a raw category in the first version. It mostly duplicates movieId.

Do not add many features at once. Add one group at a time so you know what helped.

Split multi-value genres such as `Action|Sci-Fi` into separate genre features.
