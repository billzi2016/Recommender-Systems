# User-CF

User-CF recommends items liked by similar users.

The mental model is close to asking a friend with similar taste. If two users rated many movies in a similar way, one user's liked movies can become candidates for the other.

On MovieLens, represent each user as a sparse rating vector. Compute user-user similarity with cosine similarity or Pearson correlation. For a target user, find nearest neighbors, collect movies they liked, and rank those movies by weighted neighbor scores.

The first implementation should be small because full user-user similarity can be expensive. Use a sample or build similarities only for users who share rated movies.

User-CF is useful for learning the basic neighbor idea, but it struggles when users have few ratings. That weakness helps explain why item based methods and embeddings became popular.

## Small user similarity example

Suppose there are only three movies:

| User | The Matrix | Inception | Toy Story |
| --- | --- | --- | --- |
| User A | 5 | 4 | ? |
| User B | 5 | 5 | 2 |
| User C | 1 | 2 | 5 |

The target is User A. User A has not rated Toy Story.

User B looks closer to User A because both like The Matrix and Inception. User C looks different because User C dislikes those movies but likes Toy Story. User-CF will trust User B more, so Toy Story is unlikely to be recommended.

If the table changes:

| User | The Matrix | Inception | Interstellar |
| --- | --- | --- | --- |
| User A | 5 | 4 | ? |
| User B | 5 | 5 | 5 |
| User C | 1 | 2 | 3 |

Then Interstellar becomes a good candidate because a similar user liked it.

```mermaid
flowchart LR
  U[Target user] --> N[Find similar users]
  N --> L[Collect movies neighbors liked]
  L --> F[Filter movies already seen]
  F --> R[Rank recommendations]
```

## Why User-CF can be unstable

User-CF depends on shared ratings between users. If two users only share one rated movie, a high similarity score is not very trustworthy. They may both have rated one very popular movie.

A practical first version can require a minimum number of shared rated movies before trusting a neighbor.

## Run

From the repository root:

```bash
./01-traditional-statistics/user-cf/run.sh --sample-ratings 2000000
```

Use more data or the full dataset when needed:

```bash
./01-traditional-statistics/user-cf/run.sh --sample-ratings 5000000
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
```

The script writes `report.md` and `report.zh.md`.

## You should be able to answer

- Why is User-CF similar to asking people with similar taste?
- Why are similarities unreliable when users share too few movies?
- Why can User-CF be heavier online than Item-CF?
