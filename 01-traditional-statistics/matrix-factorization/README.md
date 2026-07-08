# Matrix factorization

Matrix factorization turns IDs into vectors.

The rating table is mostly empty: most users rated only a tiny fraction of all movies. Matrix factorization assumes the table has hidden low dimensional structure. A user vector might encode taste for action, comedy, old movies, or niche genres. A movie vector lives in the same space. Their dot product becomes the predicted rating.

On MovieLens, the model learns two matrices:

- `P`: one embedding vector per user
- `Q`: one embedding vector per movie

The predicted rating is usually `dot(P[user], Q[movie])`, often with user bias, item bias, and a global mean.

The first implementation can use stochastic gradient descent on known ratings. Print nearest movies in the learned embedding space after training. That check makes the vectors less mysterious.
