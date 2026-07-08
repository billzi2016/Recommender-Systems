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
