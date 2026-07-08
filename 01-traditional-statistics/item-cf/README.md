# Item-CF

Item-CF recommends items similar to the items a user already liked.

The idea came from a practical weakness of User-CF. Users can be noisy and change over time, but item relationships are often more stable. If many people who liked `The Matrix` also liked another movie, that second movie becomes a reasonable candidate.

On MovieLens, build a movie-movie similarity matrix from user ratings. A simple first version can keep only positive ratings, such as ratings greater than or equal to 4.0, then compute cosine similarity between movie columns.

The first implementation should:

1. Load `ratings.csv`.
2. Build a sparse user-movie matrix.
3. Compute top similar movies for each movie.
4. For one user, collect movies similar to the user's liked movies.
5. Filter movies the user has already rated.

Item-CF is worth writing first because the output is easy to inspect. If a movie is recommended, you can trace it back to the movie that triggered it.
