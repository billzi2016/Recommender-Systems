# User-CF

User-CF recommends items liked by similar users.

The mental model is close to asking a friend with similar taste. If two users rated many movies in a similar way, one user's liked movies can become candidates for the other.

On MovieLens, represent each user as a sparse rating vector. Compute user-user similarity with cosine similarity or Pearson correlation. For a target user, find nearest neighbors, collect movies they liked, and rank those movies by weighted neighbor scores.

The first implementation should be small because full user-user similarity can be expensive. Use a sample or build similarities only for users who share rated movies.

User-CF is useful for learning the basic neighbor idea, but it struggles when users have few ratings. That weakness helps explain why item based methods and embeddings became popular.
