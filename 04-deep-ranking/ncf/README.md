# NCF

Neural Collaborative Filtering replaces the dot product with a neural network.

Matrix factorization uses `user_embedding dot item_embedding`. That is simple and fast, but it assumes the interaction between user and item factors is mostly linear. NCF asks whether an MLP can learn a richer interaction function.

On MovieLens, NCF usually takes a user ID and a movie ID, looks up two embeddings, concatenates them, and sends the result through dense layers. The target can be a rating or a binary liked label.

The first version should compare three models:

1. matrix factorization
2. MLP only NCF
3. a combined GMF plus MLP version

NCF is a good way to see that "deep" does not automatically mean better. If the split is small or negative sampling is weak, a simple matrix factorization baseline can be hard to beat.
