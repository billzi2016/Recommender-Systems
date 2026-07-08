# LightGCN

LightGCN simplifies graph neural recommendation.

NGCF introduced graph convolution for user-item graphs, but some neural network parts were not always helpful. LightGCN removes feature transformation and nonlinear activation, then keeps the part that matters most: passing embeddings along user-item edges and averaging layer outputs.

On MovieLens, create a bipartite graph. Users are one node type, movies are another node type, and ratings become edges. Many implementations keep only positive interactions, such as ratings greater than or equal to 4.0.

The first implementation should:

1. Build the user-movie graph.
2. Initialize user and movie embeddings.
3. Propagate embeddings for a few layers.
4. Train with Bayesian Personalized Ranking loss or sampled softmax.

LightGCN is popular because it is simple and strong. That makes it a good graph baseline.
