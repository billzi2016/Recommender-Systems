# Wide and deep

Wide and deep combines memorization with generalization.

The wide part is a linear model over raw or crossed features. It can memorize patterns that happen often in the data. The deep part uses embeddings and dense layers, so it can generalize to combinations that were not seen exactly the same way before.

On MovieLens, the wide side can use user ID, movie ID, and simple crossed features. The deep side can use embeddings for IDs and genres. The output is one score for the user-movie pair.

The first implementation should keep the wide crosses understandable. If you cannot explain why a crossed feature is there, leave it out. The model is most useful when the wide side captures known rules and the deep side handles softer similarity.
