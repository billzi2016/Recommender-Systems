# DCN

Deep and Cross Network learns explicit feature crosses with a cross network.

In many ranking problems, the important signal comes from feature combinations. DCN adds cross layers that repeatedly mix the original feature vector with the current representation. Compared with hand written crosses, it can search a larger space while keeping the structure more controlled than a plain MLP.

On MovieLens, DCN can use the same fields as DeepFM: user ID, movie ID, genres, and time buckets. It is a ranking model, so it should be evaluated after retrieval or on a sampled candidate set.

The first version should compare DCN with a plain MLP using the same embeddings. That tells you whether the cross network is doing useful work.
