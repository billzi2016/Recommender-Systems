# Two tower TFRS

A two tower model learns one network for users and one network for items.

The reason it exists is simple: retrieval must be fast. If a service has millions of items, scoring every user-item pair with a heavy model is too slow. A two tower model computes a user embedding and an item embedding separately. Retrieval becomes nearest neighbor search in embedding space.

On MovieLens, the user tower can start with only `userId`. The movie tower can start with `movieId`, then add genres later. TensorFlow Recommenders is a common way to build this model because it has retrieval tasks and candidate indexing utilities.

The first version should train on positive interactions. For example, treat high ratings as watched and liked, then train the model to place the user's next movie near the user embedding.

The main thing to inspect is candidate quality. For a few users, compare their history with the top retrieved movies before adding a ranking model.
