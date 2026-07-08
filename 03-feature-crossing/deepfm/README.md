# DeepFM

DeepFM combines FM style low order interactions with a neural network.

The FM part is good at pairwise feature crossings. The deep part takes the same feature embeddings and learns nonlinear combinations. This is useful when the signal is not just "user plus genre" but several fields working together.

On MovieLens, DeepFM can use user ID, movie ID, genres, and time buckets. The target can be rating prediction or a binary label such as rating greater than or equal to 4.0.

The first version should keep the feature set small. Add features one at a time and compare against FM. If the deep part improves metrics but recommendations look worse, inspect examples before trusting the number.
