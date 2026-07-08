# MovieLens dataset

MovieLens is the shared dataset for this repository. It has enough structure to practice most recommender system ideas without first building a data pipeline from scratch.

The useful columns are simple:

- user ID: who gave the rating
- movie ID: which movie was rated
- rating: explicit feedback from 0.5 to 5.0 in MovieLens 32M
- timestamp: when the rating happened
- genres: movie side information such as comedy, action, or sci-fi

That combination is why MovieLens works well for practice. Item-CF can use user and movie IDs. Matrix factorization learns user and movie vectors from ratings. Feature crossing models can add genres. Sequential models can sort a user's ratings by timestamp. Graph models can treat users and movies as two types of nodes.

The first useful split is a time based split: for each user, keep earlier ratings for training and later ratings for validation or testing. Random splits are easier, but they leak some of the future into the past.

## Structure

- `raw/`: original MovieLens files downloaded from the official dataset.
- `processed/`: cleaned or transformed files used by experiments.
- `scripts/`: download, preprocessing, splitting, and feature engineering scripts.

## First task

Start by unzipping `raw/ml-32m.zip`, reading `ratings.csv` and `movies.csv`, then printing one user's movie history in timestamp order. If that output makes sense, the rest of the algorithms have a solid base.
