# SASRec

SASRec uses self attention for sequential recommendation.

GRU4Rec reads the sequence step by step. SASRec lets each position attend to earlier positions, which makes it better at picking the most relevant parts of the user's history. A recent movie may matter most, but sometimes a much older movie explains the next choice better.

On MovieLens, build sequences sorted by timestamp. The model sees previous movie IDs and predicts the next movie ID. A causal mask prevents the model from looking at future items during training.

The first version should focus on correct masking and data splitting. If the model can accidentally see future movies, the metric will look good for the wrong reason.

SASRec is a strong baseline because it keeps the recommendation problem close to next token prediction, but with movie IDs instead of words.
