# Traditional statistics

This group starts with the oldest useful idea in recommendation: similar behavior is a signal.

These methods do not need neural networks. They need a user-item table, a similarity rule, and a way to rank candidates. That makes them a good first stop because every later model is still trying to solve the same basic problem: from sparse feedback, guess what a user may like next.

Start with Item-CF, then compare it with User-CF, then move to matrix factorization.
