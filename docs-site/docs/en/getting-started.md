# Getting started

Use the algorithms in this order:

1. Item-CF
2. Matrix factorization
3. Two tower retrieval
4. DeepFM
5. SASRec
6. LightGCN

The order matters because each step adds one new idea.

Item-CF teaches similarity. Matrix factorization turns IDs into vectors. A two tower model turns that vector idea into retrieval. DeepFM adds side features such as genres. SASRec uses time order. LightGCN treats the same data as a graph.

For the first pass, do not tune too much. Pick one small split, implement the data loading carefully, and print a few recommendations for real users. A bad recommendation you can inspect is more useful than a good metric you cannot explain.

