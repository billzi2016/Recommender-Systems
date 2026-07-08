# 矩阵分解实验报告

这个报告还没有生成。

从仓库根目录运行：

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none
```

脚本会用真实运行结果覆盖本文件，写入真实指标和 embedding 近邻样例。

如果运行时保存 checkpoint，生成后的报告还会写入 `.pt` 文件大小。这个经验对初学者很重要：embedding 模型的体积会受用户数、电影数和 embedding 维度影响，有时比想象中更大。
