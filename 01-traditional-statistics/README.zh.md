# 传统统计

这一组从推荐系统里最早也最耐用的想法开始：行为相似，本身就是信号。

这些方法不需要神经网络。它们需要的是用户和物品的交互表、一个相似度规则，以及一个排序候选结果的方法。它们适合放在第一站，因为后面的复杂模型其实还在回答同一个问题：在反馈很稀疏的情况下，怎么猜用户接下来可能喜欢什么。

建议先写 Item-CF，再对比 User-CF，最后进入矩阵分解。

## 运行方式

先在仓库根目录安装依赖：

```bash
pip install -r requirements.txt
```

然后分别运行三个实验：

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings none
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 0
```

`none` 表示使用全量 MovieLens 32M。想先快速试跑时，再传较小采样：

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings 2000000
./01-traditional-statistics/item-cf/run.sh --sample-ratings 5000000
```

上面的矩阵分解命令只保存 `checkpoints/best.pt`，报告会记录它的文件大小。想额外保留几个中间 checkpoint 时：

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --save-checkpoints --checkpoint-every 20 --keep-checkpoints 3
```

完全不想写 `.pt` 时，加 `--no-save-checkpoints`。

每个实验会在自己的目录生成 `report.md` 和 `report.zh.md`。

这三个实验的分工：

| 实验 | 作用 | 主要依赖 |
| --- | --- | --- |
| Item-CF | 找相似电影 | `scikit-learn`, `scipy` |
| User-CF | 找相似用户 | `scikit-learn`, `scipy` |
| 矩阵分解 | 学用户和电影 embedding | `torch` |
