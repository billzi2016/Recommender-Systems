# 推荐系统学习笔记

这个项目用 MovieLens 作为共同数据集，把推荐系统里常见的算法路线串起来。

文档站点：https://billzi2016.github.io/Recommender-Systems/

MovieLens 好用，是因为它同时有用户 ID、电影 ID、评分、时间戳和电影类型。只靠这些字段，就可以从邻居推荐做起，再走到 embedding、特征交叉、序列模型和图模型。

这里不堆名词。每一节尽量回答四个问题：

- 这个方法当初想解决什么问题？
- 它最核心的想法是什么？
- 放到 MovieLens 上具体怎么用？
- 初学者第一版代码应该先写什么？

建议先看 [MovieLens 数据集](https://billzi2016.github.io/Recommender-Systems/zh/datasets/movie_lens/)，再看 [入门路线](https://billzi2016.github.io/Recommender-Systems/zh/getting-started/)。

## 运行实验

先安装根目录依赖：

```bash
pip install -r requirements.txt
```

### 实验 01：传统统计

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings none
./01-traditional-statistics/user-cf/run.sh --sample-ratings none
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

【非主线】`none` 表示使用全量 MovieLens 32M。想先快速试跑时，再传较小采样：

```bash
./01-traditional-statistics/item-cf/run.sh --sample-ratings 2000000
./01-traditional-statistics/item-cf/run.sh --sample-ratings 5000000
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

上面的矩阵分解命令只保存 `checkpoints/best.pt`。实验报告会写入 `.pt` 文件大小。

【非主线】如果你想额外保留几个中间 checkpoint，可以这样跑：

```bash
./01-traditional-statistics/matrix-factorization/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 20 --keep-checkpoints 3
```

完全不想写 `.pt` 时，加 `--no-save-checkpoints`。

PyTorch 实验默认 `--num-workers 8`。如果机器负载太高，可以调小。

PyTorch 实验默认会复用已有的 `checkpoints/best.pt`。检测到这个文件时，脚本会跳过训练，直接进入评估和 report 生成。明确想重新训练时，再加 `--force-train`。

### 实验 02：召回

```bash
./02-retrieval/two-tower-tfrs/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### 实验 03：特征交叉

```bash
./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/deepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/xdeepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### 实验 04：深度精排

```bash
./04-deep-ranking/ncf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/wide-and-deep/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./04-deep-ranking/dcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### 实验 05：序列推荐

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./05-sequential-recommendation/sasrec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

### 实验 06：图推荐

```bash
./06-graph-recommendation/lightgcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./06-graph-recommendation/ngcf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

每个实验都会生成：

- `report.md`：英文实验报告
- `report.zh.md`：中文实验报告

这些报告也会通过 MkDocs 链接到文档站点。
