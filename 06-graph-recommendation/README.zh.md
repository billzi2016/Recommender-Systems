# 图推荐

图推荐把用户和电影看成一张图。

在 MovieLens 里，用户通过评分连接到电影。如果很多相似用户都连到同一批电影，这个结构本身就有信息，即使没有文本特征也能用。GNN 方法会沿着边传递信息，让用户和电影 embedding 吸收邻居信号。

图推荐的关键是“关系”。矩阵分解也用用户和电影的交互，但图模型会更明确地利用多跳结构：我喜欢的电影、也喜欢这些电影的人、那些人还喜欢的电影。

建议先看 LightGCN，因为它更简洁。再看 NGCF，理解早期图神经网络推荐里加入了哪些神经网络组件。

这个项目里的 06 实验会用 `rating >= 4.0` 构建用户-电影二部图，并用 BPR loss 训练。默认每个用户最多保留最近 50 条正反馈边，让本地图传播更可控。如果想使用全部正反馈边，可以传 `--max-positives-per-user 0`。

## 运行

默认全量运行：

```bash
./06-graph-recommendation/lightgcn/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./06-graph-recommendation/ngcf/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

【非主线】想先快速试跑：

```bash
./06-graph-recommendation/lightgcn/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

默认命令只保存 `checkpoints/best.pt`。生成的实验报告会写入验证指标、测试指标、推荐样例和 checkpoint 大小。
