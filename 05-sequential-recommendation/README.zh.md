# 序列推荐

序列推荐用的是顺序信息。

一个用户这周连续看了三部科幻片，和一个用户五年里分别看了同样三部电影，含义可能不一样。时间戳能让模型学习短期兴趣，而不只是长期口味。

MovieLens 的 timestamp 让这个问题可以成立。先把每个用户的高评分电影按时间排序，就得到一条电影序列。序列模型要回答的是：看完前面这些电影后，下一部更可能是什么？

建议先看 GRU4Rec，再看 SASRec。GRU4Rec 按顺序一步步读历史，SASRec 用自注意力在历史里挑重点。

这个项目里的 05 实验只保留高评分电影作为用户正反馈序列，训练 next-item prediction 模型。默认使用全量 MovieLens，最大序列长度 50，每个用户最多保留最近 20 条训练前缀，这样本地训练不会因为长历史用户生成过多样本。

## 运行

默认全量运行：

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./05-sequential-recommendation/sasrec/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

【非主线】想先快速试跑：

```bash
./05-sequential-recommendation/gru4rec/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

默认命令只保存 `checkpoints/best.pt`。生成的实验报告会写入验证指标、测试指标、序列样例和 checkpoint 大小。
