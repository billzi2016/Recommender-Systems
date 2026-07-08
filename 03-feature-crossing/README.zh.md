# 特征交叉

特征交叉模型问的是一个很实际的问题：只有 ID 够不够？

电影类型、用户属性、时间段、上下文都可能有信号。FM 把稀疏特征变成 embedding，并学习任意两个特征之间的关系。DeepFM 和 xDeepFM 保留这个想法，再用神经网络学习更复杂的组合。

在 MovieLens 上，最自然的侧信息是 genres。比如同样是用户 42，他可能不是喜欢某一部电影本身，而是长期偏爱 `Sci-Fi` 和 `Action` 的组合。特征交叉模型就是为了把这些组合关系学出来。

建议先写 FM，因为它最容易看清“二阶交叉”是什么。再写 DeepFM，把 FM 和 MLP 放在一起。最后再看 xDeepFM，理解更显式的高阶交叉。

这个项目里的 03 实验会把 MovieLens 评分转成更接近排序任务的二分类问题：`rating >= 4.0` 表示用户大概率喜欢这部电影。每条样本会用到：

- 用户 ID
- 电影 ID
- 电影 genres
- timestamp 转出来的小时段

三个实验复用同一套数据管线，这样对比才公平。

## 运行

默认全量运行：

```bash
./03-feature-crossing/fm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/deepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
./03-feature-crossing/xdeepfm/run.sh --sample-ratings none --num-workers 8 --save-checkpoints --checkpoint-every 0
```

想先快速试跑：

```bash
./03-feature-crossing/fm/run.sh --sample-ratings 2000000 --num-workers 8 --save-checkpoints --checkpoint-every 0
```

默认命令只保存 `checkpoints/best.pt`。生成的实验报告会写入 `.pt` 文件大小。
