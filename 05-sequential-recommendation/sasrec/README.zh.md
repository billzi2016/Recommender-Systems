# SASRec

SASRec 用自注意力做序列推荐。

GRU4Rec 是一步一步读序列。SASRec 让当前位置可以关注前面不同位置的电影，所以更容易抓住用户历史里真正相关的部分。最近看的电影经常重要，但有时更早的一部电影反而更能解释下一次选择。

在 MovieLens 上，先按 timestamp 构造用户序列。模型看到前面的电影 ID，预测下一部电影 ID。训练时必须用 causal mask，避免模型偷看到未来电影。

第一版最该关注的是 mask 和数据切分。如果模型不小心看到了未来，指标会很好看，但那是假的。

SASRec 是很强的 baseline，因为它把推荐问题变得很像 next token prediction，只是 token 从词变成了电影 ID。
