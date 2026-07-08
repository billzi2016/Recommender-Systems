# LightGCN

LightGCN 是图推荐里一个更简洁的版本。

NGCF 把图卷积引入用户-物品图，但里面一些神经网络组件不一定总有帮助。LightGCN 去掉特征变换和非线性激活，只保留最核心的部分：沿着用户-电影边传播 embedding，并把不同层的结果做平均。

在 MovieLens 上，可以构造一张二部图。用户是一类节点，电影是一类节点，评分记录是边。很多实现会只保留正反馈，比如评分大于等于 4.0。

第一版代码建议这样写：

1. 构建用户-电影图。
2. 初始化用户和电影 embedding。
3. 做几层 embedding 传播。
4. 用 BPR loss 或 sampled softmax 训练。

LightGCN 流行的原因是简单而且强，所以很适合作为图推荐 baseline。
