# xDeepFM

xDeepFM 想比普通深度网络更明确地建模特征交叉。

DeepFM 主要依赖 MLP 学高阶交互。xDeepFM 加了 CIN，也就是压缩交互网络，用更结构化的方式构造有限阶的特征组合。

在 MovieLens 上，xDeepFM 适合放在 FM 和 DeepFM 之后再写。它用的字段基本一样，所以真正要观察的是：这个交互模块是不是比普通 MLP 多带来了价值。

第一版应该复用 DeepFM 的数据管线。对比要公平：同一套切分、同一套标签、同一套指标，embedding 大小也尽量接近。
