# Wide and deep

Wide and deep 把记忆能力和泛化能力放在一个模型里。

Wide 部分是线性模型，使用原始特征或手动交叉特征。它适合记住训练数据里经常出现的模式。Deep 部分使用 embedding 和全连接层，适合泛化到没有完全见过的组合。

在 MovieLens 上，Wide 侧可以用用户 ID、电影 ID 和少量简单交叉特征。Deep 侧可以用 ID 和 genres 的 embedding。最后输出一个用户-电影组合的分数。

第一版要让 wide 侧的交叉特征能解释。如果说不清某个交叉为什么存在，就先别加。这个模型最有价值的地方，是 wide 侧记住明确规则，deep 侧处理更软的相似性。
