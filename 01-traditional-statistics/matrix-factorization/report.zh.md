# 矩阵分解 实验报告

## 本次运行做了什么

- 读取 MovieLens：全量数据。
- 使用 PyTorch 训练带用户偏置和电影偏置的矩阵分解模型，最多 1000 轮，early stopping patience 为 5。
- 本次使用设备：`mps`。
- 默认保存 `best.pt`，并按间隔最多保留少量中间 checkpoint；如需关闭可传 `--no-save-checkpoints`。

## 指标

- `rmse`：`0.8743`
- `mae`：`0.6647`

## 推荐样例

样例电影：`Shawshank Redemption, The (1994)`

| title                                           | genres             |   similarity |
|:------------------------------------------------|:-------------------|-------------:|
| Guardians (2016)                                | (no genres listed) |     0.848366 |
| Squeeze (1997)                                  | Drama              |     0.781397 |
| Under the Domim Tree (Etz Hadomim Tafus) (1994) | Drama              |     0.745735 |
| David Lynch: The Art Life (2016)                | (no genres listed) |     0.744699 |
| Paterno (2018)                                  | Drama              |     0.732966 |
| Braveheart (1995)                               | Action|Drama|War   |     0.732349 |
| Day The Earth Froze, The (Sampo) (1959)         | Adventure|Fantasy  |     0.726027 |
| Regret to Inform (1998)                         | Documentary        |     0.719597 |
| Random Harvest (1942)                           | Drama|Romance      |     0.692478 |
| Hitler: The Rise of Evil (2003)                 | Drama              |     0.685111 |

## Checkpoint 大小

| 文件 | 大小 MB |
| --- | ---: |
| `best.pt` | 65.26 |
