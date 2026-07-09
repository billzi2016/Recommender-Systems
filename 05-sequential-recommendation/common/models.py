from __future__ import annotations

"""GRU4Rec 和 SASRec 的 PyTorch 模型。

两个模型都输入一段电影 index 序列，输出“下一部电影”的 logits。
logits 的维度是 `num_items + 1`，其中 0 是 padding 位，不作为真实目标使用。
"""

import math

import torch
from torch import nn


class GRU4Rec(nn.Module):
    """GRU4Rec：用 GRU 顺序读取用户历史。

    GRU 会从左到右更新隐藏状态。最后一个隐藏状态可以理解成“当前兴趣压缩表示”，
    再用线性层预测下一部电影。
    """

    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """返回下一部电影的 logits。

        这里直接取 GRU 最后一层的 hidden state。
        因为输入已经左侧 padding，真正的最近行为位于序列右侧。
        """

        embedded = self.dropout(self.item_embedding(sequences))
        _, hidden = self.gru(embedded)
        final_state = hidden[-1]
        return self.output(self.dropout(final_state))


class SASRec(nn.Module):
    """SASRec：用 causal self-attention 做序列推荐。

    Transformer Encoder 默认可以看见整个序列。
    SASRec 必须使用 causal mask，让每个位置只能看见自己和更早的位置，
    否则模型会偷看未来电影，指标会虚高。
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # PyTorch 的 TransformerEncoder 默认会尝试 nested tensor 优化。
        # 这个优化在 CUDA/CPU 上通常没问题，但当前 MPS 还缺少
        # `aten::_nested_tensor_from_mask_left_aligned` 这个算子。
        # 如果不关掉，SASRec 可能训练完一个 epoch 后在验证阶段崩掉。
        # 关闭 nested tensor 后仍然是同一个 SASRec 结构，只是走普通 tensor 路径。
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        # causal mask 只和最大序列长度有关，不需要每个 batch 重新创建。
        # 把它注册成 buffer 后，`model.to(device)` 会自动把 mask 一起搬到 MPS/CUDA/CPU。
        # `persistent=False` 表示 checkpoint 里不保存这个可重建张量，减少一点无意义体积。
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """返回下一部电影的 logits。

        这里仍然使用全量 softmax：每条样本都会对所有电影打分。
        这样最容易理解，也方便直接算 Recall/NDCG；代价是 MovieLens 32M 全量训练时，
        输出层会成为主要计算开销，SASRec 比 GRU4Rec 更容易把 MPS/GPU 打满。
        """

        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        # item embedding 和 position embedding 相加，模型才能区分“同一部电影出现在不同位置”。
        x = self.item_embedding(sequences) * math.sqrt(self.item_embedding.embedding_dim)
        x = self.dropout(x + self.position_embedding(positions))
        padding_mask = sequences.eq(0)
        # MovieLens 序列是左侧 padding。若把 padding_mask 直接交给 Transformer，
        # 某些 padding query 在 causal mask 叠加后会出现“整行都被 mask”的情况。
        # CPU/CUDA 往往能绕过去，但 MPS 的 Transformer 更容易在这种行上产生 NaN。
        # 这里改成更朴素的做法：padding 位置的输入向量清零，不再把 key padding mask
        # 传给 Transformer。真实最近行为仍在最后一个位置，模型输出不依赖 padding query。
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # True 表示被遮挡。上三角遮挡未来位置，保证 causal 训练。
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        encoded = self.encoder(x, mask=causal_mask)
        encoded = self.layer_norm(encoded)
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # 左侧 padding 后，最后一个位置总是最近历史，因此用最后位置预测下一部电影。
        return self.output(encoded[:, -1, :])


def build_sequence_model(kind: str, num_items: int, max_seq_len: int, embedding_dim: int) -> nn.Module:
    """根据实验名称创建对应序列模型。"""

    if kind == "gru4rec":
        return GRU4Rec(num_items=num_items, embedding_dim=embedding_dim, hidden_dim=embedding_dim)
    if kind == "sasrec":
        return SASRec(num_items=num_items, max_seq_len=max_seq_len, embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    raise ValueError(f"Unknown sequential model: {kind}")
