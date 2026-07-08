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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """返回下一部电影的 logits。"""

        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        # item embedding 和 position embedding 相加，模型才能区分“同一部电影出现在不同位置”。
        x = self.item_embedding(sequences) * math.sqrt(self.item_embedding.embedding_dim)
        x = self.dropout(x + self.position_embedding(positions))
        padding_mask = sequences.eq(0)
        # True 表示被遮挡。上三角遮挡未来位置，保证 causal 训练。
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequences.device), diagonal=1)
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)
        # 左侧 padding 后，最后一个位置总是最近历史，因此用最后位置预测下一部电影。
        return self.output(encoded[:, -1, :])


def build_sequence_model(kind: str, num_items: int, max_seq_len: int, embedding_dim: int) -> nn.Module:
    """根据实验名称创建对应序列模型。"""

    if kind == "gru4rec":
        return GRU4Rec(num_items=num_items, embedding_dim=embedding_dim, hidden_dim=embedding_dim)
    if kind == "sasrec":
        return SASRec(num_items=num_items, max_seq_len=max_seq_len, embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    raise ValueError(f"Unknown sequential model: {kind}")

