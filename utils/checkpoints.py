from __future__ import annotations

"""checkpoint 相关小工具。

只放跨实验复用的轻量逻辑：文件大小汇总。
不在这里负责训练过程保存策略。

训练保存策略由各模型训练循环决定。
这里单独抽出来，是因为所有 report 都要解释 `.pt` 文件大小，
而这部分展示逻辑不应该在每个实验里复制。
"""

from pathlib import Path


def checkpoint_size_markdown(checkpoint_dir: Path | None) -> tuple[str, str]:
    """返回中英文 checkpoint 文件大小 Markdown。

    这个信息对初学者很有价值：embedding 模型的磁盘占用和用户数、
    物品数、embedding 维度直接相关。
    """

    if checkpoint_dir is None:
        return (
            "Checkpoint saving was disabled, so no `.pt` files were written.",
            "本次未开启 checkpoint 保存，因此没有写入 `.pt` 文件。",
        )
    if not checkpoint_dir.exists():
        return (
            "Checkpoint saving was enabled, but no checkpoint files were found.",
            "本次开启了 checkpoint 保存，但没有找到 checkpoint 文件。",
        )

    files = sorted(checkpoint_dir.glob("*.pt"))
    if not files:
        return (
            "Checkpoint saving was enabled, but no checkpoint files were found.",
            "本次开启了 checkpoint 保存，但没有找到 checkpoint 文件。",
        )

    rows = ["| file | size MB |", "| --- | ---: |"]
    rows_zh = ["| 文件 | 大小 MB |", "| --- | ---: |"]
    for path in files:
        # 只展示文件名和大小，不读取 checkpoint 内容。
        # 这样报告生成不会额外占用大量内存。
        size_mb = path.stat().st_size / (1024 * 1024)
        rows.append(f"| `{path.name}` | {size_mb:.2f} |")
        rows_zh.append(f"| `{path.name}` | {size_mb:.2f} |")
    return "\n".join(rows), "\n".join(rows_zh)
