from __future__ import annotations

from pathlib import Path


def write_report(
    output_dir: Path,
    title: str,
    title_zh: str,
    summary_lines: list[str],
    summary_lines_zh: list[str],
    metrics: dict[str, float],
    examples_markdown: str,
    examples_markdown_zh: str,
) -> None:
    """为一个实验写入英文和中文报告。

    报告保持普通 Markdown：
    - 在算法目录里，GitHub 可以直接渲染。
    - 在 docs-site 里，MkDocs 可以通过符号链接发布。

    这里不做复杂模板系统，避免把项目变成文档生成框架。
    每个算法只要传入摘要、指标和推荐样例即可。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_md = "\n".join(f"- `{name}`: `{value:.4f}`" for name, value in metrics.items())
    metrics_zh_md = "\n".join(f"- `{name}`：`{value:.4f}`" for name, value in metrics.items())

    (output_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {title} report",
                "",
                "## What ran",
                "",
                *summary_lines,
                "",
                "## Metrics",
                "",
                metrics_md or "No metrics were produced.",
                "",
                "## Examples",
                "",
                examples_markdown,
                "",
            ]
        ),
        encoding="utf-8",
    )

    (output_dir / "report.zh.md").write_text(
        "\n".join(
            [
                f"# {title_zh} 实验报告",
                "",
                "## 本次运行做了什么",
                "",
                *summary_lines_zh,
                "",
                "## 指标",
                "",
                metrics_zh_md or "本次没有生成指标。",
                "",
                "## 推荐样例",
                "",
                examples_markdown_zh,
                "",
            ]
        ),
        encoding="utf-8",
    )
