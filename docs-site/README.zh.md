# 文档站点

这里是推荐系统学习笔记的 MkDocs 站点工程。

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve
```

## 构建

```bash
mkdocs build --strict
```

生成结果会写入 `docs-site/site/`，该目录不会提交到 git。

正文内容优先维护在各目录的 `README.md` 和 `README.zh.md`，MkDocs 通过链接接入，避免重复维护。
