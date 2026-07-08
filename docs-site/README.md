# Documentation site

This directory contains the MkDocs site for the recommender systems notes.

## Local preview

```bash
pip install -r requirements.txt
mkdocs serve
```

## Build

```bash
mkdocs build --strict
```

The generated site is written to `docs-site/site/`, which is ignored by git.

