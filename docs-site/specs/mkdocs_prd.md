# MkDocs bilingual documentation PRD

## 1. Purpose

This document defines the MkDocs documentation site requirements for this repository.

The goal is to keep future work aligned with the current structure: each repository directory owns its documentation through `README.md` and `README.zh.md`, while `docs-site/` owns site configuration, navigation, build dependencies, and publishing.

This PRD targets the `Recommender-Systems` repository while keeping the design reusable. If the structure is moved to another repository, preserve the same idea: directory README files are the content source, and MkDocs is the navigation and publishing layer.

## 2. Goals

The documentation site must:

- Use MkDocs as the documentation framework.
- Support English and Chinese with an i18n plugin.
- Keep the documentation site project inside `docs-site/`.
- Use `README.md` as the English source for each directory.
- Use `README.zh.md` as the Chinese source for each directory.
- Connect `docs-site/docs/en` and `docs-site/docs/zh` to those README files through symlinks.
- Avoid duplicated algorithm and dataset content inside the MkDocs docs directory.
- Keep English and Chinese navigation as parallel as possible.
- Provide a GitHub Actions workflow for building and deploying the site.

## 3. Scope

This PRD covers:

- A bilingual MkDocs site for this repository.
- Repository README files as documentation content.
- MkDocs configuration, navigation, dependencies, and static assets.
- GitHub Pages deployment through GitHub Actions.

It does not cover application deployment, model training, data processing pipelines, or advanced multi-version documentation.

## 4. General requirements

### 4.1 Engineering independence

- `docs-site/` must contain the documentation site project.
- Site configuration, navigation, dependencies, assets, and build logic should live in `docs-site/`.
- Algorithm and dataset content should live beside the related code or data directory.
- Do not duplicate the same content in `docs-site/docs/`.
- Use symlinks so MkDocs can read README content from the repository directories.

### 4.2 Bilingual consistency

- The site must support English and Chinese.
- English files use `README.md`.
- Chinese files use `README.zh.md`.
- Do not use `README.en.md`.
- Matching English and Chinese pages should exist for the same topics when practical.
- Navigation groups, depth, and naming should remain parallel.

### 4.3 Technical consistency

- Use MkDocs.
- Use an i18n plugin for multilingual support.
- Prefer mature plugins and simple configuration.
- Avoid complex custom scripts when configuration is enough.

### 4.4 Automation

- Provide a standard GitHub Actions workflow.
- The workflow must build the MkDocs site.
- The workflow must deploy the generated static site to GitHub Pages.

## 5. Information architecture

The site should use a simple structure:

- Home: overview and reading entry points.
- Getting started: the first path through the repository.
- Datasets: MovieLens and future datasets.
- Algorithms: algorithm groups and method pages.
- Project: documentation setup, PRDs, and repository notes.

Algorithm and dataset explanations should live in their own directory README files so they remain useful even outside the generated site.

## 6. Content organization

### 6.1 Language layout

- English content source: `README.md`.
- Chinese content source: `README.zh.md`.
- `docs-site/docs/en` should expose English pages.
- `docs-site/docs/zh` should expose Chinese pages.
- Algorithm and dataset pages in MkDocs should be symlinks to their README source files.

### 6.2 Page naming

- Names should be clear and stable.
- The same topic should use corresponding English and Chinese files.
- Repository directories must use `README.md` and `README.zh.md`.
- Avoid temporary or personal naming conventions.

### 6.3 Navigation

- Navigation should reflect how readers learn the topic, not only file locations.
- Keep groups shallow and predictable.
- English and Chinese navigation should mirror each other where possible.
- Navigation entries may point to symlinked files under `docs-site/docs/`.

### 6.4 Initial content

- Empty placeholder pages are not enough for algorithm and dataset directories.
- Each algorithm page should explain why the method exists, the core idea, how it maps to MovieLens, and what the first implementation should do.
- Future edits should update the directory README source rather than copying content into MkDocs.

## 7. Configuration requirements

### 7.1 MkDocs configuration

The configuration must include:

- Site metadata.
- Theme configuration.
- Navigation.
- i18n configuration.
- Markdown extensions.
- Plugins.
- Static asset handling.
- The symlink based content convention.

### 7.2 i18n configuration

The i18n setup must:

- Declare English and Chinese.
- Set a default language.
- Support language switching.
- Keep the current mapping: English uses `README.md`, Chinese uses `README.zh.md`.
- Allow future languages without redesigning the whole structure.

### 7.3 Readability

- Keep configuration sections readable.
- Use names that a maintainer can understand quickly.
- Do not hide important behavior inside difficult scripts.

## 8. Automation and deployment

### 8.1 GitHub Actions

The workflow must:

- Check out the repository.
- Install documentation dependencies.
- Build the MkDocs site.
- Upload the generated static artifact.
- Deploy it to GitHub Pages.

### 8.2 Workflow quality

- Keep the workflow simple and stable.
- Use clear job and step names.
- Prefer official GitHub Actions.
- Avoid unnecessary complexity.

### 8.3 Deployment target

- Use GitHub Pages first.
- Keep the structure portable enough for another static host later.

## 9. Non-functional requirements

### 9.1 Reusability

- The structure should work as a template for similar repositories.
- The README based content source should remain clear.

### 9.2 Maintainability

- New contributors should know where to edit content.
- Algorithm and dataset content belongs in the related directory README files.
- Site configuration belongs in `docs-site/`.

### 9.3 Consistency

- English and Chinese structure should stay aligned.
- Page style should stay plain, concrete, and friendly to beginners.
- Avoid hollow claims, inflated language, and duplicated content.

### 9.4 Extensibility

- The site may later add new sections, more languages, search improvements, SEO, or versioned docs.
- Do not add advanced systems until they are needed.

## 10. Deliverables

The implementation must include:

- `docs-site/` documentation site project.
- `docs-site/mkdocs.yml`.
- i18n configuration.
- English and Chinese entry pages or entry links.
- `README.md` and `README.zh.md` for algorithm and dataset directories.
- Symlinks under `docs-site/docs/en` and `docs-site/docs/zh` pointing to those README files.
- Static assets directory.
- GitHub Actions deployment workflow.
- A short maintenance note.

## 11. Acceptance criteria

The task is complete when:

1. `docs-site/` exists as an independent MkDocs project.
2. The site is built with MkDocs.
3. The site supports English and Chinese.
4. Multilingual support uses an i18n plugin.
5. English and Chinese entry pages can be opened independently.
6. English and Chinese navigation are mostly parallel.
7. The site builds through GitHub Actions.
8. The site deploys to GitHub Pages or a compatible static hosting target.
9. Algorithm and dataset content has a single source of truth.
10. Content is not duplicated inside `docs-site/docs/`.
11. The structure can be extended without redesigning the site.

## 12. Implementation constraints

- Do not make the solution depend on a single contributor's habits.
- Do not over-couple the configuration to one repository detail.
- Do not skip bilingual structure.
- Do not use `README.en.md`.
- Do not copy algorithm and dataset content into `docs-site/docs/`.
- Do not omit the deployment workflow.
- Do not introduce unnecessary complexity.

## 13. Implementation instruction

Implement a documentation site that:

- Uses MkDocs.
- Uses i18n for English and Chinese.
- Keeps the site project in `docs-site/`.
- Keeps algorithm and dataset content in directory README files.
- Uses symlinks from MkDocs docs paths to README files.
- Provides clear navigation and readable initial pages.
- Provides GitHub Actions build and deployment.
- Targets GitHub Pages.

## 14. Expected result

The repository should have an independent bilingual documentation site that is easy to build, easy to deploy, and easy to extend. Readers can browse it through MkDocs, while contributors edit the README files where the relevant code, dataset, or algorithm folder already lives.

