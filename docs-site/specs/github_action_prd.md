# GitHub Actions deployment PRD

## 1. Purpose

This document defines the GitHub Actions build and deployment requirements for this repository.

The goal is to keep future changes consistent. The documentation content in this repository lives in directory level README files, while MkDocs provides navigation, build configuration, and publishing. The workflow must therefore react to changes in README files, the documentation site, and the deployment workflow itself.

This PRD targets the `Recommender-Systems` repository while keeping the structure reusable. If the pattern is moved to another repository, keep the same boundary: documentation content is maintained in README files, the documentation site builds independently, and GitHub Pages publishes the static output.

## 2. Goals

The repository needs a clear, standalone GitHub Actions workflow that:

- Builds and deploys the documentation site automatically on GitHub.
- Publishes a static documentation site.
- Uses GitHub Pages as the primary deployment target.
- Keeps the workflow readable and maintainable.
- Triggers on `docs-site/`, `.github/workflows/`, `README.md`, and `README.zh.md` changes.
- Can be reused as a reference for similar repositories.

## 3. Scope

This PRD covers:

- Static documentation site builds.
- GitHub based CI and deployment.
- A documentation workflow separated from application or experiment code.
- A reusable implementation pattern for future AI or human contributors.

This PRD does not cover application deployment, container orchestration, multi-environment infrastructure, or custom hosting platforms.

## 4. General requirements

### 4.1 Independence

- The deployment workflow must be defined independently.
- It should focus on documentation build and deployment only.
- It should not include unrelated application, experiment, or data processing tasks.

### 4.2 Reusability

- The workflow should not hardcode project specific algorithm logic.
- It may rely on the documentation site living in `docs-site/`.
- It should be easy to adapt to another repository with the same README based documentation pattern.

### 4.3 Stability

- Prefer official GitHub Actions where possible.
- Avoid fragile custom scripts.
- Use normal GitHub repository permissions and GitHub Pages behavior.

## 5. Workflow responsibilities

The workflow must:

- Trigger automatically after relevant changes on the main branch.
- Support manual execution through `workflow_dispatch`.
- Check out repository code.
- Prepare a Python environment.
- Install documentation dependencies from `docs-site/requirements.txt`.
- Build the MkDocs site.
- Upload the generated static site artifact.
- Deploy the artifact to GitHub Pages.

## 6. Trigger strategy

### 6.1 Automatic trigger

- The workflow should run on updates to the main branch.
- It must not run on every push by default.
- It must use `paths` or an equivalent mechanism to limit runs to documentation related changes.
- The path list must include `docs-site/**`, the deployment workflow file, `**/README.md`, and `**/README.zh.md`.
- README paths are required because algorithm and dataset pages use README files as their single source of content.

### 6.2 Manual trigger

- The workflow must support `workflow_dispatch`.
- Manual runs are used for debugging, redeploying, and validation.

### 6.3 Concurrency

- The workflow should use a clear concurrency group for GitHub Pages deployment.
- Newer runs may cancel older in-progress runs to avoid conflicting deployments.

## 7. Build requirements

### 7.1 Runtime

- Use a stable Ubuntu runner.
- Use a stable Python version.
- Keep setup steps simple and readable.

### 7.2 Dependency installation

- Install from `docs-site/requirements.txt`.
- Do not rely on hidden global packages.
- Dependency installation should be obvious to a maintainer reading the workflow.

### 7.3 Build command

- Run the build from `docs-site/`.
- Use `mkdocs build --strict`.
- Build failures should be visible and should not be ignored.
- The build must allow MkDocs to read symlinked pages under `docs-site/docs/`.

## 8. Deployment requirements

### 8.1 Target

- Use GitHub Pages as the primary deployment target.
- Keep the workflow simple enough to migrate to another static hosting target later.

### 8.2 Artifact handling

- Build output must stay separate from source files.
- Upload the generated `docs-site/site` directory as the Pages artifact.
- Do not commit generated site output.

### 8.3 Permissions

- Use the minimum permissions required for GitHub Pages:
  - `contents: read`
  - `pages: write`
  - `id-token: write`

## 9. Maintainability

### 9.1 Readability

- Use a clear workflow filename. The current recommendation is `.github/workflows/deploy-docs.yml`.
- Job and step names should explain what each step does.
- The workflow should be understandable without reading external notes.

### 9.2 Portability

- Avoid coupling the workflow to recommender system experiments.
- Keep the path rules centered on documentation files.

### 9.3 Extensibility

- Future additions may include link checking, formatting checks, or versioned documentation.
- Do not add those until they are needed.

## 10. Non-functional requirements

### 10.1 Simplicity

- Keep the workflow direct.
- Do not add extra jobs just to make the pipeline look more complete.

### 10.2 Consistency

- Workflow naming, step naming, and trigger rules should match the documentation structure.

### 10.3 Reliability

- Use stable Actions.
- Avoid temporary scripts and implicit environment assumptions.

## 11. Deliverables

The implementation must include:

- A GitHub Actions workflow file.
- Clear build steps.
- Clear deployment steps.
- Trigger rules that match the README based documentation source.
- Required permissions for GitHub Pages.
- Short comments or documentation where useful.

## 12. Acceptance criteria

The task is complete when:

1. The repository has a standalone documentation deployment workflow.
2. The workflow runs after relevant changes on the main branch.
3. The workflow supports manual triggering.
4. The workflow checks out code, prepares the environment, installs dependencies, and builds the site.
5. The workflow uploads and deploys the static site artifact.
6. The deployment target is GitHub Pages.
7. The workflow is easy to read and maintain.
8. Changes to any `README.md` or `README.zh.md` trigger a documentation build.
9. The workflow can be reused in similar repositories.

## 13. Implementation constraints

- Do not mix unrelated CI/CD tasks into this workflow.
- Do not trigger deployment for every code change.
- Do not omit README trigger paths.
- Do not omit manual triggering.
- Do not skip the basic build and deployment chain.
- Do not add unnecessary complexity.

## 14. Implementation instruction

Implement a GitHub Actions workflow that:

- Builds the static MkDocs documentation site.
- Supports automatic triggers on relevant main branch changes.
- Supports manual dispatch.
- Deploys to GitHub Pages.
- Uses a clear, stable, maintainable structure.
- Keeps README files as documentation content sources.

## 15. Expected result

The repository should have a focused documentation deployment workflow that can build and publish the current MkDocs site and can also serve as a reusable pattern for similar README based documentation sites.

