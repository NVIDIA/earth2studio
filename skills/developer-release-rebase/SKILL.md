---
name: developer-release-rebase
version: 1.0.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - release
    - versioning
    - changelog
description: >
  Prepare a new minor alpha release of Earth2Studio by rebasing the release
  candidate branch onto main, bumping the version, updating the changelog,
  updating the README latest-news highlights, stripping example version tags, and
  pushing for PR. Use when releasing, cutting a release, preparing a release
  branch, rebasing a release, or bumping the version for a new development cycle.
---

# Release Rebase — Prepare New Minor Alpha Release

Prepare a new minor alpha release of Earth2Studio by rebasing the release
candidate branch onto main, bumping the version, updating the changelog,
updating the README latest-news highlights, stripping pinned version tags
from examples, and pushing a PR branch.

Follow every step below **in order**. Each step that requires user input is
marked with a confirmation gate — wait for explicit approval before proceeding.

---

## Step 1 — Determine the Next Version

1. Read `earth2studio/__init__.py` to get the current `__version__` string.
2. Compute the **next minor alpha**: if current is `X.Y.*-rc*` (or `X.Y.0rcN`),
   the next version is `X.(Y+1).0a0`.
3. If the branch has already been rebased in a prior session, the version may
   already reflect the new alpha — note this.

### **[CONFIRM — Version]**

Print both versions (current and target) and ask the user to confirm before
proceeding.

---

## Step 2 — Verify Remotes and Rebase

### 2a — Check remotes

Confirm:

- `origin` points to the user's **fork** of Earth2Studio.
- `upstream` points to `https://github.com/NVIDIA/earth2studio.git` (or the SSH equivalent).

If either is wrong, stop and ask the user to fix it.

### 2b — Create or verify the rebase branch

If the repo is already on a branch named `X.Y.0-rebase`, confirm with the user
that it was created from the `X.Y.*-rc` branch and skip the checkout commands.

Otherwise, create the rebase branch:

```bash
git fetch upstream X.Y.*-rc
git checkout X.Y.*-rc
git checkout -b X.Y.0-rebase
```

(Replace `X.Y.*-rc` with the actual tag/branch name matching the current
release candidate.)

### 2c — Rebase onto main (if needed)

Before rebasing, complete Step 3 on the release branch if final notes have not
been assembled. This prevents consuming fragments for the next development cycle
that may already be on main. Commit the assembled notes and fragment deletions
before the rebase, with user approval.

Check whether the rebase branch already has `main` as an ancestor:

```bash
git merge-base --is-ancestor main HEAD && echo "Already rebased" || echo "Needs rebase"
```

If a rebase is needed:

```bash
git checkout main
git pull upstream main
git checkout X.Y.0-rebase
git rebase main
```

If the branch is already rebased, skip this and proceed.

---

## Step 3 — Assemble Release Notes

Follow [`changes/README.md`](../../changes/README.md). For an RC, preview only:

```bash
uv run --only-dev towncrier build --draft --version X.Y.0rcN
```

For the final release, first check whether `CHANGELOG.md` already contains that
stable version. If so, do not build it again. Otherwise, assemble once from the
release branch's fragments using the stable version and confirmed date:

```bash
uv run --only-dev towncrier build --yes --version X.Y.0 --date YYYY-MM-DD
```

Commit the generated changelog together with consumed-fragment deletions. Preserve
published history, and do not insert a blank section for the next alpha version.
Do not consume next-cycle fragments that reached main after the release branch
diverged; assemble on the release branch before rebasing when that is the case.

### **[CONFIRM — Release Date]**

Before a final build, ask the user what release date to use.

---

## Step 4 — Bump the Package Version

Run these two commands in sequence:

```bash
uv run hatch version minor
uv run hatch version alpha
```

After running, read `earth2studio/__init__.py` and confirm it now contains
`X.(Y+1).0a0`.

If the pre-commit hook `pyupgrade` fails due to a Python version incompatibility
(a known issue with Python 3.14), it is safe to skip with
`SKIP=pyupgrade` on the subsequent commit step — note this to the user.

---

## Step 5 — Strip Version Tags from Examples

Remove pinned `@X.Y.Z` git tags from all example install blocks:

```bash
find examples/ -type f -exec sed -i 's/@[0-9]\+\.[0-9]\+\.[0-9]\+[a-z0-9]*//g' {} \;
```

Show a `git diff --stat examples/` summary so the user can verify the changes
look correct.

---

## Step 6 — Update Install Guide Version Tag

Update the install documentation to reference the new released version tag.

1. Open `docs/userguide/about/install_options.yml` and update the `release_ref`
   field under `package:` from the **previous** release version to the **new**
   released version (e.g., `release_ref: 0.15.0`). This field drives the
   install selector widget that generates install commands on the docs site.
2. Open `docs/userguide/about/install.md` and replace all occurrences of the
   **previous** release tag (e.g., `@0.14.0`) with the **new** release tag
   (e.g., `@0.15.0`) in the hardcoded install examples (uv project, Docker,
   extras note).
3. Update the Docker container tag (e.g., `nvcr.io/nvidia/pytorch:XX.YY-py3`) to
   the latest recommended container version if it has changed.
4. Show a `git diff docs/userguide/about/` summary so the user can verify the
   changes look correct.

---

## Step 7 — Update README Latest News

Update the "Latest News" section in `README.md` with highlights from the
**released** version's CHANGELOG entry assembled in Step 3.

1. Read `CHANGELOG.md` and identify the 3–5 most notable items from the
   released version's `### Added` subsection. Prefer items that introduce new
   model classes, new data sources, or significant new capabilities.
2. Read the current `## Latest News` section in `README.md`.
3. Replace the existing bullet points (between the `> [!NOTE]` block and the
   "For a complete list…" line) with new bullets summarising the highlights.
   Follow the existing style:
   - Each bullet starts with a bold linked feature name (where a docs link
     exists) followed by a comma and a short description.
   - Keep the `> [!NOTE]` version callout and update it to reference the new
     released version number if it changed.
4. Show the diff to the user for review.

### **[CONFIRM — README]**

Print the updated Latest News section and ask the user to confirm before
proceeding.

---

## Step 8 — Update Skill Versions

**Skip this step.** Skill versions are managed separately and should NOT
be updated during the release rebase process.

---

## Step 9 — Update GitHub Issue Templates

Update the suggested version placeholder in the bug report template to
reference the new released version.

1. Open `.github/ISSUE_TEMPLATE/bug_report.yml`.
2. Replace the `placeholder:` value (e.g., `"example: 0.14.0"`) with
   `"example: X.Y.0"` (the new released version).
3. Show the diff to the user for review.

---

## Step 10 — Commit and Push

Stage only the expected files and commit:

```bash
git add CHANGELOG.md changes/
git add earth2studio/__init__.py
git add examples/
git add README.md
git add docs/userguide/about/install.md
git add docs/userguide/about/install_options.yml
git add skills/
git add .github/
git commit -m "Update version to X.(Y+1).0a0"
```

If pre-commit hooks fail due to a tool incompatibility (not a code issue), use
`SKIP=<hook-id>` to bypass the broken hook and retry.

Push the branch to origin:

```bash
git push origin X.Y.0-rebase
```

### **[REMIND — Merge Strategy]**

After pushing, remind the user:

> **Use Rebase merge, not Squash**, when merging the PR.
