---
name: release-rebase
description: Prepare a new minor alpha release of Earth2Studio by rebasing the release candidate branch onto main, bumping the version, updating the changelog, stripping example version tags, and pushing for PR. Use this skill whenever the user mentions releasing, cutting a release, preparing a release branch, rebasing a release, bumping the version for a new development cycle, or running the release process. Also trigger when the user references "release-rebase" or asks about the release workflow.
---

# Release Rebase — Prepare New Minor Alpha Release

Prepare a new minor alpha release of Earth2Studio by rebasing the release
candidate branch onto main, bumping the version, updating the changelog,
stripping pinned version tags from examples, and pushing a PR branch.

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
- `upstream` points to `git@github.com:NVIDIA/earth2studio.git`.

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

## Step 3 — Update CHANGELOG.md

Read `CHANGELOG.md` and insert a new blank section **above** the most recent
version entry. Use `xxxx-xx-xx` as the date placeholder — never fill in today's
date for the new development version.

The new section must look exactly like this (substituting the version number):

```markdown
## [X.(Y+1).0a0] - xxxx-xx-xx

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

```

Also ensure the **released** version section (the one just below):

1. Has unused (empty) subsections removed.
2. Does **not** have the alpha/rc extension in its version (e.g., `[0.14.0]` not
   `[0.14.0a0]`).
3. Has a release date set in `YYYY-MM-DD` format.

### **[CONFIRM — Release Date]**

If the released section does not already have a date set, ask the user what date
to use before proceeding.

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

## Step 6 — Commit and Push

Stage only the expected files and commit:

```bash
git add CHANGELOG.md
git add earth2studio/__init__.py
git add examples/
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
