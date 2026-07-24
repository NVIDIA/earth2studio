---
name: developer-bump-version
version: 0.16.0
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
  Bump the Earth2Studio version on main to start a new development cycle. This
  unblocks commits on main after a release branch has been merged. Performs a
  CHANGELOG update (new blank section) and a hatch version bump. Use when main
  already has the release merge but still carries the old version string, or
  when the developer-release-rebase skill detects the version was already bumped
  during rebase and only the changelog/version housekeeping remains.
---

# Bump Version — Start New Development Cycle on Main

Bump the Earth2Studio package version on `main` to the next minor alpha and
insert a blank CHANGELOG section. This is the minimal set of changes needed to
unblock further development commits on main after a release branch merge.

Follow every step below **in order**. Each confirmation gate requires explicit
user approval before proceeding.

---

## Step 1 — Determine the Next Version and Create Branch

1. Read `earth2studio/__init__.py` to get the current `__version__` string.
2. Compute the **next minor alpha**:
   - If current is `X.Y.0` (release), next is `X.(Y+1).0a0`.
   - If current is `X.Y.0rcN` or `X.Y.0aN`, next is `X.(Y+1).0a0`.
   - If the version already ends in `a0` at the expected next minor, the bump
     may already be done — note this and confirm with the user.
3. Read `CHANGELOG.md` and check whether a section for the target version
   already exists. If it does, the bump is already complete — inform the user
   and stop.

### **[CONFIRM — Version]**

Print the current version, the target version, and ask the user to confirm
before proceeding.

### 1b — Create or verify the bump branch

Check the current branch name:

```bash
git branch --show-current
```

If already on a branch named `version-bump-X.(Y+1).0a0`, skip branch creation.

Otherwise, create a new branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b version-bump-X.(Y+1).0a0
```

(Replace `X.(Y+1).0a0` with the actual computed target version.)

---

## Step 2 — Update CHANGELOG.md

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
2. Does **not** have the alpha/rc extension in its version (e.g., `[0.16.0]`
   not `[0.16.0a0]`).
3. Has a release date set in `YYYY-MM-DD` format.

### **[CONFIRM — Release Date]**

If the use doesn't know, just keep it as is, make sure to tell user this is an option.
If the released section does not already have a date set, ask the user what
date to use before proceeding.

---

## Step 3 — Bump the Package Version

Run these two commands in sequence:

```bash
uv run hatch version minor
uv run hatch version alpha
```

After running, read `earth2studio/__init__.py` and confirm it now contains the
expected `X.(Y+1).0a0` version string.

If a pre-commit hook (`pyupgrade` or similar) fails due to a Python version
incompatibility, it is safe to skip with `SKIP=pyupgrade` on the subsequent
commit step — note this to the user.

---

## Step 4 — Commit and Push

Stage only the expected files and commit:

```bash
git add CHANGELOG.md earth2studio/__init__.py
git commit -m "Bump version to X.(Y+1).0a0"
```

Push the branch (or the current branch if already on a feature branch):

```bash
git push origin HEAD
```

### **[REMIND — PR or Direct Push]**

After pushing, remind the user:

> If main is protected, open a PR titled **"Bump version to X.(Y+1).0a0"**
> targeting `main`. Otherwise the direct push is sufficient.
