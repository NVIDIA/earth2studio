---
name: developer-bump-version
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
  Bump the Earth2Studio version on main to start a new development cycle. This
  unblocks commits on main after a release branch has been merged. Performs a
  hatch version bump. Use when main
  already has the release merge but still carries the old version string, or
  when the developer-release-rebase skill detects the version was already bumped
  during rebase and only the changelog/version housekeeping remains.
---

# Bump Version — Start New Development Cycle on Main

Bump the Earth2Studio package version on `main` to the next minor alpha and
leave changelog fragments for the next release. This is the minimal set of changes needed to
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
3. If the package already has the confirmed target version, the bump is complete.
   Do not use changelog sections to detect development versions.

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

## Step 2 — Check Release Notes

The latest published release should already be in `CHANGELOG.md`. If it is
missing, ask the user to complete final release assembly using
[`changes/README.md`](../../changes/README.md).

Do not add a blank alpha section, consume fragments, or rewrite published history
as part of a development version bump. New feature notes accumulate in `changes/`.

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

## Step 4 — Double-Check User-Facing Release References

Before committing, check that these references match the **released** version
in the latest published CHANGELOG section:

- `docs/userguide/about/install_options.yml`: `package.release_ref`.
- `docs/userguide/about/install.md`: release tags in hardcoded install commands.
- `.github/ISSUE_TEMPLATE/bug_report.yml`: the example version placeholder.

These can remain stale after a release merge even when the package version and
changelog are correct. Use the released version (for example, `1.0.0`), not the
new development version (`1.1.0a0`). Update stale references and report which
were changed or already correct. If the user limits the scope to particular
files, report any remaining stale references for follow-up.

---

## Step 5 — Commit and Push

Stage only the expected files and commit:

```bash
git add earth2studio/__init__.py
git add docs/userguide/about/install_options.yml docs/userguide/about/install.md
git add .github/ISSUE_TEMPLATE/bug_report.yml
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
