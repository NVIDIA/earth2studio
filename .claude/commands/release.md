Prepare a new minor alpha release of Earth2Studio by following the steps below exactly.

## Step 1 - Determine the next version and rebase

Read `earth2studio/__init__.py` to get the current version string.
Compute the **next minor alpha**: if current is `X.Y.*-rc`, the next is `X.(Y+1).0a0`.

Depending on the rebase process on top of main, the version of the package might already
be updated to a new alpha version.

Print both versions so the user can confirm before proceeding.

## Step 2 - Rebase

Confirm that git remote `origin` is set to a fork of Earth2Studio

And confirm that git remote `upstream` is set to `git@github.com:NVIDIA/earth2studio.git`

Create a rebase branch, if the repo is already on a branch `X.Y.*-rebase` confirm with
the user that a rebase branch is already created from `X.Y.*-rc` and skip the checkout
commands below.

```bash
git fetch upstream X.Y.*-rc
git checkout X.Y.*-rc
git checkout -b X.Y.*-rebase
```

Before attempting to rebase, check to see if the branch is already rebased based on
the commit history of the current rebase branch.

If a rebase is needed, follow the next sets to ensure main is up to date.

```bash
git checkout main
git pull upstream main
git checkout X.Y.*-rebase
git rebase main
```

## Step 3 - Update CHANGELOG.md

Read `CHANGELOG.md` and insert a new blank section **above** the most recent version
entry. Use `xxxx-xx-xx` as the date placeholder — never fill in today's date.

The new section must look exactly like this (replacing the version number):

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

Also ensure that the following released section of the change log:

```markdown
## [X.(Y).0] - xxxx-xx-xx

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

```

1. Has an unused sections removed
2. Does not have the alpha extension in its version
3. Has a release date set YYYY-MM-DD

Confirm with user if the release date is not set.

## Step 4 - Bump the package version

Run these two commands in sequence:

```bash
uv run hatch version minor
uv run hatch version alpha
```

After running, read `earth2studio/__init__.py` and confirm it now contains `X.(Y+1).0a0`.

## Step 5 - Strip version tags from examples

Run this to remove pinned `@X.Y.Z` git tags from all example install blocks:

```bash
find examples/ -type f -exec sed -i 's/@[0-9]\+\.[0-9]\+\.[0-9]\+[a-z0-9]*//g' {} \;
```

Show a `git diff --stat examples/` summary so the user can verify.

## Step 6 - Commit and open a PR

Stage only the expected files and commit:

```bash
git add CHANGELOG.md
git add earth2studio/__init__.py
git add examples/
git commit -m "Update version to X.(Y+1).0a0"
```

Push the branch `X.Y.*-rebase` up to origin.

Remind the user: **use Rebase merge, not Squash**, when merging the PR.
