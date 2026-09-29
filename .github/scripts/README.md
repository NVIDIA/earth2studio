# Testmon CI baselines

`e2s-ci-test-main.yml` runs on `main` and `*-rc` pushes, or manually via
`workflow_dispatch`. Each test suite restores and updates its own branch-specific
testmon cache. RC runs also publish the database directory (including hidden
SQLite files) as `testmon-<branch>-<tox-env>` artifacts retained for 30 days.

`e2s-ci-test-pr.yml` runs on copied `pull-request/<number>` branches. It resolves
the original PR's target branch through the GitHub API and uses that branch for
both changed-file detection and testmon baseline selection. PR runs do not
publish baselines.

GitHub cache scope permits these push-triggered PR copies to read `main` caches,
but not sibling RC branch caches. RC-targeting copies therefore download the
newest unexpired database artifact for their suite from a run on the target
branch. Artifacts from other branches are excluded. If no baseline exists or a
download fails, tests run without that optimization; there is no fallback to a
different branch's database.

After deploying this change, run `e2s-ci-test-main` on `1.0.0-rc` once to seed its
databases. Subsequent RC pushes refresh them. The first run for a new RC branch
is expected to run its full test suites.

Run the baseline resolver tests locally with:

```sh
node --test .github/scripts/testmon-baseline.test.cjs
```
