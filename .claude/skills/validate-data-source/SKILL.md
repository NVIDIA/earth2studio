---
name: validate-data-source
description: Validate a newly created Earth2Studio data source by running tests, checking coverage, validating variables against real data, generating sanity-check plots, and opening a PR with automated code review. Use this skill after completing data source implementation (create-data-source skill Steps 0-12) to run the full validation, submission, and review pipeline.
argument-hint: Name of the data source class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Data Source

> **Python Environment:** This project uses **uv** for dependency
> management. Always use the local `.venv` virtual environment
> (`source .venv/bin/activate` or prefix with `uv run python`) for all
> Python commands — installing packages, running tests, executing
> scripts, etc. Use `uv add` / `uv pip install` / `uv lock` instead of `pip install`.

Validate a newly created Earth2Studio data source by running tests,
checking coverage, validating variables, generating sanity-check plots,
and opening a PR. This skill picks up after the `create-data-source`
skill completes implementation (Steps 0-12).

Each confirmation gate marked by:

```markdown
### **[CONFIRM — <Title>]**
```

requires **explicit user approval** before proceeding.

---

## Step 1 — Run Tests

### 1a. Run the new test file

```bash
uv run python -m pytest test/data/test_<filename>.py -v --timeout=60
```

All tests must pass (or `xfail` for network tests). Fix failures
and re-run until green.

### 1b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include network tests. The new data source file must achieve
**at least 90% line coverage**:

```bash
uv run python -m pytest test/data/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/data/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables network tests (marked `@pytest.mark.slow`)
- `--cov=earth2studio/data/<filename>` scopes coverage to the
  new source module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to
cover the missing lines. Common gaps:

- Error handling branches (e.g., empty products, invalid data)
- Edge cases in parsing (e.g., missing fields, corrupt records)
- Cache property paths (`cache=True` vs `cache=False`)
- `resolve_fields` with different input types

Re-run until coverage is at or above 90%.

### 1c. Run the full data test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-data
```

Confirm no regressions.

---

## Step 2 — Validate Variables, Summarize, and Sanity-Check Plots

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo.

### 2a. Script templates

**For DataSource / ForecastSource** (gridded data):

```python
"""Sanity-check plot for <SourceName> data source.

This script is for PR review only — do NOT commit to the repo.
Run it to produce a quick visualization confirming the source works.
"""
import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import SourceName

# Fetch a sample
ds = SourceName(cache=False)
time = ...  # pick a recent valid time
variables = ["t2m", "msl", "u10m"]  # 1-3 representative variables
data = ds(time, variables)

# Plot contours for each variable
fig, axes = plt.subplots(1, len(variables), figsize=(6 * len(variables), 5))
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    im = data.sel(variable=var).isel(time=0).plot(ax=ax, cmap="turbo")
    ax.set_title(f"{var}")

plt.suptitle(f"<SourceName> — {time}", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<source_name>.png")
```

**For DataFrameSource / ForecastFrameSource** (sparse data):

```python
"""Sanity-check plot for <SourceName> data source.

This script is for PR review only — do NOT commit to the repo.
Run it to produce a quick visualization confirming the source works.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import SourceName

# Fetch a sample
ds = SourceName(cache=False)
time = ...  # pick a recent valid time
variables = ["var1", "var2"]  # 1-3 representative variables
df = ds(time, variables)

# Convert lon from [0,360] to [-180,180] for cartopy
df["lon_plt"] = df["lon"].where(df["lon"] <= 180, df["lon"] - 360)

# Plot scatter on globe for each variable (cartopy projection)
fig, axes = plt.subplots(
    1, len(variables),
    figsize=(8 * len(variables), 5),
    subplot_kw={"projection": ccrs.Robinson()},
)
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    subset = df[df["variable"] == var]
    obs = subset["observation"].values
    # Use percentile clipping to avoid outliers blowing out the colorbar
    vmin, vmax = np.percentile(obs[np.isfinite(obs)], [2, 98])

    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

    sc = ax.scatter(
        subset["lon_plt"], subset["lat"],
        c=obs, s=2, cmap="turbo", alpha=0.8,
        vmin=vmin, vmax=vmax, edgecolors="none",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{var} ({len(subset)} obs)\nrange: {vmin:.0f}–{vmax:.0f} (p2–p98)")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Observation",
                 orientation="horizontal", pad=0.05)

plt.suptitle(f"<SourceName> — {time}", y=1.0)
plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<source_name>.png")
```

> **Tip:** Always use `cartopy` with a proper map projection
> (Robinson, PlateCarree, etc.) for geospatial scatter plots.
> Without a projection, satellite swath data looks distorted and
> hard to interpret. Use percentile-clipped `vmin`/`vmax` and
> `s >= 2` marker size — without clipping, outliers compress the
> colorbar and make the plot appear blank.

### 2b. Validate all lexicon variables

**Every variable in the lexicon must be validated against real data.**
A variable that exists in the lexicon but consistently produces missing
or invalid data should be **removed** from the lexicon and
`E2STUDIO_VOCAB`.

Run a validation script that fetches a representative sample and
checks every variable:

```python
"""Variable validation for <SourceName>.

Checks every lexicon variable against real data to confirm it
returns valid observations. Variables with very low valid-data
rates should be removed from the lexicon.
"""
from datetime import datetime

from earth2studio.data import SourceName
from earth2studio.lexicon import SourceNameLexicon

ds = SourceName(cache=True)
time = ...  # pick a recent valid time
all_vars = list(SourceNameLexicon.VOCAB.keys())
df = ds(time, all_vars)

print(f"{'Variable':<16} {'Obs Count':>10} {'Valid %':>8} {'Min':>10} {'Max':>10}")
print("-" * 60)
for var in sorted(all_vars):
    sub = df[df["variable"] == var]
    n_total = len(sub)
    n_valid = sub["observation"].notna().sum()
    pct = (n_valid / n_total * 100) if n_total > 0 else 0
    vmin = sub["observation"].min() if n_valid > 0 else float("nan")
    vmax = sub["observation"].max() if n_valid > 0 else float("nan")
    flag = " *** REMOVE" if pct < 10 else ""
    print(f"{var:<16} {n_total:>10} {pct:>7.1f}% {vmin:>10.2f} {vmax:>10.2f}{flag}")
```

**Action required:**

- Variables with **< 10% valid data** must be removed from:
  1. The lexicon class `VOCAB` dict
  2. `E2STUDIO_VOCAB` in `earth2studio/lexicon/base.py`
  3. The class docstring (note the removal and reason)
- Variables with **10-50% valid data** should be kept but documented
  with a note explaining the low coverage (e.g., day/night switching,
  quality filtering)
- After removing variables, re-run `make lint` and tests

### 2c. Summarize variables and time range to user

Before generating sanity-check plots, **present a summary table** to
the user covering:

1. **All valid variables** — name, description, observation count,
   value range
2. **Removed variables** — name, reason for removal (e.g., "> 97%
   missing data")
3. **Valid time range** — earliest and latest data available from the
   remote store
4. **Typical data density** — observations per orbit/file,
   approximate global coverage cadence

Example summary format:

> **Variable Summary for MetOpAMSUA (14 channels):**
>
> | Variable | Freq (GHz) | Layer | Obs/orbit | BT Range (K) |
> |----------|-----------|-------|-----------|---------------|
> | amsua01 | 23.8 | Surface | 22,950 | 142–291 |
> | amsua02 | 31.4 | Surface | 22,950 | 145–290 |
> | ... | | | | |
>
> **Removed:** `amsua15` (89.0 GHz) — L1B product marks ~97% of
> measurements as missing due to quality filtering.
>
> **Time range:** 2007-10-22 (Metop-A launch) to present.
> Metop-B (2012-09-17 to present), Metop-C (2018-11-07 to present).
>
> **Data density:** ~767 scan lines × 30 FOVs = 23,010 obs per orbit,
> ~14 orbits/day, global coverage twice daily.

This summary helps the user understand what they are getting from the
data source and verify it matches their expectations.

### 2d. Create sanity-check plot script

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo. See 2a above for templates.

### 2e. Run the script

Execute the script and **verify the output images exist and look
reasonable**:

```bash
python sanity_check_<source_name>.py
```

Check that:

- The script runs without errors
- Output PNGs are generated
- Data points appear in expected geographic regions
- Values are in physically reasonable ranges (e.g., temperatures
  200–320 K, not 0 or NaN)
- For DataFrame sources: observations form recognizable swath or
  station patterns

### 2f. **[CONFIRM — Sanity-Check Plots]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the script ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
ask them to open and inspect the output:

> The sanity-check script ran successfully and saved the following
> plot:
>
> `/absolute/path/to/sanity_check_<source_name>.png`
>
> **Please open this image and confirm it looks correct.** Check:
>
> 1. Data points are visible on the axes (not blank/empty)
> 2. Geographic coverage matches expectations (global swaths,
>    regional stations, etc.)
> 3. Colorbar values are in physically reasonable ranges
> 4. No obvious artifacts (all-NaN regions, garbled coordinates)
>
> Does the plot look correct?

**Do not proceed to Step 3 until the user explicitly confirms the
plots look correct.** If the user reports problems (empty plots,
wrong ranges, missing coverage), debug and fix the issue, re-run
the script, and ask the user to inspect again.

The output images will be uploaded to the PR description in
Step 3.

---

## Step 3 — Branch, Commit & Open PR

### **[CONFIRM — Ready to Submit]**

Before proceeding, confirm with the user:

 > All implementation steps are complete:
 >
 > - Data source class implemented with correct method ordering
 > - Lexicon class created with variable mappings
 > - **All lexicon variables validated against real data** (low-validity
 >   variables removed)
 > - **Variable summary and time range presented to user**
 > - E2STUDIO_VOCAB / E2STUDIO_SCHEMA updated (if needed)
 > - Registered in `__init__.py` files
 > - Documentation RST files updated with badges
 > - Reference URLs included in docstrings
 > - CHANGELOG.md updated
 > - Format, lint, and license checks pass
 > - Unit tests written and passing
 > - Dependencies in `data` extras group confirmed
 > - Sanity-check plots generated and confirmed by user
 >
 > Ready to create a branch, commit, and prepare a PR?

### 3a. Create branch and commit

```bash
git checkout -b feat/data-source-<name>
git add earth2studio/data/<filename>.py \
        earth2studio/data/__init__.py \
        earth2studio/data/utils.py \
        earth2studio/lexicon/<filename>.py \
        earth2studio/lexicon/__init__.py \
        earth2studio/lexicon/base.py \
        test/data/test_<filename>.py \
        docs/modules/datasources_*.rst \
        pyproject.toml \
        CHANGELOG.md
git commit -m "feat: add <SourceName> data source

Add <SourceName> <source_type> for <brief description>.
Includes lexicon, unit tests, and documentation."
```

Do **NOT** add the sanity-check script or its output image.

### 3b. Identify the fork remote and push branch

The working repository is typically a **fork** of
`NVIDIA/earth2studio`. Before pushing, confirm which git remote
points to the user's fork:

```bash
git remote -v
```

Ask the user:

> Which git remote is your fork of `NVIDIA/earth2studio`?
> (Usually `origin` — e.g., `git@github.com:<user>/earth2studio.git`)

Then push the feature branch to the **fork** remote:

```bash
git push -u <fork-remote> feat/data-source-<name>
```

### 3c. Open Pull Request (fork → NVIDIA/earth2studio)

> **Important:** PRs must be opened **from the fork** to the
> **upstream source repository** `NVIDIA/earth2studio`. The branch
> lives on the fork; the PR targets `main` (or the appropriate
> base branch) on the upstream repo.

Use `gh pr create` with explicit `--repo` and `--head` flags:

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/data-source-<name> \
  --title "feat: add <SourceName> data source" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork
(e.g., `NickGeneva`).

The PR should follow the repo's template and
include the data licensing section, the sanity-check script inside
a `<details>` block, and the sanity-check images uploaded to the
PR body:

````markdown
## Description

Add `<ClassName>` <source_type> for <brief description of what
the data source provides>.

Closes #<issue_number> (if applicable)

### Data source details

| Property | Value |
|---|---|
| **Source type** | DataSource / ForecastSource / DataFrameSource / ForecastFrameSource |
| **Remote store** | <URL to data documentation> |
| **Format** | GRIB2 / NetCDF / Zarr / CSV / etc. |
| **Spatial resolution** | X deg x Y deg (NxM grid) |
| **Temporal resolution** | Hourly / 6-hourly / daily / etc. |
| **Date range** | YYYY-MM-DD to present |
| **Region** | Global / Regional |
| **Authentication** | Anonymous / API key / etc. |

### Data licensing

> **License**: <Name of the data license>
> **URL**: <Link to the license terms>
>
> <Brief summary of key restrictions or permissions, e.g.,
> "Open data, freely available for commercial and
> non-commercial use" or "Requires attribution, non-commercial
> use only">

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `<package>` | `>=X.Y` | <License name> | [link](<URL to license>) | <brief reason> |

*(or "No new dependencies needed")*

When filling this table, look up each new dependency's license:

1. Check the package's PyPI page (`https://pypi.org/project/<package>/`)
   or its repository for the license type
2. Link directly to the license file (e.g., GitHub raw LICENSE or
   PyPI license classifier URL)
3. Flag any **non-permissive licenses** (GPL, AGPL, SSPL) — these
   may be incompatible with the project's Apache-2.0 license and
   require team review before merging

### Validation

See sanity-check validation in PR comments below (posted in Step 3d).

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues

## Dependencies

<List any new packages added to pyproject.toml, or "None">
````

### 3d. Post sanity-check plot as PR comment

After the PR is created, post the sanity-check visualization as a
separate **PR comment** so it is immediately visible to reviewers
without expanding a `<details>` block. This serves as quick visual
evidence that the data source produces correct output.

#### Image upload limitation

**GitHub has no CLI or REST API for uploading images to PR comments.**
The uploads endpoint (`uploads.github.com`) and GraphQL mutations do
not support attaching local image files to issue/PR comments. The
only way to embed an image is via the browser's drag-and-drop editor
or by referencing an already-hosted URL.

**Practical workflow:**

1. Post the PR comment **without** the image — include the
   validation table, the full sanity-check script, and a placeholder
   line: `> **TODO:** Attach sanity-check image by editing this
   comment in the browser.`
2. Tell the user: *"The image is at `<local_path>`. Edit the PR
   comment in your browser and drag the file into the editor to
   embed it."*
3. Use `gh api -X PATCH` to update the comment body (via
   `-F body=@/tmp/comment.md`) if the text needs revision later.

Do **not** waste time trying `curl` uploads, GraphQL file mutations,
or the `uploads.github.com` asset endpoint — they do not work for
issue/PR comment images.

#### Writing the comment body

Write the comment body to a temp file first, then post it. This
avoids shell quoting issues with heredocs containing backticks and
markdown:

```bash
# 1. Write body to a temp file (use your editor tool, not heredoc)
#    Include: summary table, key findings, script in <details>

# 2. Post the comment
gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
  -F "body=@/tmp/pr_comment_body.md" \
  --jq '.html_url'
```

#### Comment content template

Adapt to the source type. Include only relevant rows — omit
satellite-specific rows for gridded sources, omit grid dimensions
for DataFrame sources, etc.

```markdown
## Sanity-Check Validation

**Source:** `<ClassName>` — <brief description>
**Time:** YYYY-MM-DD HH:MM UTC
**Parameters:** <source-specific context, e.g., tolerance, satellite, product>

| Metric | Value |
|--------|-------|
| Output shape / rows | <shape for DataArray, row count for DataFrame> |
| Variables | <list or count> |
| Value range | <min>–<max> <unit> |
| Lat range | <min>–<max>° |
| Lon range | <min>–<max>° |
| Missing / NaN | <count or 0> |

*Add source-specific rows as needed (channels, satellites, grid
dimensions, lead times, etc.)*

**Key findings:**
- <bullet summarizing physical reasonableness>
- <bullet on spatial pattern / coverage>
- <bullet on comparison with reference source, if applicable>

> **TODO:** Attach sanity-check image by editing this comment in
> the browser.

<details>
<summary>Sanity-check script (click to expand)</summary>

PASTE THE FULL WORKING SCRIPT HERE — not a truncated excerpt.
The script must be copy-pasteable and produce the plot end-to-end.

</details>
```

**Important:** Always paste the **complete, runnable** script — not
a shortened version. Reviewers should be able to reproduce the plot
by copying the script directly.

#### Comparison sources

If a **comparison data source** exists for the same physical
quantity (e.g., UFSObsSat for satellite BT, ERA5 for reanalysis
fields), include a side-by-side comparison in the same comment.
Briefly summarize expected differences:

- Raw vs QC-subsetted data (different observation counts)
- Different spatial coverage or resolution
- Different time windows or update cadence
- Unit or coordinate convention differences

#### Finalize

After posting, inform the user of:

1. The comment URL
2. The local path to the image file for manual attachment
3. Instructions: *"Edit the comment in your browser and drag the
   image file into the editor to embed it."*

> **Note:** The sanity-check image and script are for PR review
> purposes only — they must NOT be committed to the repository.

---

## Step 4 — Automated Code Review (Greptile)

After the PR is created and pushed, an automated code review from
**greptile-apps** (Greptile) will be posted as PR review comments.
Wait for this review, then process the feedback.

### 4a. Wait for Greptile review

Poll for review comments from `greptile-apps[bot]` every 30 seconds
for up to **5 minutes**. Time out gracefully if no review arrives:

```bash
# Poll loop — check every 30s, timeout after 5 minutes (10 attempts)
for i in $(seq 1 10); do
  REVIEW_ID=$(gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
    --jq '.[] | select(.user.login == "greptile-apps[bot]") | .id' 2>/dev/null)
  if [ -n "$REVIEW_ID" ]; then
    echo "Greptile review found: $REVIEW_ID"
    break
  fi
  echo "Attempt $i/10 — no review yet, waiting 30s..."
  sleep 30
done
```

If no review after 5 minutes, inform the user:

> Greptile hasn't posted a review after 5 minutes. This can happen if
> the review bot is busy or the PR hasn't triggered it. You can:
>
> 1. Ask me to check again later
> 2. Skip this step and proceed without automated review
> 3. Manually request a review from Greptile on the PR page

### 4b. Pull and parse review comments

Once the review is posted, fetch all comments:

```bash
# Get all review comments on the PR
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") |
    {path: .path, line: .diff_hunk, body: .body}'
```

Also fetch the top-level review body:

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | .body'
```

### 4c. Categorize and present to user

Parse each comment and categorize it:

| Category | Description | Default action |
|---|---|---|
| **Bug / correctness** | Logic errors, wrong behavior | Fix |
| **Style / convention** | Naming, formatting, patterns | Fix if valid |
| **Performance** | Inefficiency, resource waste | Evaluate |
| **Documentation** | Missing/wrong docs, docstrings | Fix |
| **Suggestion** | Alternative approach, nice-to-have | User decides |
| **False positive** | Incorrect or irrelevant feedback | Dismiss |

### **[CONFIRM — Review Triage]**

Present each comment to the user in a summary table:

```markdown
| # | File | Line | Category | Summary | Proposed Action |
|---|------|------|----------|---------|-----------------|
| 1 | metop_amsua.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | metop_avhrr.py | 305 | Style | Use f-string | Fix: convert |
| 3 | metop.py | 45 | Suggestion | Add type alias | Skip: not needed |
| ... | ... | ... | ... | ... | ... |
```

For each comment, briefly explain:

- What Greptile flagged
- Whether you agree or disagree (with reasoning)
- Your proposed fix (or why to skip)

Ask the user to confirm which comments to address. The user may:

- Accept all proposed fixes
- Select specific fixes
- Override your recommendation on any comment
- Add their own fixes

### 4d. Implement fixes

For each accepted fix:

1. Make the code change
2. Run `make format && make lint` after all fixes
3. Run the relevant tests
4. Commit with a message like:
   `fix: address code review feedback (Greptile)`

### 4e. Respond to review comments

For each Greptile comment, post a reply on the PR:

**For fixed comments:**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <brief description of fix>"
```

**For dismissed comments (false positives / won't fix):**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <brief justification>"
```

### 4f. Push and resolve

```bash
git push origin <branch>
```

After pushing, resolve all addressed review threads if possible.

Inform the user of the final state:

- How many comments were fixed
- How many were dismissed (with reasons)
- Any remaining open threads

---

## Reminders

- **DO** use the repos local `uv` `.venv` to run python with `uv run python`
- **DO NOT** commit the sanity-check script or image to the repo
- **DO** use `loguru.logger` for logging, never `print()`, inside
  `earth2studio/`
- **DO** ensure all public functions have full type hints (mypy-clean)
- **DO** maintain alphabetical order in `__init__.py` exports,
  RST file entries, and CHANGELOG entries
- **DO** follow the canonical method ordering within the class
- **DO** use the async task dataclass pattern for parallel fetching
- **DO** use `prep_data_inputs` / `prep_forecast_inputs` to normalize
  inputs
- **DO** use `nest_asyncio.apply()` in `__init__` for notebook compat
- **DO** use `datasource_cache_root()` for cache paths
- **DO** delete temporary cache when `cache=False`
- **DO** add all new dependencies to the `data` extras group in
  `pyproject.toml` using `uv add --extra data <package>`
- **DO** include reference URLs in class docstrings and lexicon docs
- **DO** always update CHANGELOG.md under the current unreleased version
- **DO** use async utilities from `earth2studio/data/utils.py`:
  `managed_session`, `gather_with_concurrency`, `async_retry`
- **DO** use pure async I/O (`fs._cat_file()`, async zarr, etc.)
- **PREFER** fsspec-compatible filesystems (s3fs, gcsfs, adlfs, etc.)
  over dedicated client libraries
- **AVOID** `asyncio.to_thread` — pure async is ALWAYS preferred.
  Only use `cancellable_to_thread` as a last resort when no async
  alternative exists.
- **AVOID** bare `tqdm.gather(*tasks)` — use `gather_with_concurrency`
- **AVOID** using xarray for loading data — prefer direct file I/O
- **AVOID** downloading full files — use byte-range / slicing
- **AVOID** over-complicating constructor parameters
- **NEVER** call `loop.set_default_executor()` — mutates global state
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git. Always read
  credentials from environment variables at runtime. If the user
  provides test credentials during the session, use them only in
  ephemeral shell commands — never persist them.
