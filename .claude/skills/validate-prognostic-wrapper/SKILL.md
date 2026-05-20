---
name: validate-prognostic-wrapper
description: Validate a newly created Earth2Studio prognostic model wrapper by running tests with coverage, performing reference comparison (single-step and multi-step), generating sanity-check plots, and opening a PR with automated code review. Use this skill after completing the create-prognostic-wrapper and create-prognostic-tests skills.
argument-hint: Name of the prognostic model class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Prognostic Model Wrapper

Validate a newly created Earth2Studio prognostic model (px) wrapper by
running tests with coverage, performing reference comparison (single-step
and multi-step), generating sanity-check plots, and opening a PR with
automated code review.

**Prerequisites:** This skill assumes:

1. The prognostic wrapper has been implemented (`create-prognostic-wrapper` skill)
2. Unit tests have been written (`create-prognostic-tests` skill)

> **Python Environment:** This project uses **uv** for dependency
> management. Always use the local `.venv` virtual environment
> (`source .venv/bin/activate` or prefix with `uv run python`) for all
> Python commands — installing packages, running tests, executing
> scripts, etc. Use `uv add` / `uv pip install` / `uv lock` instead of
> `pip install`.

Each confirmation gate marked by:

```markdown
### **[CONFIRM — <Title>]**
```

requires **explicit user approval** before proceeding.

---

## Step 1 — Run Tests

### 1a. Run the new test file

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v --timeout=60
```

All tests must pass. Fix failures and re-run until green.

### 1b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include integration tests. The new prognostic model file must achieve
**at least 90% line coverage**:

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/px/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables integration tests (marked `@pytest.mark.slow`)
- `--cov=earth2studio/models/px/<filename>` scopes coverage to the
  new model module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to
cover the missing lines. Common coverage gaps for px models:

- Error handling in `output_coords` (wrong variable names, wrong dims)
- Device management paths (CPU vs CUDA)
- `create_iterator` edge cases (initial condition yield, hook calls)
- `load_model` and `load_default_package` (needs mock or package test)
- ONNX / non-PyTorch backend `.to()` logic

Re-run until coverage is at or above 90%.

### 1c. Run the full model test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-models
```

Confirm no regressions in existing model tests.

---

## Step 2 — Reference Comparison & Sanity-Check

This step validates the prognostic model wrapper produces correct
output by comparing against the original reference implementation
for both a single time step and a multi-step forecast, then
generating visual sanity-check plots.

### 2a. Create reference comparison script

Create a **standalone Python script** in the repo root using the template:

```text
.claude/skills/validate-prognostic-wrapper/reference-comparison-template.py.txt
```

Read the template, adapt it for the specific model:

- Fill in the reference model loading code (using only third-party packages)
- Update `ModelName` to the actual class name
- Adjust `N_STEPS` and tolerances as needed

The script compares E2S wrapper output against the original reference
for both single-step (`__call__`) and multi-step (`create_iterator`).
Do **NOT** commit this script to the repo.

### 2b. Summarize model capabilities to user

Before generating sanity-check plots, **present a summary table** to
the user covering the model's capabilities:

> **Model Summary for `<ClassName>`:**
>
> | Property | Value |
> |---|---|
> | **Input variables** | `var1`, `var2`, ... (N total) |
> | **Output variables** | `out1`, `out2`, ... (M total) |
> | **Time step** | Xh (e.g., 6h, 24h) |
> | **Spatial resolution** | X.XX deg x Y.YY deg (NxM) |
> | **History required** | None / Xh (e.g., -6h, 0h) |
> | **Checkpoint size** | XX MB |
> | **Checkpoint source** | NGC / HuggingFace / S3 |
> | **Inference time** | ~XX ms per step (on GPU/CPU) |

This summary helps the user verify the wrapper matches their
expectations for the model.

### **[CONFIRM — Data Source Selection]**

Before generating the sanity-check plot, ask the user which data source
to use for input data:

> Which data source would you like to use for the sanity-check plot?
>
> - `Random` — synthetic random data (fast, no download required)
> - `GFS` — GFS operational analysis
> - `ERA5` — ERA5 reanalysis (requires CDS credentials)
> - `HRRR` — HRRR analysis (CONUS only)
> - Other — specify a different data source
>
> The data source must support all variables required by the model.

Store the user's choice for use in the sanity-check script.

### 2c. Generate sanity-check plot script

Create a **standalone Python script** in the repo root using the template:

```text
.claude/skills/validate-prognostic-wrapper/sanity-check-template.py.txt
```

Read the template, adapt it for the specific model:

- Fill in the reference model loading code (PART 1)
- Replace `<DataSource>` with the user's chosen data source
- Update `ModelName` to the actual class name
- Adjust variable indexing for the reference model output

The script runs inference with **both** the original third-party model
and the E2S wrapper, then visualizes outputs side-by-side. Do **NOT**
commit this script to the repo.

### 2d. Run comparison and sanity-check scripts

Execute all scripts:

```bash
uv run python reference_comparison_<model_name>.py
uv run python sanity_check_<model_name>.py
```

Verify that:

- The reference comparison passes for both single-step and multi-step
- The sanity-check script runs without errors
- Output PNGs are generated
- `create_iterator` yields correct initial condition at step 0
- `lead_time` increments correctly at each step

### 2e. **[CONFIRM — Sanity-Check & Comparison]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the scripts ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
the reference comparison metrics, then ask them to inspect:

> The reference comparison and sanity-check scripts ran successfully.
>
> **Single-step reference comparison:**
>
> - Max absolute difference: `<value>`
> - Max relative difference: `<value>`
> - Correlation: `<value>`
>
> **Multi-step reference comparison (N steps):**
>
> - Step 1: max_abs=`<value>`, corr=`<value>`
> - Step N: max_abs=`<value>`, corr=`<value>`
> - (error may grow with lead time — this is expected)
>
> **Sanity-check plot saved to:**
> `/absolute/path/to/sanity_check_<model_name>.png`
>
> **Please open this image and confirm it looks correct.** Check:
>
> 1. Data is visible on the axes at all time steps (not blank/empty)
> 2. Values are in physically reasonable ranges
> 3. Spatial patterns evolve smoothly over lead time
> 4. No sudden jumps, NaN explosions, or frozen fields
> 5. Lead time labels increment correctly
>
> Does the plot look correct and do the reference comparison metrics
> look acceptable?

**Do not proceed to Step 3 until the user explicitly confirms.** If
the user reports problems, debug and fix the issue, re-run the
scripts, and ask the user to inspect again.

---

## Step 3 — Branch, Commit & Open PR

### **[CONFIRM — Ready to Submit]**

Before proceeding, confirm with the user:

> All implementation and validation steps are complete:
>
> - Prognostic model class implemented with correct method ordering
> - Triple inheritance: `torch.nn.Module + AutoModelMixin + PrognosticMixin`
> - Coordinate system with proper `handshake_dim` indices
> - `__call__` (single step) and `create_iterator` (multi-step) implemented
> - `create_iterator` yields initial condition first, uses front/rear hooks
> - Model loading implemented (`load_default_package`, `load_model`)
> - Registered in `earth2studio/models/px/__init__.py`
> - Documentation added to `docs/modules/models.rst`
> - Reference URLs included in class docstrings
> - CHANGELOG.md updated
> - Format, lint, and license checks pass
> - Unit tests written and passing with >= 90% coverage
> - Dependencies in pyproject.toml confirmed
> - Reference comparison passes (single-step and multi-step)
> - Sanity-check plots generated and confirmed by user
>
> Ready to create a branch, commit, and prepare a PR?

### 3a. Create branch and commit

```bash
git checkout -b feat/prognostic-model-<name>
git add earth2studio/models/px/<filename>.py \
        earth2studio/models/px/__init__.py \
        test/models/px/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models.rst
git commit -m "feat: add <ClassName> prognostic model

Add <ClassName> prognostic model for <brief description>.
Includes unit tests and documentation."
```

Do **NOT** add the sanity-check scripts, comparison scripts, or
their output images.

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
git push -u <fork-remote> feat/prognostic-model-<name>
```

### 3c. Open Pull Request (fork -> NVIDIA/earth2studio)

> **Important:** PRs must be opened **from the fork** to the
> **upstream source repository** `NVIDIA/earth2studio`. The branch
> lives on the fork; the PR targets `main` on the upstream repo.

Use `gh pr create` with explicit `--repo` and `--head` flags:

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/prognostic-model-<name> \
  --title "feat: add <ClassName> prognostic model" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork.

Use the PR body template from this skill's directory:

```text
.claude/skills/validate-prognostic-wrapper/pr-body-template.md
```

Read the template, fill in the placeholders with actual values, and use
as the `--body` argument. Key fields to populate:

- Model details table (architecture, time step, variables, resolution, etc.)
- **Checkpoint license** — if unknown, mark as "Unknown" with a note
- Dependencies table with license info (or "No new dependencies needed")
- Reference comparison metrics from Step 2a

### 3d. Post sanity-check as PR comment

After the PR is created, post the sanity-check visualization as a
separate **PR comment** so it is immediately visible to reviewers.

**GitHub has no CLI/API for uploading images to PR comments.** Use this
workflow:

1. Read the comment template from this skill's directory:

   ```text
   .claude/skills/validate-prognostic-wrapper/pr-comment-template.md
   ```

2. Fill in placeholders and write to `/tmp/pr_comment_body.md`

3. Post the comment:

   ```bash
   gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
     -F "body=@/tmp/pr_comment_body.md" \
     --jq '.html_url'
   ```

4. Tell the user to drag the sanity-check image into the browser editor

**Important:** Always paste the **complete, runnable** sanity-check
script in the comment — reviewers should be able to reproduce the plot
by copying the script directly.

After posting, inform the user of the comment URL and the local path to
the image file for manual attachment.

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
| 1 | <model>.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | <model>.py | 305 | Style | Use f-string | Fix: convert |
| 3 | <model>.py | 45 | Suggestion | Add type alias | Skip: not needed |
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
3. Run the relevant tests:

   ```bash
   uv run python -m pytest test/models/px/test_<filename>.py -v --timeout=60
   ```

4. Commit with a message like:

   ```bash
   git commit -m "fix: address code review feedback (Greptile)"
   ```

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
git push <fork-remote> feat/prognostic-model-<name>
```

After pushing, resolve all addressed review threads if possible.

Inform the user of the final state:

- How many comments were fixed
- How many were dismissed (with reasons)
- Any remaining open threads

---

## Reminders

- **DO** use the repo's local `uv` `.venv` to run Python with
  `uv run python`
- **DO NOT** commit sanity-check/comparison scripts or images to
  the repo
- **DO** use `loguru.logger` for logging, never `print()`, inside
  `earth2studio/`
- **DO** ensure all public functions have full type hints (mypy-clean)
- **DO** maintain alphabetical order in `__init__.py` exports,
  RST file entries, and CHANGELOG entries
- **DO** follow the canonical PX method ordering:
  `__init__`, `input_coords`, `output_coords`,
  `load_default_package`, `load_model`, `to`, private methods,
  `__call__`, `_default_generator`, `create_iterator`
- **DO** use `handshake_dim` indices matching each dimension's
  position in the `CoordSystem` OrderedDict — check existing px
  models for the predominant convention
- **DO** include reference URLs in class docstrings
- **DO** always update CHANGELOG.md under the current unreleased
  version
- **DO** add the model to `docs/modules/models.rst` in the
  `earth2studio.models.px` section (alphabetical order)
- **DO** inherit from `torch.nn.Module + AutoModelMixin + PrognosticMixin`
  (triple inheritance)
- **DO** yield initial condition first in `create_iterator`
  (step 0 = unchanged input)
- **DO** use `self.front_hook()` and `self.rear_hook()` in
  `create_iterator` for perturbation injection / post-processing
- **DO** include `lead_time` in `input_coords` starting at
  `np.timedelta64(0, "h")` and increment it in `output_coords`
- **DO NOT** make a general base class with intent to reuse the
  wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose
  essential parameters
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git
