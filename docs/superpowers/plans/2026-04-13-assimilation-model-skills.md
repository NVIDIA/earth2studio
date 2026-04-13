# Assimilation Model Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two Claude skills (`create-assimilation-wrapper`
and `validate-assimilation-wrapper`) that guide agents through
building and validating Earth2Studio data assimilation model
wrappers.

**Architecture:** Each skill is a single Markdown file in
`.claude/skills/<name>/SKILL.md` with YAML frontmatter, numbered
steps with `[CONFIRM]` gates, complete code blocks, and a
Reminders section. The create skill covers Steps 0-8
(implementation only), the validate skill covers Steps 1-5
(testing, comparison, PR, Greptile).

**Tech Stack:** Markdown skill files, Python code blocks
referencing Earth2Studio DA patterns (pandas/cudf DataFrames,
xarray DataArrays, FrameSchema, CoordSystem, send-protocol
generators).

---

## Task 1: Create `create-assimilation-wrapper` SKILL.md

**Files:**

- Create: `.claude/skills/create-assimilation-wrapper/SKILL.md`

This task creates the full skill file for guiding creation of DA
model wrappers. The file should be modeled on the existing
`create-diagnostic-wrapper/SKILL.md` structure but with
DA-specific content throughout.

**Key reference files to read before writing:**

- `.cursor/rules/e2s-013-assimilation-models.mdc` (577 lines)
  — authoritative DA implementation rules
- `earth2studio/models/da/base.py` (188 lines)
  — AssimilationModel Protocol
- `earth2studio/models/da/interp.py` (593 lines)
  — simple stateless DA example
- `earth2studio/models/da/utils.py`
  — DA utilities (validate, filter, convert)
- `.claude/skills/create-diagnostic-wrapper/SKILL.md`
  — structural template (layout, step structure, CONFIRM gates)
- `docs/superpowers/specs/2026-04-13-assimilation-model-skills-design.md`
  — approved spec

**Structure to follow:**

The file must contain:

1. **YAML Frontmatter** (lines 1-5):

   ```yaml
   ---
   name: create-assimilation-wrapper
   description: >-
     Create a new Earth2Studio data assimilation model (da)
     wrapper from a reference inference script or repository.
     DA models ingest sparse observations (DataFrames) and/or
     gridded state arrays (DataArrays) and produce analysis
     output — they do NOT use @batch_func, @batch_coords,
     or PrognosticMixin.
   argument-hint: >-
     URL or local path to reference inference script/repo
     (optional — will be asked if not provided)
   ---
   ```

2. **Title and preamble** — "Create Assimilation Model Wrapper",
   CONFIRM gate explanation, uv/venv environment note.

3. **Critical Differences from px/dx table** — The full table
   from the spec (18 rows). This MUST be placed prominently
   near the top, right after the preamble. It's the most
   important reference for any agent using this skill.

4. **Step 0 — Obtain Reference & Analyze Model**
   - 0a: Accept `$ARGUMENTS` or ask user (same as dx skill)
   - 0b: Analyze the reference to determine:
     - Input types: DataFrame only, DataArray only, or mixed
     - Output types: DataArray (gridded), DataFrame (tabular)
     - Stateful vs stateless (state between steps?)
     - Whether `@torch.inference_mode()` is safe
     - Dependency packages
   - No NN/physics branching — all DA models get the same flow
   - Present analysis summary to user
   - `[CONFIRM — Model Analysis]`

5. **Step 1 — Examine Reference & Propose Dependencies**
   - 1a: Identify packages (physicsnemo, scipy, healpy, etc.)
   - 1b: Propose pyproject.toml group `da-<model-name>`
   - Highlight cudf/cupy as optional GPU acceleration
   - `[CONFIRM — Dependencies]`

6. **Step 2 — Add Dependencies to pyproject.toml**

7. **Step 3 — Create Skeleton Class File**
   - File path: `earth2studio/models/da/<filename>.py`
   - License header (SPDX Apache-2.0, 2024-2026)
   - Dual inheritance: `torch.nn.Module + AutoModelMixin`
     (NO PrognosticMixin)
   - Class-level `@check_optional_dependencies()` decorator
   - Optional dependency try/except with
     `OptionalDependencyFailure("da-<name>")`
   - cupy/cudf optional imports

   **Canonical DA method ordering** (MUST be enforced):
   1. `__init__` — register `device_buffer`, store model,
      normalize time tolerance
   2. `device` property — `return self.device_buffer.device`
   3. `init_coords` — `None` for stateless,
      `tuple[CoordSystem]` for stateful
   4. `input_coords` — `tuple[FrameSchema]` for DataFrame,
      `tuple[CoordSystem]` for DataArray
   5. `output_coords` — accept `input_coords` tuple +
      `request_time` kwarg, return tuple
   6. `load_default_package` — classmethod
   7. `load_model` — classmethod with
      `@check_optional_dependencies()`
   8. `to` — device management, return `AssimilationModel`
   9. Private/support methods
   10. `__call__` — stateless forward
   11. `create_generator` — bidirectional generator with send

   **Complete skeleton code block** — must include all methods
   with pseudocode `# TODO` comments. Use the complete example
   from `e2s-013-assimilation-models.mdc` lines 420-544 as
   the primary template.

   **Key differences from dx/px skeletons** to document:

   - No `@batch_func()`, no `@batch_coords()`,
     no `PrognosticMixin`
   - `input_coords` and `output_coords` return **tuples**
     (even for single)
   - `FrameSchema` for tabular inputs,
     `CoordSystem` for gridded outputs
   - `device` property instead of
     `next(self.parameters()).device`
   - `create_generator` with send protocol instead of
     `create_iterator`
   - `validate_observation_fields()` instead of
     `handshake_coords`/`handshake_dim` for DataFrames
   - `request_time` from `obs.attrs`, not a coordinate dim

   - `[CONFIRM — Skeleton]`

8. **Step 4 — Implement Coordinate System**
   - 4a: `init_coords()` — return `None` for stateless,
     or tuple of `CoordSystem`/`FrameSchema` for stateful.
     Show both patterns.
   - 4b: `input_coords()` — return tuple of `FrameSchema`.
     Show the standard 5-column schema: time, lat, lon,
     observation, variable. Explain `np.empty(0, dtype=...)`
     for unbounded vs `np.array([...])` for enumerated.
   - 4c: `output_coords()` — accept tuple + `request_time`,
     return tuple of `CoordSystem`. Show handshake helpers
     only for `CoordSystem` inputs. Show
     `validate_observation_fields()` for `FrameSchema`.
   - `[CONFIRM — Coordinates]`

9. **Step 5 — Implement Forward Pass**
   - 5a: `__call__` — extract `request_time` from
     `obs.attrs`, validate with
     `validate_observation_fields()`, filter with
     `filter_time_range()`, convert with
     `dfseries_to_torch()`, run model, build
     `xr.DataArray` with cupy/numpy based on device
   - 5b: `create_generator` — two patterns:
     - Stateless: `observations = yield None`, loop calling
       `self.__call__(observations)`, handle `GeneratorExit`
     - Stateful: accept init args,
       `obs = yield initial_state`, loop with state updates,
       handle `GeneratorExit`
   - Code blocks for both patterns
   - `[CONFIRM — Forward Pass]`

10. **Step 6 — Implement Model Loading**
    - `load_default_package` — Package with cache options
    - `load_model` — `@check_optional_dependencies()`,
      `package.resolve()`, `.eval()`, `.requires_grad_(False)`
    - `.to()` — `super().to(device)`, return
      `AssimilationModel`
    - `[CONFIRM — Model Loading]`

11. **Step 7 — Register in `__init__.py`**
    - Add to `earth2studio/models/da/__init__.py` (alpha order)
    - Add to `__all__` if present

12. **Step 8 — Verify Style/Format/Lint**
    - `make format`, `make lint`, `make license`
    - Common mypy issues for DA: DataFrame type annotations,
      optional cudf types

13. **Reminders section** — Complete DA-specific DO/DON'T rules:
    - DO return tuples from `input_coords`/`output_coords`
    - DO use `FrameSchema` for tabular, `CoordSystem` for grid
    - DO validate `request_time` from `obs.attrs`
    - DO use `validate_observation_fields()`,
      `filter_time_range()`, `dfseries_to_torch()`
    - DO prime generator with `yield None` and handle
      `GeneratorExit`
    - DO return cupy arrays on GPU, numpy on CPU
    - DO register `device_buffer` and expose `device` property
    - DO use `loguru.logger`, never `print()`
    - DO ensure all public functions have full type hints
    - DO run `make format && make lint` before finalizing
    - DO use `uv run python` for all Python commands
    - DO NOT use `@batch_func()` or `@batch_coords()`
    - DO NOT use `PrognosticMixin`
    - DO NOT use `create_iterator` — DA uses
      `create_generator` with send protocol
    - DO NOT assume tensor inputs — inputs are DFs/DataArrays
    - DO NOT forget cudf/cupy optional import pattern
    - DO NOT make a general base class with intent to reuse
    - DO NOT over-populate `load_model()` API

- [ ] **Step 1: Read all reference files**

Read:

- `.cursor/rules/e2s-013-assimilation-models.mdc` (577 lines)
- `earth2studio/models/da/base.py` (188 lines)
- `earth2studio/models/da/interp.py` (593 lines — primary ref)
- `earth2studio/models/da/utils.py` (~175 lines)
- `.claude/skills/create-diagnostic-wrapper/SKILL.md`
  (structural template)
- `docs/superpowers/specs/2026-04-13-assimilation-model-skills-design.md`
  (297 lines)

- [ ] **Step 2: Create directory**

```bash
mkdir -p .claude/skills/create-assimilation-wrapper
```

- [ ] **Step 3: Write SKILL.md**

Write `.claude/skills/create-assimilation-wrapper/SKILL.md`
with all content described above. The file should be
approximately 1200-1500 lines (similar to the dx create skill
at 1440 lines, but slightly shorter since there is no
NN/physics branching).

**Self-review checklist before reporting:**

- [ ] Frontmatter has correct name, description, argument-hint
- [ ] Critical differences table is present near the top
- [ ] All 9 steps (0-8) are present with correct headers
- [ ] 6 CONFIRM gates: Model Analysis, Dependencies,
  Skeleton, Coordinates, Forward Pass, Model Loading
- [ ] Complete skeleton code block with all 11 method slots
- [ ] FrameSchema and CoordSystem patterns shown correctly
- [ ] `create_generator` with send protocol (both patterns)
- [ ] `validate_observation_fields`, `filter_time_range`,
  `dfseries_to_torch` all shown
- [ ] cupy/cudf optional import and output patterns
- [ ] `device` property pattern (not
  `next(self.parameters()).device`)
- [ ] No stale px/dx references (no `@batch_func`,
  no `@batch_coords`, no `PrognosticMixin`,
  no `create_iterator`, no `handshake_dim` for DF inputs)
- [ ] License header template (SPDX Apache-2.0, 2024-2026)
- [ ] Reminders section with all DO/DON'T rules from spec
- [ ] No placeholder TODOs in prose (skeleton code blocks may
  have `# TODO` pseudocode)

- [ ] **Step 4: Self-review and report**

Report status: DONE / DONE_WITH_CONCERNS / BLOCKED /
NEEDS_CONTEXT

---

## Task 2: Create `validate-assimilation-wrapper` SKILL.md

**Files:**

- Create: `.claude/skills/validate-assimilation-wrapper/SKILL.md`

This task creates the full validation skill file. Model on the
existing `validate-diagnostic-wrapper/SKILL.md` structure but
with DA-specific content.

**Key reference files to read before writing:**

- `.claude/skills/validate-diagnostic-wrapper/SKILL.md`
  (831 lines — structural template)
- `.cursor/rules/e2s-013-assimilation-models.mdc` (577 lines)
- `test/models/da/test_da_interp.py` (359 lines — test patterns)
- `earth2studio/models/da/interp.py` (593 lines — reference)
- `docs/superpowers/specs/2026-04-13-assimilation-model-skills-design.md`
  (297 lines)

**Structure to follow:**

1. **YAML Frontmatter**:

   ```yaml
   ---
   name: validate-assimilation-wrapper
   description: >-
     Validate a newly created Earth2Studio data assimilation
     model wrapper by writing unit tests (90% coverage
     required), performing reference comparison with
     DataFrame/DataArray outputs, generating sanity-check
     plots, and opening a PR with automated code review.
     Use after completing create-assimilation-wrapper
     Steps 0-8.
   argument-hint: >-
     Name of the DA model class and test file (optional —
     will be inferred from recent changes if not provided)
   ---
   ```

2. **Title and preamble** — uv/venv note, CONFIRM gate
   explanation.

3. **Step 1 — Write Pytest Unit Tests**

   DA-specific test patterns (NOT the same as dx/px tests):

   - `PhooModelName` dummy class — a simple
     `torch.nn.Module` that returns known output shapes as
     `xr.DataArray`. It must accept `pd.DataFrame` input
     and return `xr.DataArray` output (NOT tensor-to-tensor
     like dx/px dummies).
   - `test_package` fixture — create dummy checkpoint,
     save to tmp_path
   - **Parametrize over pandas AND cudf** —
     `@pytest.fixture` for pandas DataFrame, separate
     fixture for cudf with `pytest.skip` guard
   - **Parametrize over CPU AND GPU** devices
   - `test_model_call` — create DataFrame with obs.attrs,
     call model, verify:
     - Output is `xr.DataArray` (not torch.Tensor)
     - Output dims match output_coords
     - Output data type matches device (cupy/numpy)
   - `test_generator_protocol` — prime, send, close:

     ```python
     gen = model.create_generator()
     result = gen.send(None)  # Prime
     assert result is None  # or initial state
     da = gen.send(obs_df)   # Send observations
     assert isinstance(da, xr.DataArray)
     gen.close()
     ```

   - `test_init_coords` — verify returns None (stateless)
     or tuple (stateful)
   - `test_input_coords` — verify returns tuple of
     FrameSchema
   - `test_time_tolerance` — verify filter_time_range works
   - `test_empty_dataframe` — verify graceful handling
   - `test_invalid_attrs` — verify raises on missing
     `request_time`
   - `test_validate_observation_fields` — verify raises on
     bad columns
   - `test_model_exceptions` — invalid inputs raise
     ValueError/KeyError
   - `@pytest.mark.package` integration test

   Show complete test file template with all test methods.

   - `[CONFIRM — Package Test]` (for integration test only)

4. **Step 2 — Run Tests & Achieve 90% Coverage**
   - 2a: Run test file with `-v --timeout=60`
   - 2b: Coverage with `--slow`
     `--cov=earth2studio/models/da/<filename>`
     `--cov-report=term-missing` `--cov-fail-under=90`
   - DA-specific coverage gaps: GeneratorExit cleanup path,
     cudf code paths, empty DataFrame handling, time
     tolerance edge cases, obs.attrs validation branches,
     cupy vs numpy output paths
   - 2c: Optional full suite `make pytest TOX_ENV=test-da`

5. **Step 3 — Reference Comparison & Sanity-Check**
   - 3a: Reference comparison script — compare `__call__`
     output AND multi-step generator output against
     reference implementation
     - For DataArray: max_abs_diff, max_rel_diff,
       correlation, torch.allclose
     - For DataFrame: row counts, value ranges, spatial
       coverage
   - 3b: Model summary table — input schema, output grid,
     variables, stateful/stateless, observation types,
     cudf support
   - 3c: Three sanity-check plot templates:
     1. **Spatial assimilated output** — `contourf` of
        gridded DataArray output
     2. **Observation overlay** — scatter of input DataFrame
        observations on assimilated grid (unique to DA)
     3. **Generator sequence** — multi-step evolution
   - 3d: Side-by-side comparison scripts (ref vs E2S)
   - 3e-3f: Run scripts, user confirms plots
   - `[CONFIRM — Sanity-Check & Comparison]`

6. **Step 4 — Branch, Commit & Open PR**
   - `[CONFIRM — Ready to Submit]` with DA checklist
   - 4a: Create branch `feat/da-model-<name>`, commit
   - 4b: Identify fork remote, push
   - 4c: Open PR with DA-specific template:
     - Model type: Stateless / Stateful
     - Input format: DataFrame / DataArray / Mixed
     - Output format: DataArray / DataFrame
     - Observation schema: columns and types
     - Grid specification
     - Time tolerance
     - cudf/cupy support
     - Reference comparison metrics
   - 4d: Post sanity-check as PR comment

7. **Step 5 — Automated Code Review (Greptile)**
   - Same polling/triage/fix pattern as dx/px validate skills
   - 5a: Poll for `greptile-apps[bot]` review
   - 5b: Fetch + parse comments
   - 5c: Categorize and present triage table
   - `[CONFIRM — Review Triage]`
   - 5d: Implement fixes
   - 5e: Respond to comments
   - 5f: Push and resolve

8. **Reminders section** — Same DA-specific DO/DON'T as
   create skill, plus:
   - DO NOT commit sanity-check scripts/images
   - DO NOT commit secrets/credentials
   - DO maintain alphabetical order in `__init__.py` exports
   - NEVER call `loop.set_default_executor()`

- [ ] **Step 1: Read all reference files**

Read:

- `.claude/skills/validate-diagnostic-wrapper/SKILL.md`
  (831 lines)
- `.cursor/rules/e2s-013-assimilation-models.mdc` (577 lines)
- `test/models/da/test_da_interp.py` (359 lines)
- `docs/superpowers/specs/2026-04-13-assimilation-model-skills-design.md`
  (297 lines)

- [ ] **Step 2: Create directory**

```bash
mkdir -p .claude/skills/validate-assimilation-wrapper
```

- [ ] **Step 3: Write SKILL.md**

Write `.claude/skills/validate-assimilation-wrapper/SKILL.md`
with all content described above. Target approximately
900-1100 lines (similar to dx validate at 831 lines, but
slightly longer due to DA-specific test templates).

**Self-review checklist before reporting:**

- [ ] Frontmatter has correct name, description, argument-hint
- [ ] All 5 steps present with correct headers
- [ ] 4 CONFIRM gates: Package Test, Sanity-Check &
  Comparison, Ready to Submit, Review Triage
- [ ] Complete test file template with DA-specific patterns
  (DataFrame fixtures, cudf parametrize, generator protocol
  test, obs.attrs test)
- [ ] 90% coverage requirement with DA-specific gap list
- [ ] 3 sanity-check plot templates (spatial, observation
  overlay, generator sequence)
- [ ] Reference comparison script with DataArray AND
  DataFrame metrics
- [ ] DA-specific PR template with all required fields
- [ ] Greptile polling/triage workflow
- [ ] Reminders section with all DA-specific rules
- [ ] No stale dx/px references (no `@batch_func`, no tensor
  assertions, no `handshake_dim` for DataFrame assertions)
- [ ] No data-source leftovers

- [ ] **Step 4: Self-review and report**

Report status: DONE / DONE_WITH_CONCERNS / BLOCKED /
NEEDS_CONTEXT

---

## Task 3: Final Verification and Commit

**Files:**

- Verify: `.claude/skills/create-assimilation-wrapper/SKILL.md`
- Verify: `.claude/skills/validate-assimilation-wrapper/SKILL.md`

**Depends on:** Tasks 1 and 2

- [ ] **Step 1: Verify both files exist and are non-empty**

```bash
ls -la .claude/skills/create-assimilation-wrapper/SKILL.md
ls -la .claude/skills/validate-assimilation-wrapper/SKILL.md
wc -l .claude/skills/create-assimilation-wrapper/SKILL.md
wc -l .claude/skills/validate-assimilation-wrapper/SKILL.md
```

Expected: create skill ~1200-1500 lines, validate ~900-1100.

- [ ] **Step 2: Verify frontmatter is parseable**

```bash
head -6 .claude/skills/create-assimilation-wrapper/SKILL.md
head -6 .claude/skills/validate-assimilation-wrapper/SKILL.md
```

Both must start with `---` and have `name:`, `description:`,
`argument-hint:`.

- [ ] **Step 3: Check for stale placeholders**

```bash
grep -n "TODO" \
  .claude/skills/create-assimilation-wrapper/SKILL.md \
  | grep -v "# TODO" | head -20
grep -n "TODO" \
  .claude/skills/validate-assimilation-wrapper/SKILL.md \
  | grep -v "# TODO" | head -20
grep -n "TBD" \
  .claude/skills/create-assimilation-wrapper/SKILL.md
grep -n "TBD" \
  .claude/skills/validate-assimilation-wrapper/SKILL.md
```

No stale TODOs outside of skeleton code blocks. No TBDs.

- [ ] **Step 4: Verify no stale px/dx references**

```bash
grep -n \
  "@batch_func\|@batch_coords\|PrognosticMixin" \
  .claude/skills/create-assimilation-wrapper/SKILL.md \
  | head -20
grep -n \
  "@batch_func\|@batch_coords\|PrognosticMixin" \
  .claude/skills/validate-assimilation-wrapper/SKILL.md \
  | head -20
```

Any matches must be in "DO NOT" / negative context only.

- [ ] **Step 5: Cross-reference spec coverage**

Read spec file and verify every requirement has
corresponding content in the skill files:

- 6 CONFIRM gates in create skill
- 4 CONFIRM gates in validate skill
- Critical differences table in create skill
- All Reminders items present

- [ ] **Step 6: Commit**

```bash
git add \
  .claude/skills/create-assimilation-wrapper/SKILL.md \
  .claude/skills/validate-assimilation-wrapper/SKILL.md
git commit -m "feat: add create and validate assimilation wrapper skills

Add create-assimilation-wrapper (Steps 0-8) and
validate-assimilation-wrapper (Steps 1-5) skills for
guiding DA model wrapper implementation and validation.
DA-specific patterns: FrameSchema, send-protocol generators,
cudf/cupy support, validate_observation_fields."
```

- [ ] **Step 7: Verify commit**

```bash
git log --oneline -1
git status
```
