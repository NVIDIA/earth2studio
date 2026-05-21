---
name: create-prognostic-wrapper
description: Create a new Earth2Studio prognostic model (px) wrapper from a reference inference script or repository
argument-hint: URL or local path to reference inference script/repo (optional — will be asked if not provided)
---

# Create Prognostic Model Wrapper

> **Note:** Creating a complete prognostic model wrapper involves three phases:
>
> 1. **`create-prognostic-wrapper`** (this skill) — Steps 0-8: Implement wrapper
> 2. **`create-prognostic-tests`** — Step 9: Write tests and comparison scripts
> 3. **`validate-prognostic-wrapper`** — Final validation and PR
>
> Complete all steps in this skill first, then invoke the next skill.

Each `### **[CONFIRM — <Title>]**` gate requires explicit user approval.

---

## Step 0 — Obtain Reference Script / Repository

If `$ARGUMENTS` is provided, use it (URL → WebFetch, local path → Read).

If not provided, ask:

> Please provide a reference inference script or repository URL/path that
> demonstrates how this model runs inference.

---

## Step 1 — Examine Reference & Propose Dependencies

### 1a. Analyze the reference code

Identify: Python packages, model architecture (PyTorch/ONNX/JAX), input/output
shapes, time step, spatial resolution, checkpoint format.

### 1b. Propose pyproject.toml dependency group

```toml
model-name = ["package1>=version", "package2"]
```

Group name: lowercase-hyphenated. Also add to `all` aggregate (px models line).

### **[CONFIRM — Dependencies]**

Present: 1) Group name, 2) Package list, 3) Ask if correct.

---

## Step 2 — Add Dependencies to pyproject.toml

Edit `pyproject.toml`: add group alphabetically, add to `all` aggregate.

---

## Step 3 — Create Skeleton Class File

### 3a. Determine class name and file name

- **Class**: PascalCase (e.g., `Pangu24`, `Aurora`)
- **File**: `earth2studio/models/px/<lowercase>.py`

### 3b. Write skeleton with pseudocode

Read `references/skeleton_template.py` for the complete template structure.

Every `.py` file must start with the license header from `test/_license/header.txt`.

**Method ordering** (canonical):

1. `__init__` — constructor
2. `input_coords` — input coordinate system
3. `output_coords` — (decorated `@batch_coords()`)
4. `load_default_package` — classmethod
5. `load_model` — classmethod
6. `to` — device management (optional, only for non-PyTorch state)
7. Private/support methods
8. `__call__` — single-step forward (`@batch_func()`)
9. `_default_generator` — (`@batch_func()`)
10. `create_iterator` — public entry point

### **[CONFIRM — Skeleton]**

Present: 1) Class name, 2) File name/path, 3) Ask if acceptable.

---

## Step 4 — Implement Coordinate System

### 4a. Map variables to E2STUDIO_VOCAB

Read `earth2studio/lexicon/base.py`. Verify all model variables exist (282 entries).

Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.

Flag missing variables for discussion.

### 4b. Implement input_coords

See `references/method_templates.py` for template.

**Rules:**
- `batch` first with `np.empty(0)`
- `time` is `np.empty(0)` (dynamic)
- `lead_time` starts at `np.timedelta64(0, "h")` (or negative for history)
- `lat`: 90 to -90 (north to south)
- `lon`: 0 to 360

### 4c. Implement output_coords

See `references/method_templates.py` for template with handshake validation.

### **[CONFIRM — Coordinates]**

Present: 1) Variable list + mapping issues, 2) Spatial dimensions, 3) History
needs, 4) Time step.

---

## Step 5 — Implement Forward Pass

### 5a. Implement `__call__`

See `references/method_templates.py` for template.

Key notes:
- `@batch_func()` handles batch dimension
- Input shape: `(batch, time, lead_time, variable, lat, lon)`
- Reshape to model format → call model → reshape back

### 5b. Implement create_iterator

See `references/method_templates.py` for template.

- MUST yield initial condition first (step 0)
- Use `front_hook`/`rear_hook` for perturbation injection

### **[CONFIRM — Forward Pass]**

Ask: 1) Is reshape logic correct? 2) Special considerations?

---

## Step 6 — Implement Model Loading

### 6a. Implement load_default_package

See `references/method_templates.py`. Lock HuggingFace URLs to commit:
`hf://org/repo@commit`.

### 6b. Implement load_model

See `references/method_templates.py`.

**Key patterns:**
- Use `package.resolve("filename")`
- Load with `map_location="cpu"`
- Set `eval()` mode
- Use `@check_optional_dependencies()`

### 6c. Implement .to()

Only override if non-PyTorch state exists (ONNX, JAX). Otherwise omit.

### **[CONFIRM — Model Loading]**

Present: 1) Checkpoint URL, 2) File names/loading logic, 3) Multiple files?,
4) `.to()` implementation.

---

## Step 7 — Register the Model

### 7a. Add to `__init__.py`

Edit `earth2studio/models/px/__init__.py` — add import alphabetically.

### 7b. Verify pyproject.toml

Confirm dependency group exists and is in `all` aggregate.

---

## Step 8 — Verify Style, Documentation, Format & Lint

### 8a. Run formatting

```bash
make format
```

### 8b. Run linting

```bash
make lint
```

Watch for: missing type hints, unused imports, import ordering, type errors.

### 8c. Check license headers

```bash
make license
```

### 8d. Verify documentation

Check: NumPy-style docstrings, complete `Parameters`/`Returns`/`Raises`
sections, type hints on all public methods.

### 8e. Add model to documentation

Edit `docs/modules/models_px.rst` — add class alphabetically to autosummary.

### 8f. Update CHANGELOG.md

Add to `### Added` section:

```markdown
- Added <ModelName> prognostic model (`<ClassName>`) with <brief description>
```

If new dependencies, add to `### Dependencies`:

```markdown
- Added `<model-name>` optional dependency group for <ModelName> model
```

### 8g. Update install.md with model dependencies

Add to `docs/userguide/about/install.md` Prognostics section (alphabetically).

See `references/install_templates.md` for format and examples.

---

## Step 9 — Test the Wrapper

Invoke **`create-prognostic-tests`** skill:

```text
/skill create-prognostic-tests <ModelName>
```

---

## Reminders

- **DO NOT** make a general base class for reuse across models
- **DO NOT** over-populate `load_model()` API
- **DO** add model to `docs/modules/models_px.rst` (Step 8e)
- **DO** update `CHANGELOG.md` (Step 8f)
- **DO** use `loguru.logger`, never `print()`
- **DO** ensure full type hints on public functions
- **DO** run `make format` and `make lint` before finalizing
