---
name: earth2studio-install
version: 0.18.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - install
    - deployment
    - environment
description: >
  Guide installing Earth2Studio via uv or pip, selecting model extras, and
  configuring the environment. Do NOT use for writing inference code, choosing
  models, or PhysicsNeMo questions.
---

# Earth2Studio Installation Skill

## Never install packages automatically

You **MUST NOT** install, upgrade, or modify packages on the user's
behalf. Provide the exact command; the user runs it. No exceptions.

**Forbidden:** running `pip install`, `uv pip install`, `uv add`,
`uv sync`, `uv init`, `uv venv`, `conda install`, `apt install`, or any package manager.

When a package is needed:

1. Identify it
2. Provide the exact command
3. Explain why it is needed
4. **Wait for the user to confirm they ran it**

Even if the user says "just install it", give the command and require
them to execute it themselves.

## Purpose

Help users install Earth2Studio and its optional model dependencies for their
use case: pick the package manager, select optional extras, set environment
variables, and verify the install. Do not write inference code or compose
workflows.

## Core principle: read the source, do not recall it

Install commands, release refs, extras, and build quirks change every release.
Nothing in this skill is authoritative. Before recommending any command, read
the sources below and quote from them.

| Need | Where to look (repo path first, live page as a fallback) |
|------|------------------------------------------------|
| Install commands, release ref, extras, which classes each extra enables (`api_refs`), per-extra pre-install steps and warnings | `docs/userguide/about/install_options.yml` (generates the live install page) |
| Recommended environment (the "uv Project" section), Docker, prerequisites, verification, environment variables | `docs/userguide/about/install.md` |
| Supported Python versions | `requires-python` in `pyproject.toml` |
| Full list of extras and what each pulls in | `[project.optional-dependencies]` in `pyproject.toml` |
| Known failures | `docs/userguide/support/troubleshooting.md` and `docs/userguide/support/faq.md` |

Live pages (note the `/main/` segment; URLs without it return 404):

```text
https://nvidia.github.io/earth2studio/main/userguide/about/install/
https://nvidia.github.io/earth2studio/main/userguide/support/troubleshooting/
https://nvidia.github.io/earth2studio/main/userguide/support/faq/
```

If working in a local clone, read the repo files directly. Otherwise fetch the
raw files from `https://github.com/NVIDIA/earth2studio` (the `main` branch, or the
release tag the user wants) or use the live pages. The install page renders its
commands with JavaScript, so prefer `install_options.yml` when the page looks
incomplete.

## Instructions

1. **Read the sources above.** Note the current release ref, supported Python
   versions, and the extras relevant to the user.
2. **Recommend a uv project by default.** The docs recommend a dedicated uv
   project (`uv init` in a fresh directory, then `uv add` for Earth2Studio and
   its extras) rather than installing into a system or shared Python. It keeps
   the environment isolated and reproducible, and the install page's commands
   are written as `uv add`. Give the project-setup commands from the "uv Project" section of
   `install.md`, using the Python version it recommends. Only fall back to
   pip, an existing environment, or Docker if the user asks for it or already
   has one; say what the tradeoff is.
3. **Ask only what is missing** (at most 3 questions): whether they already have
   a project or environment to add to, and which models or data sources they
   plan to use.
4. **Map the models and data sources they want to extras.** In
   `install_options.yml`, each entry under `categories[].items[]` has an `extra`
   (the name to install) and `api_refs` (the classes it enables, for example
   `earth2studio.models.px.Aurora1p5` belongs to the `aurora` extra). Search
   `api_refs` for the class name the user needs. Data sources generally share one
   `data` extra, and perturbation, statistics, and utilities have their own
   groups. If a class is not listed, check `[project.optional-dependencies]` in
   `pyproject.toml` and the class docstring before guessing. Several extras can
   go in one command.
5. **Provide the command(s)** built from the `command_templates` and the extras
   found above. Surface any `preinstall` steps (keyed by package manager, and for uv by
   source, so pick the one matching the command you chose), `warnings`, or `notes` attached to
   the chosen entries (slow builds such as flash-attn, `--no-build-isolation`,
   CUDA-specific packages, large checkpoint downloads, and so on). Do not
   paraphrase from memory.
6. **Wait for the user to run it**, then have them verify using the command in
   `install.md` (use the `uv run` form inside a uv project).
7. **Mention configuration only when relevant** (limited disk, shared
   filesystem, CI). The environment variables and their meaning are listed in
   `install.md`.
8. **On failure**, start from `troubleshooting.md` and `faq.md`. Check that
   PyTorch and CUDA are working before blaming Earth2Studio.

## Limitations

- Does not cover runtime errors unrelated to missing dependencies.
- Does not cover model checkpoint downloads (these happen at first use).
- Does not set up data sources beyond selecting the relevant extra.

## Ownership and out-of-scope

**Owns:** package installation, optional-extra selection, environment variable
guidance, install verification.

**Does not own:** writing inference or training code, composing Earth2Studio
workflows, choosing models (use `earth2studio-discover`), PhysicsNeMo questions.
