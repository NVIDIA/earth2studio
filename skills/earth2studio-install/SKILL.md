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
  Guide installing Earth2Studio with uv or pip, selecting versions and model extras, and
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
use case: select a version, add optional extras with uv, set environment
variables, and verify the install. Do not write inference code or compose
workflows.

## Core principle: read the source, do not recall it

Install commands, release refs, extras, and build quirks change every release.
Nothing in this skill is authoritative. Before recommending any command, read
the sources below and quote from them.

- `docs/userguide/about/install_options.yml`: command templates, release ref,
  extras, class mappings (`api_refs`), pre-install steps, and warnings.
- `https://github.com/NVIDIA/earth2studio/releases/latest` or
  `gh api repos/NVIDIA/earth2studio/releases/latest`: latest stable release and exact tag.
- `docs/userguide/about/install.md`: uv project setup, Docker, prerequisites,
  verification, and environment variables.
- `pyproject.toml`: `requires-python` and `[project.optional-dependencies]`.
- `docs/userguide/support/troubleshooting.md` and `docs/userguide/support/faq.md`:
  known failures.

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

1. **Read the sources above.** Check `package.release_ref` against the latest
   stable GitHub release. Use the verified release tag, not a prerelease or the
   version of a development checkout. If the lookup is unavailable, label the
   configured ref as the documented release, not a verified latest release.
2. **Ask which version they want**, unless they already specified one. Present
   these choices in order, replacing `<version>` with the actual release number:

   > Which Earth2Studio version would you like to install?
   >
   > 1. **Latest release: `<version>` (recommended).** Best default for most users.
   > 2. **Main branch (second choice).** For unreleased features; may be unstable
   >    and include breaking changes.
   > 3. **Older or different version.** Which release or tag do you need?

   Wait for their choice before giving version-specific install commands. If
   they ask you to choose, use the latest stable release. For a specific tag,
   verify it exists. Read the selected tag's install guide and `pyproject.toml`
   (and `install_options.yml` when available) so extras and Python support match
   that version. For main, read those sources from main. If local sources target
   a different version, fetch the matching sources rather than mixing releases.
3. **Use uv to add Earth2Studio to their Python project.** Ask only what is
   missing (at most 3 questions total): version choice, existing project, and
   models or data sources needed. For a new project, follow the guide's
   "uv Project" setup with **Python 3.13** by default:

   ```bash
   mkdir earth2studio-project && cd earth2studio-project
   uv init --python=3.13
   ```

   Then provide `uv add` with the selected version and extras. In an existing
   project, run it from the project root; do not reinitialize the project or
   change its Python pin automatically. Check compatibility with the selected
   release, especially for older versions. Use a different Python version only
   when required by that release or the user's project. **pip is also supported**;
   use it when requested or when the user already has a pip-based workflow.
   Use Docker when requested or required by their existing workflow.
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
   found above. Follow the guide's default uv source: `github` for a release tag,
   `github-main` for main. Substitute the verified or requested tag into
   `{release_ref}`, or `package.main_ref` into `{main_ref}`. Use `uv add`, not
   `uv pip install`, for uv projects; combine extras as `earth2studio[extra1,extra2]`.
   Include the main instability warning even when the user chose main up front.
   For pip, use `command_templates.<source>.pip` in `install_options.yml`,
   including the matching pip-specific `preinstall` steps. Follow
   `defaults.sources.pip` (currently `pypi`) for releases; use `github-main` for
   main or `github` for an explicitly requested Git tag install. For PyPI,
   pin the selected release with `==<version>`.
   Surface any `preinstall` steps (keyed by package manager, and for uv by
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
