<!-- markdownlint-disable MD025 -->

(developer_agents)=

# AI Agent Skills

Earth2Studio ships a set of **agent skills** that guide AI coding assistants through
common development workflows step by step.
Each skill is a structured Markdown document that lives in `.claude/skills/` and can be
invoked by compatible agents (Claude Code, OpenCode, Cursor, etc.) to scaffold new
components, run validation, and open pull requests.

:::{note}
Agent skills are not a substitute for understanding the codebase.
They encode the project's conventions and review expectations so that an AI assistant
can follow them reliably, but a human developer should still review the output.
:::

## Available Skills

The skills are organised in **create / validate** pairs.
The *create* skill walks through implementation from a reference script to a working
wrapper; the *validate* skill picks up where *create* leaves off and handles testing,
comparison, and PR submission.

| Skill | Purpose |
| ----- | ------- |
| `create-data-source` | Wrap a remote data store as a data source |
| `create-prognostic-wrapper` | Wrap a prognostic (time-stepping) model as a PrognosticModel |
| `validate-prognostic-wrapper` | Test, compare, and submit a new prognostic wrapper |
| `create-diagnostic-wrapper` | Wrap a diagnostic (single-step) model as a DiagnosticModel |
| `validate-diagnostic-wrapper` | Test, compare, and submit a new diagnostic wrapper |
| `create-assimilation-wrapper` | Wrap a data assimilation model as an AssimilationModel |
| `validate-assimilation-wrapper` | Test, compare, and submit a new assimilation wrapper |

## How Skills Work

Each skill document is a numbered step-by-step guide with **confirmation gates**.
At every gate the agent pauses and asks the developer to review what has been produced
before continuing.

A typical *create* skill follows this flow:

1. **Obtain a reference** -- the developer provides a URL or path to existing inference
   code.
2. **Analyse dependencies** -- the agent reads the reference, proposes a
   `pyproject.toml` extras group, and waits for confirmation.
3. **Create a skeleton** -- a class file is generated with the correct inheritance,
   method ordering, and pseudocode bodies.
4. **Implement coordinates, forward pass, model loading** -- each section has its own
   confirmation gate.
5. **Register, format, lint** -- the wrapper is added to `__init__.py` and checked with
   `make format && make lint`.

A typical *validate* skill continues with:

1. **Run tests** -- unit tests are executed and coverage must reach 90 %.
2. **Reference comparison** -- the agent produces a side-by-side script comparing the
   new wrapper against the original inference code and generates sanity-check plots.
3. **Open a PR** -- the agent creates a branch, commits, and opens a pull request to
   `NVIDIA/earth2studio` with a standardised template.
4. **Greptile review** -- the agent polls for the automated code review, triages
   feedback, and applies accepted fixes.

## Using a Skill

### Claude Code / OpenCode

Invoke a skill directly by name or let the agent auto-detect it from context:

```text
> /create-prognostic-wrapper https://github.com/example/model/infer.py
```

The agent will load `.claude/skills/create-prognostic-wrapper/SKILL.md` and begin
Step 0.
You can also describe what you want in plain language and the agent will select the
appropriate skill:

```text
> I want to add the FourCastNet model as a new prognostic wrapper.
```

### Cursor

The skills are referenced by the corresponding `.cursor/rules/` rule files.
When working inside a file that matches a rule's glob pattern (e.g.,
`earth2studio/models/px/*.py`), Cursor will surface the relevant conventions
automatically.

## Relationship to Cursor Rules

The `.cursor/rules/` directory contains concise rule files that enforce coding
conventions (method ordering, decorators, coordinate validation, etc.).
Agent skills **build on** these rules: they reference the same conventions but add the
full workflow -- dependency management, skeleton creation, testing, PR submission, and
automated review handling.

| Resource | Location | Scope |
| -------- | -------- | ----- |
| Cursor rules | `.cursor/rules/e2s-*.mdc` | Conventions and style enforcement |
| Agent skills | `.claude/skills/*/SKILL.md` | End-to-end development workflows |
| Slash commands | `.claude/commands/*.md` | One-shot tasks (format, lint, test, docs) |
| CLAUDE.md | `CLAUDE.md` | Quick-reference entry point for agents |

## Writing a New Skill

If you need to create a skill for a new component type:

1. Create a directory under `.claude/skills/<skill-name>/`.
2. Add a `SKILL.md` file with YAML frontmatter (`name`, `description`,
   `argument-hint`).
3. Structure the document as numbered steps with `[CONFIRM -- <Title>]` gates at
   decision points.
4. Reference the matching `.cursor/rules/` file for coding conventions.
5. Include a **Reminders** section at the end with DO / DO NOT rules.

Look at an existing skill (e.g., `create-prognostic-wrapper`) as a template.
