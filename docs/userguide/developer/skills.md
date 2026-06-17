<!-- markdownlint-disable MD025 -->

(developer_skills)=

# AI Agent Skills

Earth2Studio includes AI agent skills that provide specialized guidance for AI coding
assistants (Claude Code, Codex, etc.) when working with Earth2Studio. These skills help
agents build inference pipelines, select models, and generate correct code.

## Overview

Skills are located in the `skills/` directory at the repository root. Each skill
contains:

- `SKILL.md` - Instructions and workflow for the AI agent
- `evals/evals.json` - Evaluation tasks for testing skill effectiveness
- `evals/targets/` - Reference outputs for evaluation grading
- `references/` - Supporting documentation (optional)

:::{note}
Skill development is currently internal to NVIDIA. External contributions to skills
are not accepted at this time.
:::

## Skill Validation

Skills are validated using the harbor agent evaluation framework via `nv-base`. To run
skill evaluations locally:

```bash
nv-base agent-eval <path-to-skill> -a claude-code,codex -r cli,html,json -o ./eval-results/
```

For example, to evaluate the deterministic forecast skill:

```bash
nv-base agent-eval skills/earth2studio-deterministic-forecast \
    -a claude-code,codex \
    -r cli,html,json \
    -o ./eval-results/
```

This produces:

- `cli` - Console output with pass/fail summary
- `html` - Interactive HTML report for detailed analysis
- `json` - Machine-readable results for CI integration

## Evaluation Metrics

Skills are evaluated across five dimensions:

| Dimension | Description |
|-----------|-------------|
| Security | Avoids unsafe operations, secret leakage, unauthorized access |
| Correctness | Agent follows expected workflow and produces correct output |
| Discoverability | Agent loads skill when relevant, avoids when irrelevant |
| Effectiveness | Agent performs better with skill than without |
| Efficiency | Agent uses fewer tokens and avoids redundant work |
