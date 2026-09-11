---
name: developer-open-refactor-pr
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags: [earth2studio, developer, pull-request, github]
description: >
  Open an Earth2Studio 1.0 refactor pull request with the required base, label,
  title prefix, and concise description. Use only when asked to open the PR.
---

# Open an Earth2Studio 1.0 Refactor PR

Use GitHub CLI to open a pull request against `NVIDIA/earth2studio`.

## Requirements

- Target branch: `1.0.0-rc`
- Label: `1.0.0`
- Title prefix: `[E2S 1.0]`
- Description: concise and derived from the committed diff

## Workflow

1. Verify `gh auth status`, the current branch, and a clean working tree.
2. Review commits and changes relative to `1.0.0-rc`.
3. Propose a short title with exactly one `[E2S 1.0]` prefix.
4. Show the complete title and ask the user to confirm it. Do not push or create
   the pull request before confirmation.
5. Confirm the `1.0.0` label exists. Do not create repository labels.
6. Push the current branch to `origin` when needed.
7. Write a brief body containing:
   - `## Summary` with one to three bullets
   - `## Testing` listing only checks that were actually run
8. Create the pull request with `gh pr create`, targeting `1.0.0-rc`, applying
   `1.0.0`, and using the confirmed title. For forks, derive the head owner from
   `origin` and pass `<owner>:<branch>` as the head.
9. Return the pull request URL. If one already exists for the branch, return its
   URL instead of creating a duplicate.

Stop and report the issue if authentication, the base branch, the label, or a
clean committed branch is unavailable.
