# Changelog fragments

Feature PRs should add a fragment here instead of editing `CHANGELOG.md`.
Tests-only, docs-only, CI-only, and similar maintenance PRs do not need one.
CI validates existing fragments; it does not require an entry on every PR.

## Adding an entry

Use `<PR-number>.<category>.md`, for example `1420.added.md`. Write the
user-facing description in Markdown, without a heading or leading bullet.
Use separate fragments when a PR covers multiple categories.

Categories, in release-note order:

| Category | Content |
| --- | --- |
| `added` | New features |
| `changed` | Changes to existing behavior |
| `deprecated` | Features scheduled for removal |
| `removed` | Removed features |
| `fixed` | Bug fixes |
| `security` | Security fixes |
| `dependencies` | Dependency changes |

Run from the repository root:

```bash
uv run --only-dev towncrier create 1420.added.md --content "Added a new feature."
make changelog-check
```

Before a PR number exists, use a unique name such as
`+pipeline-reductions.added.md`. Rename it to the PR number when available;
`+` fragments are also supported at release time without a PR link.

## Releases

Preview accumulated notes for an RC without changing files or deleting fragments:

```bash
uv run --only-dev towncrier build --draft --version 1.1.0rc1
```

At the final release, assemble notes once with the stable version and release date:

```bash
uv run --only-dev towncrier build --yes --version 1.1.0 --date YYYY-MM-DD
```

Replace the example version and date with the release values. Review the generated
`CHANGELOG.md` and commit it together with the consumed fragment deletions.
Towncrier stages those changes; it does not commit them. Published history below
the insertion marker is preserved. Keep `README.md` in this directory.

Do not consume fragments for RCs or create blank development sections in the
changelog. Package version bumps are independent of changelog assembly.
