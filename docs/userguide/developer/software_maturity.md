# Software Maturity

Earth-2 consists of features and models that range from highly experimental to
production-focused. As a result, Earth-2
distributes code and products intended for different consumption and usage patterns,
from research experiments to integrations and long-term deployments.

The Earth-2 software maturity model describes a repository's **intended usage**,
**compatibility commitments**, and **maintenance expectations**. It helps users decide
whether software fits an experiment, an integration, or a longer-term deployment.
It does not rank scientific value, model accuracy, or code quality.

## Maturity at a glance

| Level | Intended usage | Compatibility and support expectations |
| --- | --- | --- |
| [L0 — Research](#l0-research) | Explore ideas and reproduce research. | Rapid changes; no stability or maintenance guarantee. |
| [L1 — Emerging](#l1-emerging) | Build initial integrations beyond the original authors. | Documented setup, tested public behavior, and contribution guidance; APIs may change. |
| [L2 — Hardened](#l2-hardened) | Adopt supported interfaces with production-quality reference guidance. | Defined compatibility boundaries, deprecation timelines, tested environments, and migration guidance. |
| [L3 — Strategic](#l3-strategic) | Build on a sustained, long-term ecosystem investment. | Compatibility commitments maintained across releases, predictable releases, and established maintenance processes. |

Levels are **cumulative**: a repository meets the expectations of its published level
and all preceding levels. A badge describes the current demonstrated state, rather
than a future target. Higher maturity is not necessary for every project; Research
can be the appropriate long-term state for a repository.

Use the repository's README and version-specific documentation to identify its current
level, supported interfaces, and supported environments. A release number alone does
not establish a maturity level.

## L0 — Research { #l0-research }

![Maturity: L0 Research](https://img.shields.io/badge/Maturity-L0%20Research-76B900)

**Purpose:** Explore ideas and demonstrate research outcomes.

**Usage expectations:**

- Code, notebooks, or experiments demonstrate the core idea.
- The research purpose, scope, license, setup requirements, and major limitations
  are documented.
- Users should expect to adapt the software to their own experiments and environments.

**Compatibility and support:**

Interfaces and workflows may change rapidly. No API stability, release cadence, or
ongoing maintenance guarantee is implied. Check the repository's documentation for
available ways to ask questions or report problems.

## L1 — Emerging { #l1-emerging }

![Maturity: L1 Emerging](https://img.shields.io/badge/Maturity-L1%20Emerging-A4C639)

**Purpose:** Enable use beyond the original authors and support initial integration
with other software.

**Usage expectations:**

- Installation instructions, basic usage, and representative examples are available.
- The project is installable with a reproducible environment or dependency specification.
- Integration paths and supported environments are documented.
- Public APIs have useful docstrings, and automated unit tests cover supported behavior.

**Compatibility and support:**

Another project can integrate the software, but formal backward-compatibility
guarantees are not yet implied. Some interfaces may be relatively stable while others
are evolving. Plan to validate upgrades and adapt integrations when APIs change.

Contribution guidance, issue templates, and pull-request templates provide a way to
report problems and propose improvements. Automated formatting, linting, and CI checks
support ongoing development.

## L2 — Hardened { #l2-hardened }

![Maturity: L2 Hardened](https://img.shields.io/badge/Maturity-L2%20Hardened-5C6BC0)

**Purpose:** Support broad adoption through managed compatibility commitments and
production-quality reference guidance.

**Usage expectations:**

- Comprehensive user and API documentation is published on a hosted website.
- Maintained end-to-end reference workflows, architecture documentation, and deployment
  recipes explain how to use the software in larger systems.
- User-facing examples, operational guidance, and migration paths are documented.
- CI validates the supported Python, dependency, and platform compatibility matrix.

**Compatibility and support:**

A formal versioning policy, such as semantic versioning, and a maintained changelog
communicate release changes. Supported public APIs and backward-compatibility boundaries
are explicitly documented and enforced by automated tests. A deprecation policy specifies
notice, transition guidance, and removal timelines.

!!! note "Stable and evolving interfaces"
    A Hardened repository may still contain experimental or evolving interfaces.
    Compatibility guarantees apply to the interfaces explicitly covered by its policy.
    Check the documentation for the interface and release you intend to use.

For deployments, follow the documented support matrix and operational guidance, and use
the migration instructions when upgrading.

## L3 — Strategic { #l3-strategic }

![Maturity: L3 Strategic](https://img.shields.io/badge/Maturity-L3%20Strategic-CB4B16)

**Purpose:** Sustain long-term use and ecosystem integration through continued
maintenance and release commitments.

**Usage expectations:**

- A repeatable release process produces versioned artifacts and release notes on a
  predictable cadence.
- Formal packaging and distribution, such as published PyPI packages, support adoption.
- Maintained ecosystem integrations support use alongside other software.

**Compatibility and support:**

Compatibility and deprecation commitments are sustained across releases. Established
contributor processes, security-response procedures, and ongoing project-health monitoring
support long-term maintenance. Published release and support policies explain how users
can plan upgrades and report issues.

## Choosing and using a repository

Choose a maturity level that matches your needs: experimentation, initial integration,
managed compatibility, or sustained ecosystem support. Before adopting or upgrading,
review the repository's:

- **Scope and limitations:** intended workflows and documented constraints.
- **Supported environments:** Python versions, dependencies, hardware, and platforms.
- **Compatibility policy:** stable APIs, experimental features, and deprecation timelines.
- **Release information:** versioned documentation, changelog, and migration guidance.
- **Support channels:** where to ask questions, report bugs, and disclose security issues.

Maturity describes engineering and maintenance commitments. Support availability,
response times, and supported release lifetimes are defined by each project's published
policies; a maturity badge alone does not specify them.

For Earth2Studio, see the [installation guide](../about/install.md),
[support pages](../support/index.md), and
[GitHub issue tracker](https://github.com/NVIDIA/earth2studio/issues).
Use documentation for your installed release when checking supported behavior.

## Reading and applying badges

A repository should display the badge for the level whose expectations it currently
meets. Completing some expectations of a higher level does not change its classification.
If its commitments change, the published level and documentation should be reassessed
and any change explained to users.

Copy the Markdown for the repository's current maturity level below.

### L0 badge

```markdown
![Maturity: L0 Research](https://img.shields.io/badge/Maturity-L0%20Research-76B900)
```

### L1 badge

```markdown
![Maturity: L1 Emerging](https://img.shields.io/badge/Maturity-L1%20Emerging-A4C639)
```

### L2 badge

```markdown
![Maturity: L2 Hardened](https://img.shields.io/badge/Maturity-L2%20Hardened-5C6BC0)
```

### L3 badge

```markdown
![Maturity: L3 Strategic](https://img.shields.io/badge/Maturity-L3%20Strategic-CB4B16)
```
