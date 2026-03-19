# Introduction

Earth2Studio is a Python package built to empower researchers, scientists,
and enthusiasts in the fields of weather and climate science with the latest artificial
intelligence models/capabilities.
With an intuitive design and comprehensive feature set, this package serves as a robust
toolkit for exploring this AI revolution in the weather and climate science domain.

## Package Design

Given that the goal of this package is to enable the user to extrapolate and build
beyond what is implemented here, we have focused on providing the building blocks to
enable this.

:::{admonition} Core Design Principles
:class: tip

- Modularity
- Simple and Explicit Data
- API Consistency
- User Progression
- Test Coverage and Stability
- Performance

(*in relative order of importance*)
:::

The design philosophy of Earth2Studio embodies a modular architecture where
the inference workflow acts as a flexible adhesive, seamlessly binding together various
specialized software components with well-defined interfaces.
Each component within the package serves a distinct purpose in typical inference
workflows.

```{figure} https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/0.1.0/docs/e2studio-arch.png
:alt: earth2studio-arch
:width: 500px
:align: center
```

By viewing the inference workflow as a dynamic connector, Earth2Studio
facilitates effortless integration of these components, allowing researchers to easily
swap out or augment functionalities to suit their specific needs.
We recognize that many users will have their own custom workflow needs, thus encourage
users to use the provided features as a starting point to build their own.

```{figure} https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/0.1.0/docs/e2studio-wf-samples.png
:alt: earth2studio-wf-samples
:width: 600px
:align: center
```

Significant importance is placed on the interface that enables the connection between
the components and the workflow.
These are simple python protocols that all variants of a particular component must share.
This not only enables a consistent API but also the generalization of workflows.

## Key Features

While Earth2Studio contains a large collection of general utilities,
functions and tooling the following six are considered the core.
For more information on these features, see the dedicated documentation for each.

:::::{grid} 3

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
Built-in Workflows
^^^
Multiple built-in inference workflows to accelerate your development and research.
::::

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
Prognostic Models
^^^
Support for the latest AI weather forecast models
offered under a coherent interface.
::::

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
Diagnostic Models
^^^
Diagnostic models for mapping to other quantities of interest.
::::

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
Datasources
^^^
Datasources to connect on-prem and remote data stores to inference workflows.
::::

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
IO
^^^
Simple, yet powerful IO utilities to export data for post-processing.
::::

::::{grid-item-card}
:margin: 2 2 0 0
:class-card: sd-text-black
Statistical Operators
^^^
Statistical methods to fuse directly into your inference workflow for more complex
uncertainty analysis.
::::

:::::
