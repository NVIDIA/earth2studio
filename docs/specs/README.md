# Specifications

Specs define the required, externally observable behavior of Earth2Studio
components. They are contracts for authors, callers, and reviewers, not evidence
that an implementation conforms. Code and tests must satisfy the applicable specs.

## Writing a spec

1. Copy [TEMPLATE.md](TEMPLATE.md) to `E2S-<NN>_<TOPIC>_SPEC.md`. Assign the next
   unused number, use the same ID in the title, and add it to the index below.
   Spec IDs are permanent; do not renumber or reuse them. Cite individual rules
   as, for example, `E2S-06/P1`.
2. Use **Purpose** and **Contract** as the top-level sections. Within Contract,
   group interfaces, requirements, verification, and other details under descriptive
   subheadings. State the scope and link related contracts.
3. Describe inputs, outputs, ordering, metadata, ownership, and failure behavior
   where relevant. Retain public API names only when they define the contract.
4. Write observable, testable requirements. **MUST** is required, **SHOULD** is
   recommended with a documented reason for exceptions, and **MAY** is optional.
5. Give requirements stable IDs, unique within the spec. Preserve existing IDs;
   never reuse a retired ID for a different requirement.
6. Add verification criteria covering normal use, boundaries, and failures. Keep
   them independent of test files and implementation techniques.
7. Review for ambiguity, conflicting requirements, and broken links. Update the
   spec when changing a contract; assess compatibility with callers and related specs.

For new requirements, use short, direct sentences. Avoid named-model examples,
internal classes, source paths, implementation recipes, rollout history, and
speculative features. Keep
research and unresolved proposals in `dev/scratch`; they are not requirements.

Existing specs retain their examples and notes; relocation and formatting do not
change their contracts.

## Index

Initial numbering follows foundational dependencies, then consumers. Integration
cross-references may point forward. New specs keep permanent IDs; list prerequisites
in their Contract section rather than renumbering existing specs.

| ID | Spec | Contract |
| --- | --- | --- |
| E2S-01 | [Core metadata and coordinate systems](E2S-01_CORE_METADATA_SPEC.md) | Coordinate objects, axis names and types, attributes, DataFrame schemas |
| E2S-02 | [Grids](E2S-02_GRID_SPEC.md) | Geometry, identity, coordinates, selection |
| E2S-03 | [Time statistics](E2S-03_TIME_STATISTICS_SPEC.md) | Temporal quantities, windows, source aggregation |
| E2S-04 | [Regridding](E2S-04_REGRID_SPEC.md) | Mapping fields between grids and composing sources |
| E2S-05 | [Data fetching](E2S-05_FETCH_DATA_SPEC.md) | Source requests, assembly, members, device placement |
| E2S-06 | [Models](E2S-06_MODEL_CONTRACT_SPEC.md) | Coordinates, execution, state, hooks, randomness, domains |
| E2S-07 | [Data assimilation](E2S-07_DA_MODEL_CONTRACT_SPEC.md) | Independent analyses and stateful assimilation |
| E2S-08 | [Perturbations](E2S-08_PERTURBATION_SPEC.md) | Transforming ensemble initial conditions |
| E2S-09 | [Statistics](E2S-09_STATISTICS_SPEC.md) | Field reductions and forecast metrics |
| E2S-10 | [IO](E2S-10_IO_SPEC.md) | Storage templates, writes, metadata, lifecycle |
| E2S-11 | [Execution](E2S-11_EXECUTION_CONTRACT_SPEC.md) | Work items, streams, ensembles, distribution, resume |
| E2S-12 | [Coupling](E2S-12_COUPLING_SPEC.md) | Field exchange and scheduling between components |
| E2S-13 | [STAC](E2S-13_STAC_SPEC.md) | Catalog discovery and raster access |
