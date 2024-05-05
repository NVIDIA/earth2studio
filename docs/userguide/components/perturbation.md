(perturbation_userguide)=

# Perturbations

Perturbations are an integral part of ensembling workflows and a variety are built into
Earth2Studio. These can range from standard noise perturbations to more complex methods
such as Bred Vector.

The list of perturbation methods that are already built into Earth2studio can be found
in the API documentation {ref}`earth2studio.perturbation`.

## Perturbation Interface

The full requirements for a perturbation method are defined explicitly in the
`earth2studio/perturbation/base.py`.

```{literalinclude} ../../../earth2studio/perturbation/base.py
:lines: 24-
:language: python
```
