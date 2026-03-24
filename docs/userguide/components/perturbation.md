(perturbation_userguide)=

# Perturbations

Perturbations add controlled noise to initial conditions for ensemble generation.
They are an integral part of ensembling workflows and a variety are built into
Earth2Studio. These can range from standard noise perturbations to more complex methods
such as Bred Vector.

The list of perturbation methods that are already built into Earth2Studio can be found
in the API documentation {ref}`earth2studio.perturbation`.
For complete workflows that use perturbations, refer to, for example, `03_ensemble_workflow` and `05_ensemble_workflow_extend`.

## Perturbation Interface

The full requirements for a perturbation method are defined explicitly in the
`earth2studio/perturbation/base.py`.

```{literalinclude} ../../../earth2studio/perturbation/base.py
:lines: 24-
:language: python
```

:::{note}
Perturbation methods modify the input tensor directly. They are not functions that just
generate noise.
:::

## Perturbation Usage

All perturbation methods provide a {func}`__call__` function, which takes in
a data tensor with coordinate system and returns the perturbed output.

```python
# Assume perturb is an instance of a Perturbation
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = perturb(x, coords)  # Apply perturbation
```

## Normalization in Perturbations

As discussed in the {ref}`data_userguide` section, data is always moved between
components with physical units.
This implies that all perturbation methods will apply noise onto unnormalized data.
This is not ideal for many perturbation methods and it is recommended that you
extend the perturbation methods to include the normalization (and subsequent
denormalization).

## Custom Perturbation Methods

Integrating your own perturbation only requires implementing the interface above.
We recommend that you review {ref}`extension_examples`, which will
step you through the basic process of implementing your own perturbation method.

:::{warning}
TODO: 🚧 Under construction 🚧
:::

## Contributing a Perturbation Method

Want to add your perturbation to the package? We are happy to work with you.
At a minimum, we expect the perturbation to abide by the defined interface and to meet
the requirements set forth in our contribution guide.

Open an issue when you have an initial implementation you would like us to review.
