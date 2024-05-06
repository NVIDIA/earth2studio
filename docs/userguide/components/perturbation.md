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

:::{note}
Perturbations methods modify the input tensor directly. They are not functions that just
generate noise.
:::

## Perturbation Usage

All perturbation methods provide a {func}`__call__` function which takes in
a data tensor with coordinate system and returns the perturbed output.

```python
# Assume perturb is an instance of a Perturbation
x = torch.Tensor(...)  # Input tensor
coords = CoordSystem(...)  # Coordinate system
x, coords = perturb(x, coords)  # Predict a single time-step
```

## Normalization in Perturbations

As discussed in the {ref}`data_userguide` section, data is always moved between
components with physical units.
This implies that all perturbation methods will apply noise onto unnormalized data.
Naturally this is not ideal for many perturbation methods and its recommended users
extend the perturbation methods to conclude the normalization (and subsequent
denormalization).

## Custom Perturbation Methods

Integrating your own perturbation only requires inplementing the interface above.
We recommend users have a look at the {ref}`extension_examples` examples, which will
step users through the simple process of implementing their own perturbation method.

:::{warning}
TODO: 🚧 Under construction 🚧
:::

## Contributing a Perturbation Method

Want to add your perturbation to the package? Great, we will be happy to work with you.
At the minimum we expect the model to abide by the defined interface as well as meet
the requirements set forth in our contribution guide.

Open an issue when you have an initial implementation you would like us to review.
