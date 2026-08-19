# Perturbations

`earth2studio.perturbation`

Perturbation methods are used for perturbing the input data they are provided, typically
with some random noise. This is commonly done to perturb initial state fields when
creating ensemble forecasts.

<!-- e2s-autosummary
currentmodule: earth2studio
template: perturbation
output: generated/perturbation/1
-->

{% autosummary %}
earth2studio.perturbation.Brown
earth2studio.perturbation.BredVector
earth2studio.perturbation.CorrelatedSphericalGaussian
earth2studio.perturbation.Gaussian
earth2studio.perturbation.HemisphericCentredBredVector
earth2studio.perturbation.LaggedEnsemble
earth2studio.perturbation.SphericalGaussian
earth2studio.perturbation.Zero
{% endautosummary %}
