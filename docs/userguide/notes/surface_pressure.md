---
orphan: true
---

# Surface pressure interpolation

This documentation describes the surface pressure interpolation provided by `earth2studio.models.dx.derived.DerivedSurfacePressure`.

## Variables

The following variables are used below:

| **Symbol** | **Variable** | **Value (if constant)** |
| --- | --- | --- |
| {math}`g` | Gravitational acceleration | {math}`9.8067\ \mathrm{m\,s^{-2}}` |
| {math}`R_\mathrm{s}` | Gas constant for dry air | {math}`287.053\ \mathrm{J\,kg^{-1}\, K^{-1}}` |
| {math}`L` | Average temperature lapse rate | {math}`-6.5 \times 10^{-3}\ \mathrm{K\,m^{-1}}` |
| {math}`p` | Pressure | |
| {math}`\rho` | Density | |
| {math}`h` | Height | |
| {math}`\Phi` | Geopotential | |
| {math}`T` | Temperature | |
| {math}`L_\Phi` | Lapse rate w.r.t. geopotential {math}`L/g` | |
| {math}`\beta` | Inverse temperature {math}`T^{-1}` | |

## Physical basis

The rate of pressure change by altitude is given by the hydrostatic equation, which can be written as

```{math}
    \mathrm{d}p = -g \rho \, \mathrm{d}h = -\rho \, d\Phi.
```

When combined with the ideal gas law

```{math}
p = \rho R_\mathrm{s} T
```

we get

```{math}
\frac{\mathrm{d}p}{\mathrm{d}\Phi} = -\frac{p}{R_\mathrm{s}T}
```

which is equivalent to

```{math}
:label: eq:log_p_deriv
\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} = -\frac{1}{R_\mathrm{s}T}
```

## Log-linear interpolation

The relative change of temperature by altitude, when expressed as absolute temperature
(i.e. in Kelvin), is rather slow: for example, a {math}`10~\mathrm{K}` change in
temperature is only {math}`\approx 3.6\%` at {math}`273.15 K = 0~°\mathrm{C}`.
Therefore, a reasonable first-order approximation is to assume that {math}`T` is
constant over a short interval and thus, by Eq. {eq}`eq:log_p_deriv`, so is
{math}`\frac{\mathrm{d}\log p}{\mathrm{d}\Phi}`. Under this approximation,
{math}`\log p` has constant slope with respect to {math}`\Phi` and the integral

```{math}
\int_{\Phi_0}^{\Phi}\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} \, \mathrm{d}\Phi
```

in the interval {math}`\Phi_0 \leq \Phi \leq \Phi_1` can be computed by linear interpolation as

```{math}
:label: eq:log_linear_interp
\log p(\Phi) = \log p_0 + \frac{\Phi - \Phi_0}{\Phi_1 - \Phi_0} (\log p_1 - \log p_0)
```

where {math}`p_0 = p(\Phi_0)` and {math}`p_1 = p(\Phi_1)` are the known values of
pressure at the ends of the interval.

## Second-order correction for temperature lapse rate

Typically, the temperature locally changes at an approximately linear rate as a function
of altitude, and thus the slope of {math}`\log p` is not exactly constant. We can model
the temperature around {math}`\Phi_\mathrm{m} = \frac{1}{2}(\Phi_0+\Phi_1)` as

```{math}
:label: eq:lapse_rate
T(\Phi) = T_\mathrm{m} + \frac{L}{g}(\Phi-\Phi_\mathrm{m}) = T_\mathrm{m} + L_\Phi(\Phi-\Phi_\mathrm{m})
```

where {math}`T_\mathrm{m} = T(\Phi_m)`. Plugging this into Eq. {eq}`eq:log_p_deriv`
makes it complicated to solve as {math}`T` is in the denominator. However, we can
instead use the inverse temperature {math}`\beta = T^{-1}`, which can also be
approximated to have a linear relationship with {math}`\Phi`. This can be justified with
the Taylor series of the inverse of Eq. {eq}`eq:lapse_rate` at {math}`\Phi_\mathrm{m}`:

```{math}
\beta(\Phi) =
\frac{1}{T_\mathrm{m}}\left (1 - \frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}} +
O\left ( \left (\frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}} \right )^2
\right ) \right )
```

Thus for small relative variations of temperature we can model {math}`\beta` as:

```{math}
\beta(\Phi) = \frac{1}{T} = \frac{1}{T_\mathrm{m}} - \frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}^2}
```

Equation {eq}`eq:log_p_deriv` then becomes

```{math}
:label: log_p_deriv_corr
\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} = -\frac{1}{R_\mathrm{s}T_\mathrm{m}} + \frac{L_\Phi(\Phi-\Phi_\mathrm{m})}{R_\mathrm{s}T_\mathrm{m}^2}
```

The first term on the right-hand side is identical to Eq. {eq}`eq:log_p_deriv`, so we
can compute it using the log-linear interpolation of Eq. {eq}`eq:log_linear_interp` and
consider the second term as a correction

```{math}
:label: log_p_corr
\Delta \log p = \frac{L_\Phi(\Phi-\Phi_\mathrm{m})}{R_\mathrm{s}T_\mathrm{m}^2}
```

Within an interval {math}`\Phi_0 \leq \Phi \leq \Phi_1` where the pressure is known at
the endpoints {math}`(\Phi_0, \Phi_1)`, we know that the correction should be zero at
those endpoints. Also, the integration of a linear relationship will give a quadratic
function. The only function that satisfies those constraints is of the form

```{math}
\Delta \log p = c(\Phi-\Phi_0)(\Phi-\Phi_1)
```

and the derivative of this is

```{math}
\frac{\mathrm{d}\Delta \log p}{\mathrm{d}\Phi} =
c((\Phi-\Phi_0) + (\Phi-\Phi_1)) = 2c(\Phi - \Phi_m)
```

Comparing this to Eq. {eq}`log_p_deriv_corr` shows that we should have

```{math}
c = \frac{L_\Phi}{2R_\mathrm{s}T_\mathrm{m}^2} = \frac{L}{2gR_\mathrm{s}T_\mathrm{m}^2}
```

## Empirical correction

In principle, we could compute the lapse rate {math}`L_\Phi` from the temperature of the
endpoints of the interpolation interval. However, this does not seem to yield good
results in practice. This may be because differences like this (similar to numerical
derivatives) tend to be sensitive to noise, or because the near-surface temperature is
often affected by the surface. We get better results by using the average lapse rate of
{math}`-6.5 \times 10^3\ \mathrm{K\,m^{-1}}`.

Applying the correction from Eq.~(\ref{eq:correction_approx}) using the average lapse
rate improves the approximation result but leaves some remaining bias and room for
improvement in the mean square error (MSE). To reduce this, we apply a further linear
correction

```{math}
    (\Delta \log_p)_\mathrm{adj} = a + b\Delta \log_p
```

where {math}`a` and {math}`b` are chosen empirically to minimize the MSE of the
approximated surface pressure over one year of ERA5 data. This analysis gives
{math}`a = 3.4257 \times 10^{-5}` and {math}`b = 1.5224`. Equivalently, the value of {math}`b`
indicates that the optimal assumed lapse rate is
{math}`-9.9 \times 10^3\ \mathrm{K\,m^{-1}}`. This is within the physical range of variability but corresponds
to a highly unstable atmosphere. Therefore it seems that the correction also compensates
some other errors besides that due to varying temperature.
