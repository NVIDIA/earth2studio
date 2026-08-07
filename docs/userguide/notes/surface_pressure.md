---
orphan: true
---

# Surface Pressure Interpolation

This documentation explains how surface pressure is estimated from pressure-level geopotential (and
optionally temperature) by log-pressure interpolation, plus second-order lapse-rate and
empirical corrections. Familiarity with the
hydrostatic equation and ideal gas law is helpful.

## Variables

The following variables are used below:

| **Symbol** | **Variable** | **Value (if constant)** |
| --- | --- | --- |
| $g$ | Gravitational acceleration | $9.8067\ \mathrm{m\,s^{-2}}$ |
| $R_\mathrm{s}$ | Gas constant for dry air | $287.053\ \mathrm{J\,kg^{-1}\, K^{-1}}$ |
| $L$ | Average temperature lapse rate | $-6.5 \times 10^{-3}\ \mathrm{K\,m^{-1}}$ |
| $p$ | Pressure | |
| $\rho$ | Density | |
| $h$ | Height | |
| $\Phi$ | Geopotential | |
| $T$ | Temperature | |
| $L_\Phi$ | Lapse rate w.r.t. geopotential $L/g$ | |
| $\beta$ | Inverse temperature $T^{-1}$ | |

## Physical Basis

The rate of pressure change by altitude is given by the hydrostatic equation, which can be written as:

$$
    \mathrm{d}p = -g \rho \, \mathrm{d}h = -\rho \, \mathrm{d}\Phi.
$$

When combined with the ideal gas law:

$$
p = \rho R_\mathrm{s} T
$$

We get:

$$
\frac{\mathrm{d}p}{\mathrm{d}\Phi} = -\frac{p}{R_\mathrm{s}T}
$$

Which is equivalent to:

$$
\label{eq:log_p_deriv}
\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} = -\frac{1}{R_\mathrm{s}T}
$$

## Log-linear Interpolation

The relative change of temperature by altitude, when expressed as absolute temperature
(that is, in Kelvin), is rather slow.

For example, a $10~\mathrm{K}$ change in
temperature is only $\approx 3.6\%$ at $273.15 K = 0~°\mathrm{C}$.
Therefore, a reasonable first-order approximation is to assume that $T$ is
constant over a short interval and thus, by equation $\eqref{eq:log_p_deriv}$, so is
$\frac{\mathrm{d}\log p}{\mathrm{d}\Phi}$. Under this approximation,
$\log p$ has constant slope with respect to $\Phi$ and the integral:

$$
\int_{\Phi_0}^{\Phi}\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} \, \mathrm{d}\Phi
$$

In the interval $\Phi_0 \leq \Phi \leq \Phi_1$ can be computed by linear interpolation as:

$$
\label{eq:log_linear_interp}
\log p(\Phi) = \log p_0 + \frac{\Phi - \Phi_0}{\Phi_1 - \Phi_0} (\log p_1 - \log p_0)
$$

Where $p_0 = p(\Phi_0)$ and $p_1 = p(\Phi_1)$ are the known values of
pressure at the ends of the interval.

## Second-order Correction for Temperature Lapse Rate

Typically, the temperature locally changes at an approximately linear rate as a function
of altitude, and thus the slope of $\log p$ is not exactly constant. We can model
the temperature around $\Phi_\mathrm{m} = \frac{1}{2}(\Phi_0+\Phi_1)$ as:

$$
\label{eq:lapse_rate}
T(\Phi) = T_\mathrm{m} + \frac{L}{g}(\Phi-\Phi_\mathrm{m}) = T_\mathrm{m} + L_\Phi(\Phi-\Phi_\mathrm{m})
$$

Where $T_\mathrm{m} = T(\Phi_\mathrm{m})$. Plugging this into equation $\eqref{eq:log_p_deriv}$
makes it complicated to solve as $T$ is in the denominator. However, we can
instead use the inverse temperature $\beta = T^{-1}$, which can also be
approximated to have a linear relationship with $\Phi$. This can be justified with
the Taylor series of the inverse of equation $\eqref{eq:lapse_rate}$ at $\Phi_\mathrm{m}$:

$$
\beta(\Phi) =
\frac{1}{T_\mathrm{m}}\left (1 - \frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}} +
O\left ( \left (\frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}} \right )^2
\right ) \right )
$$

Thus for small relative variations of temperature we can model $\beta$ as:

$$
\beta(\Phi) = \frac{1}{T} = \frac{1}{T_\mathrm{m}} - \frac{L_\Phi (\Phi-\Phi_\mathrm{m})}{T_\mathrm{m}^2}
$$

Equation $\eqref{eq:log_p_deriv}$ then becomes:

$$
\label{log_p_deriv_corr}
\frac{\mathrm{d}\log p}{\mathrm{d}\Phi} = -\frac{1}{R_\mathrm{s}T_\mathrm{m}} + \frac{L_\Phi(\Phi-\Phi_\mathrm{m})}{R_\mathrm{s}T_\mathrm{m}^2}
$$

The first term on the right-hand side is identical to equation $\eqref{eq:log_p_deriv}$, so we
can compute it using the log-linear interpolation of equation $\eqref{eq:log_linear_interp}$ and
consider the second term as a correction:

$$
\label{log_p_corr}
\Delta \log p = \frac{L_\Phi(\Phi-\Phi_\mathrm{m})}{R_\mathrm{s}T_\mathrm{m}^2}
$$

Within an interval $\Phi_0 \leq \Phi \leq \Phi_1$ where the pressure is known at
the endpoints $(\Phi_0, \Phi_1)$, we know that the correction should be zero at
those endpoints. Also, the integration of a linear relationship will give a quadratic
function. The only function that satisfies those constraints is of the form:

$$
\Delta \log p = c(\Phi-\Phi_0)(\Phi-\Phi_1)
$$

And the derivative of this is:

$$
\frac{\mathrm{d}\Delta \log p}{\mathrm{d}\Phi} =
c((\Phi-\Phi_0) + (\Phi-\Phi_1)) = 2c(\Phi - \Phi_\mathrm{m})
$$

Comparing this to equation $\eqref{log_p_deriv_corr}$ shows that we should have:

$$
c = \frac{L_\Phi}{2R_\mathrm{s}T_\mathrm{m}^2} = \frac{L}{2gR_\mathrm{s}T_\mathrm{m}^2}
$$

## Empirical Correction

In principle, we could compute the lapse rate $L_\Phi$ from the temperature of the
endpoints of the interpolation interval. However, this does not seem to yield good
results in practice. This may be because differences like this (similar to numerical
derivatives) tend to be sensitive to noise, or because the near-surface temperature is
often affected by the surface. We get better results by using the average lapse rate of
$-6.5 \times 10^{-3}\ \mathrm{K\,m^{-1}}$.

Applying the correction from equation $\eqref{log_p_corr}$ using the average lapse
rate improves the approximation result but leaves some remaining bias and room for
improvement in the mean square error (MSE). To reduce this, we apply a further linear
correction:

$$
    (\Delta \log_p)_\mathrm{adj} = a + b\Delta \log_p
$$

Where $a$ and $b$ are chosen empirically to minimize the MSE of the
approximated surface pressure over one year of ERA5 data. This analysis gives
$a = 3.4257 \times 10^{-5}$ and $b = 1.5224$. Equivalently, the value of $b$
indicates that the optimal assumed lapse rate is
$-9.9 \times 10^{-3}\ \mathrm{K\,m^{-1}}$. This is within the physical range of
variability but corresponds to a highly unstable atmosphere. Therefore it seems that
the correction also compensates some other errors besides that due to varying
temperature.
