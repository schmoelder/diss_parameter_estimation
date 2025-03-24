# Calibration of Conductivity Sensor

A conductivity detector is integrated into the chromatographic system to monitor salt concentration in real time during chromatography experiments.
Conductivity $\kappa$ describes a solution's ability to conduct electricity and is defined as

$$
\kappa = \frac{l}{A \cdot R},
$$

where $A$ is the conductor’s cross-sectional area, $l$ is its length, and $R$ is the resistance.
Conductivity is measured in siemens per meter ($S \cdot \text{m}^{-1}$), where $[S] = [\Omega^{-1}] = [A/V]$ @TODO: cite.
Higher ion density increases conductivity @TODO: cite.

To quantify the relationship between salt concentration and conductivity, a calibration curve is recorded by measuring the conductivity of solutions with varying salt concentrations at a constant pH of 5.
Conductivity values are recorded over one minute at predefined salt concentrations (20 mM, 270 mM, 570 mM, 770 mM, and 1020 mM).
Since the relationship between salt concentration and conductivity is nonlinear, a quadratic function is fitted to the measured data using the least squares method.
This function is then used to determine salt concentration from conductivity in subsequent analyses.
