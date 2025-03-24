# UV Calibration

To compare simulations with experimental data, the UV signal must also be converted into concentration values.
A calibration curve can be generated for different protein concentrations, and a trendline can be used for interpolation.
For this work, however, the UV signal is rescaled using the mass balance.
For each experiment, a known amount of protein is injected into the column.
The UV signal is then rescaled so that its integral matches the inserted protein amount.
To define the peak's start and endpoint for integration, start and stop times are set.
Baseline correction is also applied by adjusting the UV signal to zero if necessary.
