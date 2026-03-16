Fragen an Lukas:
- CV (4.7 vs pi/4 * D² * L) vs Säulendurchmesser von Hersteller? -> ?
- Boundaries (nu) -> "experience"
- Warum überhaupt C6 (Acetone)? Alle Parameter werden später noch mal gefittet -> See capacity
- Capacity Skript vs Thesis -> Fit after C6
- Warum film diffusion noch mal in C9 fitten? -> Somehow not consistent with C7

# Experiments

1. Acetone tracer without column; tubing bevor column
- Goal: Determine length and dispersion of pre-column tubing
- Flow sheet: #1
- Injected component: Acetone
- Concentration: 171.2 mM (1 %)
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

2. Acetone tracer without column; tubing after column
- Goal: Determine length and dispersion of post-column tubing
- Flow sheet: #2
- Injected component: Acetone
- Concentration: 171.2 mM (1 %)
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

3. Salt pulse without column; tubing between UV and conductivity sensor
- Goal: Determine length and dispersion of detector tubing
- Flow sheet: #3
- Injected component: High-salt buffer (Buffer B?)
- Concentration: 1020 mM
- Volume: 50e-9 m³
- Eluent: A
- Measurement: Cond

4. Salt step without column; from buffer inlet
- Goal: Determine and volume of mixer
- Flow sheet: #4
- Injected component:
- Concentration: -
- Volume: -
- Eluent: A -> B (step)
- Measurement: Cond

5. Blue Dextran with column
- Goal: Determine external porosity (and axial dispersion)
- Flow sheet: #5
- Injected component: Blue dextran (2 mDa)
- Concentration: 0.0005 mM
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

6. Acetone with column
- Goal: Determine particle porosity / total porosity
- Flow sheet: #5
- Injected component: Acetone
- Concentration: 171.2 mM (1 %)
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

7. Protein in non-binding condition with column
- Goal: Film Diffusion / Pore Diffusion (/ Axial Dispersion)
- Flow sheet: #5
- Injected component: Lysozyme
- Concentration: 0.2 mM
- Volume: 50e-9 m³
- Eluent: B???
- Measurement: UV

The output repository can be found at: [https://github.com/schmoelder/diss_parameter_estimation_output](https://github.com/schmoelder/diss_parameter_estimation_output)
