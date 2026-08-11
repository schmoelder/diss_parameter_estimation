# Experiments

The experiments and their role in the parameter estimation workflow are described in Chapter 5 of the dissertation.
See the experiment overview table in `doc/05_characterization/01_materials_and_methods.md` for the authoritative list of experiments, injected components, volumes, eluents, and fitted parameters.

The output repository can be found at: [https://github.com/schmoelder/diss_parameter_estimation_output](https://github.com/schmoelder/diss_parameter_estimation_output)


# Reproducing the study

Create a fresh environment from the pinned specification and run the full study from the package directory:

```bash
git clone https://github.com/schmoelder/diss_parameter_estimation.git
cd diss_parameter_estimation
conda env create -f environment.yml
conda activate diss_parameter_estimation
cd parameter_estimation
python run_all.py
```

Runs with push access to the output repository can publish CADET-RDM result branches automatically.
Without push access, the calculations can still be reproduced locally from the pinned environment and source state, but result branches remain local or pushing must be disabled.

The scripted reproduction follows the model choices used for the reported study.
Intermediate model-selection decisions were made during the original analysis and are encoded in `run_all.py`.
