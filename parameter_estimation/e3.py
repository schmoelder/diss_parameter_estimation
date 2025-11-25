"""
# E3: Characterize detector tubing.

- Parameters: tubing length and axial dispersion
- Flow sheet: Bypass column
- Injected component: High-salt buffer
- Concentration: 1020 mM
- Volume: 50e-9 m³
- Eluent: A
- Measurement: Cond

Status: done

"""

# %% Imports
from cadetrdm import Options, ProjectRepo, tracks_results

from knauer import PulseInjection, KnauerExperimentalData
from calibration import correct_baseline_and_normalize
from characterization import (
    CharacterizeTubing,
    optimize,
)
from parameters import (
    component_system_salt,
    knauer_system_options_bypass_column,
    c_salt_high,
    n_sample_salt,
    time_offset,
    cycle_time_system_periphery,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

tubing = "tubing_detectors"
solution_path = f"{tubing}.outlet"
components = None

DEFAULT_OPTIONS = Options({
    "prior_branch_name": None,
    "commit_message": "E3",
    "debug": False,
    "push": True,
})


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e3.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = correct_baseline_and_normalize(
        knauer_data.conductivity, target_area=n_sample_salt
    )

    return reference


def setup_process():
    """Set up process."""
    return PulseInjection(
        "e3",
        component_system_salt,
        knauer_system_options_bypass_column,
        c_buffer_a=[0],
        c_sample=[c_salt_high],
        cycle_time=cycle_time_system_periphery,
        flow_rate=flow_rate,
    )


def run_optimization(
    results_directory,
    process,
    reference,
    prior_branch_name=None,
):
    """Run optimization."""
    return optimize(
        results_directory,
        process,
        CharacterizationType=CharacterizeTubing,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        characterization_options={"tubing": tubing},
        optimizer_options=optimizer_options,
        prior_branch_name=prior_branch_name,
    )


# %% Run optimization

@tracks_results
def main(repo:ProjectRepo, options: Options):
    # Setup process
    process = setup_process()

    # Setup reference data
    reference = None if options.use_synthetic_data else setup_reference()

    # Run optimization
    return run_optimization(
        repo.output_path,
        process,
        reference,
        prior_branch_name=options.prior_branch_name
    )


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.debug = True
    options.prior_branch_name = None

    e3_optimization_results, posteriour_branch_name = main(options)
