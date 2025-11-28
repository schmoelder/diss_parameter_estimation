"""
# E2: Characterize post-column tubing.

- Parameters: tubing length and axial dispersion
- Flow sheet: Bypass column, detector
- Injected component: Acetone
- Concentration: 171.2 mM (1 %)
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

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
    component_system_acetone,
    knauer_system_options_bypass_column_detector,
    c_acetone,
    n_sample_acetone,
    time_offset,
    cycle_time_system_periphery,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

tubing = "tubing_post_column"
solution_path = f"{tubing}.outlet"
components = None

DEFAULT_OPTIONS = Options({
    "prior_branch_name": None,
    "commit_message": "E2",
    "debug": False,
    "push": True,
    "_temp_directory_base": None,
    "_cache_directory_base": None,
    "_cadet_options": {
        "install_path": None,
        "use_dll": True,
    },
})


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e2.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = correct_baseline_and_normalize(
        knauer_data.uv_1, target_area=n_sample_acetone
    )

    return reference


def setup_process():
    """Set up process."""
    return PulseInjection(
        "e2",
        component_system_acetone,
        knauer_system_options_bypass_column_detector,
        c_buffer_a=[0],
        c_sample=[c_acetone],
        cycle_time=cycle_time_system_periphery,
        flow_rate=flow_rate,
    )


def run_optimization(
    results_directory,
    process,
    reference,
    prior_branch_name=None,
    cache_directory_base=None,
    temp_directory_base=None,
    cadet_options=None,
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
        cache_directory_base=cache_directory_base,
        temp_directory_base=temp_directory_base,
        cadet_options=cadet_options,
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
        prior_branch_name=options.prior_branch_name,
        temp_directory_base=options._temp_directory_base,
        cache_directory_base=options._cache_directory_base,
        cadet_options=options._cadet_options,
    )


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.debug = True
    options.prior_branch_name = None

    e2_optimization_results, posteriour_branch_name = main(options)
