"""
# E1: Characterize pre-column tubing.

- Parameters: tubing length and axial dispersion
- Flow sheet: Bypass column, post-column tubing, detector
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
    knauer_system_options_bypass_column_post_detector,
    c_acetone,
    n_sample_acetone,
    time_offset,
    cycle_time_system_periphery,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Helper methods

tubing = "tubing_pre_column"
solution_path = f"{tubing}.outlet"
components = None

DEFAULT_OPTIONS = Options({
    "use_synthetic_data": False,
    "commit_message": "E1",
    "debug": False,
    "push": True,
})


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e1.csv"

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
        "e1",
        component_system_acetone,
        knauer_system_options_bypass_column_post_detector,
        c_buffer_a=[0],
        c_sample=[c_acetone],
        cycle_time=cycle_time_system_periphery,
        flow_rate=flow_rate,
    )


def run_optimization(
    results_directory,
    process,
    reference,
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
        prior_branch_name=None,
    )


# %% Run optimization

@tracks_results
def main(repo:ProjectRepo, options: Options):
    # Setup process
    process = setup_process()

    # Setup reference data
    reference = None if options.use_synthetic_data else setup_reference()

    # Run optimization
    return run_optimization(repo.output_path, process, reference)


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.use_synthetic_data = False
    options.debug = True

    e1_optimization_results, posteriour_branch_name = main(options)
