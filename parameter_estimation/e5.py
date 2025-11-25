"""
# E5: Characterize bed.

- Parameters: bed porosity and axial dispersion
- Flow sheet: Full system
- Injected component: Blue dextran (2 MDa)
- Concentration: 0.0005 mM
- Volume: 50e-9 m³
- Eluent: A
- Measurement: UV

Status: done

"""

# %% Imports

from CADETProcess.processModel import GeneralRateModel, LumpedRateModelWithPores
from cadetrdm import Options, ProjectRepo, tracks_results

from knauer import PulseInjection, KnauerExperimentalData
from calibration import correct_baseline_and_normalize
from characterization import (
    CharacterizeBed,
    optimize,
)
from parameters import (
    component_system_dextran,
    knauer_system_options,
    c_dextran,
    n_sample_dextran,
    time_offset,
    cycle_time_bed,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

solution_path = "tubing_post_column.outlet"
components = None

knauer_system_options = knauer_system_options.copy()
knauer_system_options['ColumnModel'] = LumpedRateModelWithPores
knauer_system_options['BindingModel'] = None

start_peak = 3*60
peak_max = 4*60

DEFAULT_OPTIONS = Options({
    "prior_branch_name": None,
    "commit_message": "E5",
    "debug": False,
    "push": True,
})


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e5.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = correct_baseline_and_normalize(
        knauer_data.uv_1, target_area=n_sample_dextran
    )

    return reference


def setup_process():
    """Set up process."""
    return PulseInjection(
        "e5",
        component_system_dextran,
        knauer_system_options,
        c_buffer_a=[0],
        c_sample=[c_dextran],
        cycle_time=cycle_time_bed,
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
        CharacterizationType=CharacterizeBed,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        start_times=start_peak,
        end_times=peak_max,
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

    e5_optimization_results, posteriour_branch_name = main(options)
