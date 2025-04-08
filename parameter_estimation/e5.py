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

from knauer import PulseInjection, KnauerExperimentalData
from calibration import normalize_area
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


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e5.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = normalize_area(knauer_data.uv_1, n_sample_dextran)

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
    process,
    reference,
    prior_branch_name=None,
    debug=False,
):
    """Run optimization."""
    commit_message = "E5"
    if prior_branch_name is None:
        commit_message += "_synthetic"
    elif prior_branch_name == "parameters_lukas":
        commit_message += "_lukas"

    return optimize(
        process,
        CharacterizationType=CharacterizeBed,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        optimizer_options=optimizer_options,
        prior_branch_name=prior_branch_name,
        commit_message=commit_message,
        debug=debug,
    )


# %% Run optimization

def main(prior_branch_name=None, debug=False):
    # Setup reference data
    reference = None if prior_branch_name is None else setup_reference()

    # Setup process
    process = setup_process()

    # Run optimization
    return run_optimization(process, reference, prior_branch_name, debug)


if __name__ == "__main__":
    debug = False
    prior_branch_name = None

    e5_optimization_results, posteriour_branch_name = main(prior_branch_name, debug)
