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

from knauer import PulseInjection, KnauerExperimentalData
from calibration import normalize_area
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


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e1.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = normalize_area(knauer_data.uv_1, n_sample_acetone)

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
    process,
    reference,
    debug=False,
):
    """Run optimization."""
    commit_message = "E1"

    return optimize(
        process,
        CharacterizationType=CharacterizeTubing,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        characterization_options={"tubing": tubing},
        optimizer_options=optimizer_options,
        prior_branch_name=None,
        commit_message=commit_message,
        debug=debug,
    )


# %% Run optimization

def main(prior_branch_name=None, debug=False, use_synthetic_data=False):
    # Setup reference data
    reference = None if use_synthetic_data else setup_reference()

    # Setup process
    process = setup_process()

    # Run optimization
    return run_optimization(process, reference, debug)


if __name__ == "__main__":
    debug = False
    use_synthetic_data = False

    e1_optimization_results, posteriour_branch_name = main(use_synthetic_data, debug)
