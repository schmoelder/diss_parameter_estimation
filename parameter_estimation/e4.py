"""
# E4: Characterize pre-injection system.

- Parameters: mixer volume, tubing length and axial dispersion
- Flow sheet: Bypass column
- Injected component: -
- Concentration: -
- Volume: -
- Eluent: A -> B (step)
- Measurement: Cond

Note, the original data seems missing, so here, we use the already calibrated data.

Status: done

"""

# %% Imports

import numpy as np
from CADETProcess.reference import ReferenceIO

from knauer import Step
from characterization import (
    CharacterizePreInjection,
    optimize,
)
from parameters import (
    component_system_salt,
    knauer_system_options_bypass_column,
    c_salt_high,
    time_offset,
    cycle_time_system_periphery,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

solution_path = "outlet.outlet"
components = None


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e4.csv"

    data = np.loadtxt(file_path, delimiter=',')
    time = data[:, 0]
    signal = data[:, 1]

    # Time offset to account for delayed injection.
    time = time - time_offset

    # Remove negative time values and synchronize concentration array
    valid_indices = np.where((time >= 0))
    time = time[valid_indices]
    signal = signal[valid_indices]

    reference = ReferenceIO("", time, signal, flow_rate)

    return reference


def setup_process():
    """Set up process."""
    return Step(
        "e4",
        component_system_salt,
        knauer_system_options_bypass_column,
        c_buffer_a=[0],
        c_buffer_b=[c_salt_high],
        cycle_time=cycle_time_system_periphery,
        flow_rate=flow_rate,
    )


def run_optimization(
    process,
    reference,
    prior_branch_name=None,
    debug=False
):
    """Run optimization."""
    commit_message = "E4"
    if prior_branch_name is None:
        commit_message += "_synthetic"
    elif prior_branch_name == "parameters_lukas":
        commit_message += "_lukas"

    return optimize(
        process,
        CharacterizationType=CharacterizePreInjection,
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
    # Setup process
    process = setup_process()

    # Setup reference data
    reference = None if prior_branch_name is None else setup_reference()

    # Run optimization
    return run_optimization(process, reference, prior_branch_name, debug)


if __name__ == "__main__":
    debug = False
    prior_branch_name = None

    e4_optimization_results, posteriour_branch_name = main(prior_branch_name, debug)
