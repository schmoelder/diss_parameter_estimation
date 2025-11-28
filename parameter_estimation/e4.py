"""
# E4: Characterize pre-injection system.

- Parameters: mixer volume and tubing length
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

from CADETProcess.reference import ReferenceIO
from cadetrdm import Options, ProjectRepo, tracks_results
import numpy as np

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

DEFAULT_OPTIONS = Options({
    "prior_branch_name": None,
    "commit_message": "E4",
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
        CharacterizationType=CharacterizePreInjection,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
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

    e4_optimization_results, posteriour_branch_name = main(options)
