"""
# E8: Resin capacity titration.

- Parameters: capacity
- Flow sheet: Full system
- Injected component: -
- Volume: -
- Eluent: A
- Measurement: Cond

Status: done

Note, this needs to be performed after c6 to get the correct total porosity.
For this purpose, the particle porosity of Acetone is used to determine the
total capacity which is then later adapted to a specific capacity for Lysozyme,
taking into account the lower apparent particle porosity, while keeping total
capacity constant.

"""

# %% Imports

import numpy as np
import pytest

from CADETProcess.processModel import LumpedRateModelWithPores
from CADETProcess.reference import ReferenceIO
from CADETProcess.solution import slice_solution

from knauer import KnauerExperimentalData, Step
from parameters import (
    component_system_naoh,
    knauer_system_options,
    c_dextran,
    n_sample_dextran,
    time_offset,
    cycle_time_bed,
    flow_rate,
    metrics,
    optimizer_options,
)

from utils import (
    experimental_data_path,
    tracks_results,
    load_parameters,
    save_parameters,
    update_process_parameters,
)


# %% Setup methods

pH = 12.260
c_NaOH = 10**-(14-pH)*1000  # mM

start_NaOH = 188 * 60   # s

knauer_system_options = knauer_system_options.copy()
knauer_system_options['ColumnModel'] = LumpedRateModelWithPores


def setup_references():
    references = []
    for run in range(3):
        file_path = experimental_data_path / "e8" / f"run_{run}.csv"

        knauer_data = KnauerExperimentalData(
            file_path=file_path,
            flow_rate=flow_rate,
        )
        references.append(knauer_data.conductivity)
    return references


def setup_process():
    """Set up process."""
    return Step(
        "e8",
        component_system_naoh,
        knauer_system_options,
        c_buffer_a=[0],
        c_buffer_b=[c_NaOH],
        cycle_time=19000,
        flow_rate=flow_rate,
    )


# %% Process data

def remove_equilibration_signal(
        reference: ReferenceIO,
        start_elution,
        ) -> ReferenceIO:
    return slice_solution(reference, coordinates={"time": (start_elution, None)})


def determine_breakthrough(reference: ReferenceIO, percentage: float = 10) -> float:
    # Calculate the maximum value of the array
    max_value = reference.solution.max()

    # Calculate the threshold value
    threshold = (percentage / 100) * max_value

    # Find the indices where the values are above the threshold
    breakthrough_10_index = np.where(reference.solution > threshold)[0][0]
    breakthrough_10 = reference.time[breakthrough_10_index]

    return breakthrough_10


def calculate_NaOH_volume(breakthrough_10: float) -> float:
    """
    Calculate volume of NaOH used up to 10 % breakthrough.

    Parameters
    ----------
    breakthrough_10 : float
        The time or point at which 10 % breakthrough occurs.

    Returns
    -------
    float
        Volume of NaOH used to get to the 10 % breakthrough point.
    """
    return (breakthrough_10 - start_NaOH) * flow_rate


def calculate_total_capacity(V_NaOH: float, c_NaOH: float) -> float:
    """
    Calculate the total capacity based on NaOH volume and concentration.

    Parameters
    ----------
    V_NaOH : float
        The volume of NaOH.
    c_NaOH : float
        The concentration of NaOH.

    Returns
    -------
    float
        The total capacity based on the given NaOH volume and concentration.
    """
    return V_NaOH * c_NaOH


def calculate_total_column_capacity(
    column_volume: float,
    total_porosity: float,
    specific_capacity: float,
) -> float:
    """
    Calculate the total capacity of a column resin.

    Parameters
    ----------
    column_volume : float
        The volume of the column.
    total_porosity : float
        The total porosity of the column.
    specific_capacity : float
        The volume specific capacity of the column material.

    Returns
    -------
    float
        The total capacity of the column.
    """
    return specific_capacity * (1 - total_porosity) * column_volume


def calculate_specific_capacity(
    column_volume: float,
    total_porosity: float,
    total_capacity: float,
) -> float:
    """
    Calculate the specific capacity of the column material.

    Parameters
    ----------
    column_volume : float
        The volume of the column.
    total_porosity : float
        The total porosity of the column.
    total_capacity : float
        The total capacity of the column.

    Returns
    -------
    float
        The volume specific capacity of the column material.
    """
    return total_capacity / ((1 - total_porosity) * column_volume)


def calculate_V_NaOH_used():
    V_NaOHs = []
    breakthrough_10s = []

    references = setup_references()
    for reference in references:

        reference = remove_equilibration_signal(reference, start_NaOH)
        breakthrough_10 = determine_breakthrough(reference, percentage=10)
        breakthrough_10s.append(breakthrough_10)

        V_NaOH = calculate_NaOH_volume(breakthrough_10)
        V_NaOHs.append(V_NaOH)

    V_NaOH_mean = np.mean(V_NaOHs)

    return V_NaOH_mean


def set_total_capacity(knauer_process, total_capacity):
    column = knauer_process.flow_sheet.column

    specific_capacity = calculate_specific_capacity(
        column.volume, column.total_porosity, total_capacity
    )

    column.binding_model.capacity = specific_capacity
    column.q = [specific_capacity] + (column.n_comp - 1) * [0]


@tracks_results
def determine_total_capacity(
    prior_branch_name=None,
):
    parameters = load_parameters(prior_branch_name)
    process = setup_process()
    update_process_parameters(process, parameters)

    V_NaOH_used = calculate_V_NaOH_used()
    V_NaOH = V_NaOH_used - process.flow_sheet.system_dead_volume
    total_capacity = calculate_total_capacity(V_NaOH, c_NaOH)

    parameters["total_capacity"] = total_capacity
    parameters["prior_branch_name"] = prior_branch_name
    parameters["case"] = "e8"

    save_parameters(parameters)

    return total_capacity


# %% Main

def main(
    prior_branch_name=None,
    debug=False,
):
    commit_message = "E8"
    if prior_branch_name is None:
        commit_message += "_synthetic"

    return determine_total_capacity(
        prior_branch_name=prior_branch_name,
        debug=debug,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    debug = True
    prior_branch_name = None

    total_capacity, new_branch = main(
        prior_branch_name=prior_branch_name,
        debug=debug,
    )


# %% Pytest

@pytest.mark.parametrize(
    "column_volume, total_porosity, capacity, expected",
    [
        (2.0, 0.5, 1.0, 1.0),
        (1.0, 0.5, 1.0, 0.5),   # Half capacity
        (1.0, 1.0, 1.0, 0.0),   # No capacity
        (0.0, 0.5, 0.5, 0.0),   # No volume
        (1.0, 0.0, 1.0, 1.0),   # Only particles
        (1.0, 1.0, 1.0, 0.0),   # No particles
    ],
)
def test_total_capacity(column_volume, total_porosity, capacity, expected):
    assert calculate_total_column_capacity(
        column_volume, total_porosity, capacity,
    ) == expected


@pytest.mark.parametrize(
    "column_volume, total_porosity, total_capacity, expected",
    [
        (1.0, 0.5, 1.0, 2.0),
        (2.0, 0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0, ZeroDivisionError),
    ],
)
def test_specific_capacity(column_volume, total_porosity, total_capacity, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            assert calculate_specific_capacity(
                column_volume, total_porosity, total_capacity,
            )
    else:
        assert calculate_specific_capacity(
            column_volume, total_porosity, total_capacity,
        ) == expected


if __name__ == "__main__":
    run_tests = False
    if run_tests:
        pytest.main([__file__])
