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

from CADETProcess.reference import ReferenceIO
from CADETProcess.solution import slice_solution

from knauer import KnauerExperimentalData
from parameters import flow_rate
from utils import experimental_data_path


# %% Setup methods

pH = 12.260
c_NaOH = 10**-(14-pH)*1000  # mM

start_NaOH = 188 * 60   # s


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
    Calculate the volume of NaOH that flowed through the column up to 10 % breakthrough.

    Parameters
    ----------
    breakthrough_10 : float
        The time or point at which 10 percent breakthrough occurs.

    Returns
    -------
    float
        The volume of NaOH that has flowed through the column until the 10 percent
        breakthrough point.
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


def calculate_total_porosity(
        bed_porosity: float,
        particle_porosity: float,
        ) -> float:
    """
    Calculate the total porosity of a packed bed column.

    Parameters
    ----------
    bed_porosity : float
        The bed porosity.
    particle_porosity : float
        The particle porosity.

    Returns
    -------
    float
        The total porosity of the column, accounting for both bed and particle porosity.
    """
    return bed_porosity + (1 - bed_porosity) * particle_porosity


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


def calculate_V_NaOH():
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


def determine_total_capacity(V_NaOH_mean, knauer_process):
    system_dead_volume = knauer_process.flow_sheet.system_dead_volume
    V_NaOH = V_NaOH_mean - system_dead_volume
    total_capacity = V_NaOH * c_NaOH

    return total_capacity


def set_total_capacity(knauer_process, total_capacity):
    column = knauer_process.flow_sheet.column

    specific_capacity = calculate_specific_capacity(
        column.volume, column.total_porosity, total_capacity
    )

    column.binding_model.capacity = specific_capacity
    column.q = [specific_capacity] + (column.n_comp - 1) * [0]


def update_capacity(knauer_process):
    total_capacity = determine_total_capacity(knauer_process)
    set_total_capacity(knauer_process, total_capacity)


def __main__(
    prior_branch_name=None,
    debug=False,
):





# %% Pytest

@pytest.mark.parametrize(
    "bed_porosity, particle_porosity, expected",
    [
        (0.5, 0.5, 0.75),
        (0.5, 0.0, 0.5),    # Solid particles
        (0.0, 0.5, 0.5),    # Only particles
        (1.0, 0.5, 1.0),
    ],
)
def test_total_porosity(bed_porosity, particle_porosity, expected):
    assert calculate_total_porosity(bed_porosity, particle_porosity) == expected


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
    run_tests = True

    if run_tests:
        pytest.main([__file__])
