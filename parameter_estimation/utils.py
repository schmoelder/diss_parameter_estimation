import copy
import functools
import json
import os
from pathlib import Path
from typing import Any, Optional

from cadetrdm import ProjectRepo

from CADETProcess.dataStructure import update_dict_recursively
from CADETProcess.processModel import ComponentSystem, StericMassAction, NoBinding

from parameters import column_length, column_diameter
from knauer import KnauerSystemProcess


repo = ProjectRepo(__file__)


def get_experimental_data_path() -> os.PathLike:
    """
    Get path for experimental data root directory.

    Note, this function uses CADET-RDM. The data is stored in the output repo.
    The branch name is currently hard-coded and needs to be updated in case the data
    changes.
    """
    branch_name = "2025-11-23_15-36-31_main_ab9cb48"
    cache_path = repo.copy_data_to_cache(branch_name)

    return cache_path / "experimental_data"


experimental_data_path = get_experimental_data_path()


def load_parameters_from_file(file_name: os.PathLike) -> dict:
    """Load parameters from file."""
    with open(file_name) as f:
        parameters = json.load(f)

    return parameters


def load_synthetic_parameters() -> dict:
    """Load synthetic parameters."""
    return load_parameters_from_file(Path(__file__).parent / "parameters_synthetic.json")


def load_parameters_lukas():
    """Load parameters as estimated by Lukas Thiel."""
    return load_parameters_from_file(Path(__file__).parent / "parameters_lukas.json")


def load_parameters_from_previous_run(
    branch_name: str
) -> dict:
    """Load parameters from previous run."""
    cache_path = repo.copy_data_to_cache(branch_name)
    parameters_path = cache_path / "parameters.json"
    return load_parameters_from_file(parameters_path)


def load_parameters(prior_branch_name: Optional[str]) -> dict:
    """Load parameters."""
    if prior_branch_name is None:
        return load_synthetic_parameters()
    elif prior_branch_name == "parameters_lukas":
        return load_parameters_lukas()

    return load_parameters_from_previous_run(prior_branch_name)


def load_all_parameters(prior_branch_name: str) -> dict:
    """Return dict with parameters used for each sequential experiment."""
    parameters = {}
    while prior_branch_name is not None:
        case_parameters = load_parameters(prior_branch_name)
        name = case_parameters["case"]
        parameters[name] = case_parameters
        prior_branch_name = case_parameters["prior_branch_name"]

    return parameters


def save_parameters(parameters: dict, use_synthetic_data: bool = False) -> None:
    """Save parameters to file."""
    file_name = "parameters"

    if use_synthetic_data:
        file_name += "_synthetic_recovered"

    with open(repo.output_path / f"{file_name}.json", "w") as f:
        json.dump(parameters, f, indent=4)


def resolve_component_values(value: Any, component_system: ComponentSystem) -> Any:
    """
    Replace component-based dicts with lists.

    Parameters
    ----------
    value : Any
        The value to process. If it's a dictionary with component names as keys,
        it will be replaced with a corresponding list.
    component_system : ComponentSystem
        The list of components to resolve dictionary keys against.

    Returns
    -------
    Any
        The resolved value, either unchanged or transformed into lists recursively.
    """
    if isinstance(value, dict):
        # Check if this dictionary contains component-based values
        if all(comp.name in value for comp in component_system):
            return [value[comp.name] for comp in component_system]
        else:
            # Recursively process nested dictionaries
            return {
                key: resolve_component_values(sub_value, component_system)
                for key, sub_value in value.items()
            }
    return value  # Return unchanged if not a dict


def update_process_parameters(
    knauer_process: KnauerSystemProcess,
    parameters: dict,
) -> None:
    """
    Update Knauer System Process parameters from file.

    Parameters
    ----------
    knauer_process : KnauerSystemProcess
        The Knauer process instance containing unit operations.
    parameters : dict
        The current parameter values.
    """
    component_system = knauer_process.component_system
    resolved_parameters = resolve_component_values(parameters, component_system)

    for unit in knauer_process.flow_sheet:
        if unit.name not in resolved_parameters:
            continue

        unit_parameters = copy.deepcopy(unit.parameters)
        updated_parameters = update_dict_recursively(
            unit_parameters,
            resolved_parameters[unit.name],
            only_existing_keys=True
        )
        axial_dispersion = updated_parameters.get("axial_dispersion")
        if isinstance(axial_dispersion, dict) and len(axial_dispersion) == 1:
            axial_dispersion = next(iter(axial_dispersion.values()))
            updated_parameters["axial_dispersion"] = axial_dispersion

        if isinstance(unit.binding_model, NoBinding):
            updated_parameters.pop("binding_model", None)

        unit.parameters = updated_parameters

        if unit.name == "column":
            unit.length = column_length
            unit.diameter = column_diameter

    if "column" in knauer_process.flow_sheet and "total_capacity" in parameters:
        if isinstance(knauer_process.flow_sheet.column.binding_model, StericMassAction):
            from e8 import set_total_capacity
            set_total_capacity(knauer_process, parameters["total_capacity"])


def restore_component_values(value: Any, component_system: ComponentSystem) -> Any:
    """
    Recursively restore lists back to dictionaries based on component names.

    Parameters
    ----------
    value : Any
        The value to process. If it's a list of component-based values,
        it will be converted back into a dictionary.
    component_system : ComponentSystem
        The list of components to restore dictionary keys against.

    Returns
    -------
    Any
        The restored value, either unchanged or transformed into a dictionary.
    """
    if isinstance(value, list) and len(value) == len(component_system):
        return {comp.name: val for comp, val in zip(component_system, value)}
    elif isinstance(value, dict):
        return {
            key: restore_component_values(sub_value, component_system)
            for key, sub_value in value.items()
        }
    return value


def update_parameters(
    prior_parameters: dict,
    knauer_process: KnauerSystemProcess,
    prior_branch_name: Optional[str] = None,
) -> dict:
    """
    Update parameters with fitted values.

    Parameters
    ----------
    prior_parameters : dict
        The paramters before the fitting.
    knauer_process : KnauerSystemProcess
        The Knauer system instance containing unit operations.
    prior_branch_name : Optional[str] = None
        Name of the prior parameters branch.

    Returns
    -------
    dict
        The updated parameters
    """
    component_system = knauer_process.component_system
    new_parameters = {
        unit.name: unit.parameters
        for unit in knauer_process.flow_sheet
    }

    restored_parameters = restore_component_values(
        new_parameters, component_system
    )
    updated_parameters = update_dict_recursively(
        prior_parameters,
        restored_parameters,
        only_existing_keys=True
    )
    updated_parameters["prior_branch_name"] = prior_branch_name
    updated_parameters["case"] = knauer_process.name

    return updated_parameters


def tracks_results(func):
    """Tracks results using CADET-RDM."""

    @functools.wraps(func)
    def track_results_wrapper(
        *args,
        commit_message: str,
        debug: bool = False,
        repo_path: os.PathLike = '.',
        **kwargs,
    ) -> tuple[str, Any]:
        project_repo = ProjectRepo(repo_path)

        with project_repo.track_results(
                commit_message,
                debug=debug,
        ) as new_branch_name:
            results = func(*args, **kwargs)

        if not debug and "push":
            project_repo.push()

        return results, new_branch_name

    return track_results_wrapper
