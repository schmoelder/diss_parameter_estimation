from dataclasses import dataclass
import os
from typing import Any, Literal, Optional, Type

import numpy as np

from cadetrdm import ProjectRepo
from CADETProcess import settings
from CADETProcess.dataStructure import set_nested_value
from CADETProcess.comparison import Comparator
from CADETProcess.optimization import (
    OptimizationProblem, OptimizerBase, U_NSGA3, OptimizationResults
)
from CADETProcess.reference import ReferenceIO
from CADETProcess.simulator import Cadet

from knauer import KnauerSystemProcess, LWE
from utils import (
    load_parameters,
    update_process_parameters,
    update_parameters,
    save_parameters,
    tracks_results,
)


@dataclass
class ReferenceConfig:
    """Configuration for a reference used in the Comparator."""

    reference: ReferenceIO
    solution_path: str
    metrics: list[Literal["Shape", "ShapeFront", "NRMSE"]]
    components: Optional[list[str]] = None,
    process: Optional[KnauerSystemProcess] = None
    start: Optional[float] = None
    end: Optional[float] = None


class CharacterizeBase(OptimizationProblem):
    """Base class for characterizing sections of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
        variables: list[dict[str, Any]],
    ) -> None:

        # Ensure processes is always a list
        if not isinstance(processes, list):
            processes = [processes]

        super().__init__(name=name)

        for process in processes:
            self.add_evaluation_object(process)

        # Add variables dynamically
        for var in variables:
            self.add_variable(**var)

        simulator = Cadet()
        self.add_evaluator(simulator)

        comparators = setup_comparators(processes, reference_configs)

        for process in processes:
            self.add_objective(
                comparators[process.name],
                n_objectives=comparators[process.name].n_metrics,
                requires=[simulator],
                evaluation_objects=process,
            )

        def callback(
            simulation_results,
            individual,
            evaluation_object,
            callbacks_dir="./"
        ):
            comparators[evaluation_object.name].plot_comparison(
                simulation_results,
                file_name=f"{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png",
                show=False
            )

        self.add_callback(
            callback, requires=[simulator], frequency=1
        )


class CharacterizeTubing(CharacterizeBase):
    """Characterize tubing of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        tubing: str,
        reference_configs: ReferenceConfig | list[ReferenceConfig],
    ) -> None:

        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=[
                {
                    "name": f"{tubing}_length",
                    "parameter_path": f"flow_sheet.{tubing}.length",
                    "lb": 1e-2, "ub": 2.0,
                    "transform": "auto"
                },
                {
                    "name": f"{tubing}_axial_dispersion",
                    "parameter_path": f"flow_sheet.{tubing}.axial_dispersion",
                    "lb": 1e-9, "ub": 1e-2,
                    "transform": "auto"
                }
            ],
        )


class CharacterizePreInjection(CharacterizeBase):
    """Characterize pre-injection section of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
    ) -> None:

        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=[
                {
                    "name": "tubing_pre_injection_length",
                    "parameter_path": "flow_sheet.tubing_pre_injection.length",
                    "lb": 1e-2, "ub": 2.0,
                    "transform": "auto"
                },
                {
                    "name": "mixer_volume",
                    "parameter_path": "flow_sheet.mixer.init_liquid_volume",
                    "lb": 1e-8, "ub": 1e-5,
                    "transform": "auto"
                }
            ],
        )


class CharacterizeBed(CharacterizeBase):
    """Characterize packed bed of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
    ) -> None:
        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=[
                {
                    "name": "bed_porosity",
                    "parameter_path": "flow_sheet.column.bed_porosity",
                    "lb": 0.2, "ub": 0.6,
                    "transform": "auto",
                },
                {
                    "name": "axial_dispersion",
                    "parameter_path": "flow_sheet.column.axial_dispersion",
                    "lb": 1e-11, "ub": 1e-3,
                    "transform": "auto",
                }
            ],
        )


class CharacterizeParticles(CharacterizeBase):
    """Characterize particles of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
        include_axial_dispersion: Optional[bool] = False,
        include_particle_porosity: Optional[bool] = False,
        include_film_diffusion: Optional[bool] = False,
        include_pore_diffusion: Optional[bool] = False,
        component_index: Optional[int] = 0,
    ) -> None:
        variables = []
        if include_axial_dispersion:
            variables.append({
                "name": "axial_dispersion",
                "parameter_path": "flow_sheet.column.axial_dispersion",
                "lb": 1e-9, "ub": 1e-5,
                "indices": [component_index],
                "transform": "auto"
            })
        if include_particle_porosity:
            variables.append({
                "name": "particle_porosity",
                "parameter_path": "flow_sheet.column.particle_porosity",
                "lb": 0.6, "ub": 0.9,
                "transform": "auto"
            })
        if include_film_diffusion:
            variables.append({
                "name": "film_diffusion",
                "parameter_path": "flow_sheet.column.film_diffusion",
                "lb": 1e-7, "ub": 1e-3,
                "indices": [component_index],
                "transform": "auto"
            })
        if include_pore_diffusion:
            variables.append({
                "name": "pore_diffusion",
                "parameter_path": "flow_sheet.column.pore_diffusion",
                "lb": 1e-12, "ub": 1e-6,
                "indices": [component_index],
                "transform": "auto"
            })

        if len(variables) == 0:
            raise ValueError("Must specify at least one variable.")

        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=variables
        )


class CharacterizeCapacity(CharacterizeBase):
    """Characterize capacity of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: KnauerSystemProcess | list[KnauerSystemProcess],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
    ) -> None:
        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=[
                {
                    "name": "capacity",
                    "parameter_path": "flow_sheet.column.binding_model.capacity",
                    "lb": 1e-2, "ub": 2.0,
                    "transform": "auto"
                },
            ],
        )


class CharacterizeAdsorptionParameters(CharacterizeBase):
    """Characterize adsorption parameters of KnauerSystem."""

    def __init__(
        self,
        name: str,
        processes: LWE | list[LWE],
        reference_configs: ReferenceConfig | list[ReferenceConfig],
        is_kinetic: bool = False,
        include_film_diffusion: Optional[bool] = False,
        include_pore_diffusion: Optional[bool] = False,
        component_index: Optional[int] = 0,
    ) -> None:
        column = processes[0].flow_sheet.column
        binding_model = column.binding_model
        lambda_ = binding_model.capacity

        for process in processes:
            process.flow_sheet.column.binding_model.is_kinetic = is_kinetic

            if is_kinetic:
                # Set reference concentrations
                process.flow_sheet.column.binding_model.reference_liquid_phase_conc = lambda_
                process.flow_sheet.column.binding_model.reference_solid_phase_conc = lambda_

            if not is_kinetic:
                process.flow_sheet.column.binding_model.desorption_rate = 1

        variables = []
        variables.append({
            "name": "characteristic_charge",
            "parameter_path": "flow_sheet.column.binding_model.characteristic_charge",
            "lb": 1, "ub": 10,
            "indices": [component_index],
            "transform": "auto",
        })

        k_eq_lb = 1e-3
        k_eq_ub = 1

        if not is_kinetic:
            variables.append({
                "parameter_path": "flow_sheet.column.binding_model.adsorption_rate",
                "lb": k_eq_lb, "ub": k_eq_ub,
                "name": "adsorption_rate",
                "indices": [component_index],
                "transform": "auto",
            })
        else:
            variables.append({
                "parameter_path": "flow_sheet.column.binding_model.adsorption_rate",
                "name": "adsorption_rate",
                "indices": [component_index],
            })

            variables.append({
                "parameter_path": "flow_sheet.column.binding_model.desorption_rate",
                "name": "desorption_rate",
                "indices": [1],
            })

            variables.append({
                "name": "equilibrium_constant",
                "lb": k_eq_lb, "ub": k_eq_ub,
                "transform": "auto",
                "evaluation_objects": None,
            })

            variables.append({
                "name": "kinetic_constant",
                "lb": 1e-9, "ub": 1,
                "transform": "auto",
                "evaluation_objects": None,
            })

        if include_film_diffusion:
            variables.append({
                "name": "film_diffusion",
                "parameter_path": "flow_sheet.column.film_diffusion",
                "lb": 1e-7, "ub": 1e-4,
                "indices": [component_index],
                "transform": "auto"
            })
        if include_pore_diffusion:
            variables.append({
                "name": "pore_diffusion",
                "parameter_path": "flow_sheet.column.pore_diffusion",
                "lb": 1e-12, "ub": 1e-6,
                "indices": [component_index],
                "transform": "auto"
            })

        if len(variables) == 0:
            raise ValueError("Must specify at least one variable.")

        super().__init__(
            name=name,
            processes=processes,
            reference_configs=reference_configs,
            variables=variables,
        )

        if is_kinetic:
            self.add_variable_dependency(
                dependent_variable="desorption_rate",
                independent_variables=["kinetic_constant"],
                transform=lambda k_kin: 1 / k_kin
            )

            self.add_variable_dependency(
                dependent_variable="adsorption_rate",
                independent_variables=["kinetic_constant", "equilibrium_constant"],
                transform=lambda k_kin, k_eq: k_eq / k_kin
            )

        def sort_and_filter(pareto_population):
            n_max = 1
            f_sum = np.sum(pareto_population.f_minimized, axis=1)
            p = f_sum.argsort()
            x = pareto_population.x[p]
            if len(x) > n_max:
                x = x[0:n_max]

            return x

        self.add_multi_criteria_decision_function(sort_and_filter)


# %% Configure processes/references/comparators

def update_processes(
    knauer_processes: KnauerSystemProcess | list[KnauerSystemProcess],
    prior_parameters: dict,
) -> list[KnauerSystemProcess]:
    """
    Update system parameters for the given Knauer system processes.

    Parameters
    ----------
    knauer_processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A list of Knauer system processes to be characterized.
    prior_parameters : dict
        The prior process parameters.

    Returns
    -------
    list[KnauerSystemProcess]
        The updated Knauer system processes.
    """
    if not isinstance(knauer_processes, list):
        knauer_processes = [knauer_processes]

    for process in knauer_processes:
        update_process_parameters(process, prior_parameters)
    return knauer_processes


def setup_reference_configs(
    knauer_processes: KnauerSystemProcess | list[KnauerSystemProcess],
    solution_path: str,
    use_synthetic_data: Optional[bool] = False,
    components: Optional[list[str]] = None,
    references: Optional[list[ReferenceIO]] = None,
    metrics: Optional[list[str]] = None,
    start_times: Optional[float | list[float]] = None,
    end_times: Optional[float | list[float]] = None,
) -> list[ReferenceConfig]:
    """
    Set up reference configurations for Knauer system processes.

    Parameters
    ----------
    knauer_processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A single or list of Knauer system processes to configure.
    solution_path : str
        The file path where the solution data will be saved.
    use_synthetic_data : Optional[bool], optional
        If True, generate synthetic data for references. Default is False.
    components : Optional[list[str]], optional
        Components used for parameter estimation. Default is None.
    references : Optional[list[ReferenceIO]], optional
        List of reference data. If None, synthetic data is used. Default is None.
    metrics : Optional[list[str]], optional
        List of metrics to be used for evaluation. Default is None.
    start_times : Optional[list[float]], optional
        Start times of references to consider for comparison. Default is None.
    end_times : Optional[list[float]], optional
        End times of references to consider for comparison. Default is None.

    Returns
    -------
    list[ReferenceConfig]
        A list of configured reference configurations.
    """
    if not isinstance(knauer_processes, list):
        knauer_processes = [knauer_processes]

    # Generate or load reference data
    if use_synthetic_data:
        references = [
            process.generate_synthetic_data(
                solution_path=solution_path,
                components=components,
            )
            for process in knauer_processes
        ]
    else:
        if not isinstance(references, list):
            references = [references]

    if start_times is None:
        start_times = [None for _ in range(len(references))]
    if not isinstance(start_times, list):
        start_times = [start_times]
    if end_times is None:
        end_times = [None for _ in range(len(references))]
    if not isinstance(end_times, list):
        end_times = [end_times]

    # Configure the references for comparison
    return [
        ReferenceConfig(
            reference=reference,
            solution_path=solution_path,
            metrics=metrics,
            components=components,
            process=process.name,
            start=start_time,
            end=end_time,
        )
        for process, reference, start_time, end_time
        in zip(knauer_processes, references, start_times, end_times)
    ]


def setup_comparators(
    processes: KnauerSystemProcess | list[KnauerSystemProcess],
    reference_configs: ReferenceConfig | list[ReferenceConfig]
) -> list[Comparator]:
    """
    Set up comparators for Knauer system processes based on reference configurations.

    Parameters
    ----------
    processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A single or list of Knauer system processes to configure comparators for.
    reference_configs : ReferenceConfig | list[ReferenceConfig]
        A single or list of reference configurations to use for comparison.

    Returns
    -------
    list[Comparator]
        A list of configured comparators for each process.
    """
    # Ensure processes and reference_configs are always lists
    if not isinstance(reference_configs, list):
        reference_configs = [reference_configs]

    if not isinstance(processes, list):
        processes = [processes]

    comparators = {
        process.name: Comparator(process.name) for process in processes
    }

    for config in reference_configs:
        if config.process is None:
            config.process = processes[0].name

        comparator = comparators[config.process]

        comparator.add_reference(config.reference)
        for metric in config.metrics:
            comparator.add_difference_metric(
                metric,
                config.reference,
                config.solution_path,
                components=config.components,
                start=config.start,
                end=config.end,
            )

    return comparators


# %% Configure and run Optimizer

def setup_optimizer(
    optimization_problem: OptimizationProblem,
    optimizer_options: dict,
) -> OptimizerBase:
    """
    Set up an optimizer for a given optimization problem.

    Parameters
    ----------
    optimization_problem : OptimizationProblem
        The optimization problem to solve.
    optimizer_options : dict
        Dictionary containing options for the optimizer. Must include the key
        "optimizer" specifying the type of optimizer to use.

    Returns
    -------
    OptimizerBase
        Configured optimizer instance.

    Raises
    ------
    ValueError
        If the specified optimizer is unknown.
    """
    optimizer_options = optimizer_options or {}

    if optimizer_options["optimizer"] == "U_NSGA3":
        optimizer = U_NSGA3()
        default_options = {
            "n_cores": -4,
            "pop_size": optimization_problem.n_variables * 32,
            "n_max_gen": 16,
        }
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_options.optimizer}")

    default_options.update(optimizer_options)
    for key, value in default_options.items():
        setattr(optimizer, key, value)

    return optimizer


def setup_characterization(
    results_directory: os.PathLike,
    knauer_processes: KnauerSystemProcess | list[KnauerSystemProcess],
    CharacterizationType: Type[CharacterizeBase],
    reference_configs: list,
    characterization_options: Optional[dict] = None,
) -> CharacterizeBase:
    """
    Initialize the characterization object.

    Parameters
    ----------
    knauer_processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A list of Knauer system processes to be characterized.
    CharacterizationType : Type[CharacterizeBase]
        The characterization configuration class to be used.
    reference_configs : list
        Reference configurations for the characterization.
    characterization_options : Optional[dict], optional
        Additional options for the characterization process. The default is None.

    Returns
    -------
    CharacterizeBase
        The initialized characterization object.
    """
    if not isinstance(knauer_processes, list):
        knauer_processes = [knauer_processes]

    characterization_options = characterization_options or {}
    name = characterization_options.pop("name", knauer_processes[0].name)
    settings.working_directory = results_directory
    characterization = CharacterizationType(
        name=name,
        processes=knauer_processes,
        reference_configs=reference_configs,
        **characterization_options,
    )
    return characterization


def setup_optimization_problem(
    results_directory: os.PathLike,
    knauer_processes: KnauerSystemProcess | list[KnauerSystemProcess],
    CharacterizationType: Type[CharacterizeBase],
    solution_path: str,
    components: Optional[list[str]] = None,
    references: Optional[list[ReferenceIO]] = None,
    metrics: Optional[list[str]] = None,
    start_times: Optional[list[float]] = None,
    end_times: Optional[list[float]] = None,
    characterization_options: Optional[dict] = None,
    parameters_overwrite: Optional[dict] = None,
    prior_branch_name: Optional[str] = None,
) -> tuple[CharacterizeBase, list[KnauerSystemProcess], dict]:
    """
    Set up the complete characterization object including processes and reference configurations.

    Parameters
    ----------
    results_directory: os.PathLike
        Path to results directory.
    knauer_processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A list of Knauer system processes to be characterized.
    CharacterizationType : Type[CharacterizeBase]
        The characterization configuration class to be used.
    solution_path : str
        The file path where the solution data will be saved.
    components : Optional[list[str]]
        Components used for parameter estimation. The default is None.
    references : Optional[list[ReferenceIO]]
        If None, synthetic data is used.
    metrics : Optional[list[str]], optional
        List of metrics to be used for evaluation. The default is None.
    start_times : Optional[list[float]]
        Start times of references to consider for comparison. The default is None.
    end_times : Optional[list[float]]
        End times of references to consider for comparison. The default is None.
    characterization_options : Optional[dict], optional
        Additional options for the characterization process. The default is None.
    parameters_overwrite : Optional[dict]
        Parameters to be overwritten.
    prior_branch_name : Optional[str]
        Name of the output repository branch to be used to load prior parameters.
        If None are provided, synthetic data is used.

    Returns
    -------
    tuple[CharacterizeBase, list[KnauerSystemProcess], dict]
        The initialized characterization object, updated processes, and prior parameters.
    """
    prior_parameters = load_parameters(prior_branch_name)
    if parameters_overwrite:
        for key, value in parameters_overwrite.items():
            set_nested_value(
                prior_parameters,
                key,
                value,
            )

    knauer_processes = update_processes(
        knauer_processes, prior_parameters
    )
    use_synthetic_data = references is None
    reference_configs = setup_reference_configs(
        knauer_processes,
        solution_path,
        use_synthetic_data,
        components,
        references,
        metrics,
        start_times,
        end_times,
    )
    characterization = setup_characterization(
        results_directory,
        knauer_processes,
        CharacterizationType,
        reference_configs,
        characterization_options,
    )
    return characterization, knauer_processes, prior_parameters


def optimize(
    results_directory: os.PathLike,
    knauer_processes: KnauerSystemProcess | list[KnauerSystemProcess],
    CharacterizationType: Type[CharacterizeBase],
    solution_path: str,
    components: Optional[list[str]] = None,
    references: Optional[list[ReferenceIO]] = None,
    metrics: Optional[list[str]] = None,
    start_times: Optional[list[float]] = None,
    end_times: Optional[list[float]] = None,
    characterization_options: Optional[dict] = None,
    parameters_overwrite: Optional[dict] = None,
    optimizer_options: Optional[dict] = None,
    prior_branch_name: Optional[str] = None,
) -> OptimizationResults:
    """
    Run optimization to characterize Knauer system parameters.

    Parameters
    ----------
    results_directory: os.PathLike,
        Path to results directory.
    knauer_processes : KnauerSystemProcess | list[KnauerSystemProcess]
        A list of Knauer system processes to be characterized.
    CharacterizationType : Type[CharacterizeBase]
        The characterization configuration class to be used.
    solution_path : str
        The file path where the solution data will be saved.
    components : Optional[list[str]]
        Components used for parameter estimation. The default is None.
    references : Optional[list[ReferenceIO]]
        If None, synthetic data is used.
    metrics : Optional[list[str]], optional
        List of metrics to be used for evaluation. The default is None.
    start_times : Optional[list[float]]
        Start times of references to consider for comparison. The default is None.
    end_times : Optional[list[float]]
        End times of references to consider for comparison. The default is None.
    characterization_options : Optional[dict], optional
        Additional options for the characterization process. The default is None.
    parameters_overwrite : Optional[dict]
        Parameters to be overwritten.
    optimizer_options : Optional[dict]
        Additional options for the optimization process. The default is None.
    prior_branch_name : Optional[str]
        Name of the output repository branch to be used to load prior parameters.
        If None are provided, synthetic data is used.

    Returns
    -------
    OptimizationResults
        The optimization results containing the estimated parameters.
    """
    characterization, knauer_processes, prior_parameters = setup_optimization_problem(
        results_directory,
        knauer_processes,
        CharacterizationType,
        solution_path,
        components,
        references,
        metrics,
        start_times,
        end_times,
        characterization_options,
        parameters_overwrite,
        prior_branch_name,
    )
    optimizer = setup_optimizer(
        characterization,
        optimizer_options,
    )
    optimization_results = optimizer.optimize(characterization)

    characterization.set_variables(optimization_results.meta_front.x[0])

    updated_parameters = update_parameters(
        prior_parameters, knauer_processes[0], prior_branch_name
    )
    save_parameters(updated_parameters, references is None)

    return optimization_results
