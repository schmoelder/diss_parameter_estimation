"""
# E9: Determine adsorption parameters (Lysozyme).

- Parameters: adsorption rate, desorption rate, characteristic charge, steric factor
- Flow sheet: full system
- Injected component: Lysozyme
- Concentration: 0.2 mM
- Volume: 50e-9 m³
- Eluent: Gradient Low Salt -> High Salt
- Measurement: UV, Cond

Status: open

"""

# %% Imports

from CADETProcess.processModel import (
    ComponentSystem, StericMassAction, GeneralRateModel, LumpedRateModelWithPores
)

from e0 import coeffcients
from calibration import correct_baseline_and_normalize, apply_polynomial_calibration
from characterization import (
    CharacterizeAdsorptionParameters,
    optimize,
)
from knauer import LWE, KnauerExperimentalData
from parameters import (
    component_system_lysozyme,
    knauer_system_options,
    column_volume,
    c_salt_low,
    c_salt_high,
    c_lysozyme,
    n_sample_lysozyme,
    delta_t_wash,
    delta_t_final_wash,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

solution_path_lysozyme = "tubing_post_column.outlet"
solution_path_salt = "tubing_detectors.outlet"
components = ["Lysozyme"]

gradient_lengths_cv = [4, 8, 12, 16]


def setup_references():
    """Set up reference data."""
    references_lysozyme = []
    references_salt = []
    for gradient in gradient_lengths_cv:
        file_path = experimental_data_path / "e9" / f"{gradient}_cv.csv"

        knauer_data = KnauerExperimentalData(
            file_path=file_path,
            flow_rate=flow_rate,
            time_offset=0,
        )

        # Rescale Lysozyme
        reference_lysozyme = correct_baseline_and_normalize(
            knauer_data.uv_1,
            start_baseline=60,
            target_area=n_sample_lysozyme,
        )
        reference_lysozyme.component_system = ComponentSystem(components)
        references_lysozyme.append(reference_lysozyme)

        # Apply conductivity calibration
        reference_salt = apply_polynomial_calibration(
            knauer_data.conductivity, coeffcients
        )
        reference_salt.component_system = ComponentSystem(["Salt"])
        references_salt.append(reference_salt)

    return references_lysozyme, references_salt


def setup_lwe(
    gradient_length: int,
    include_pore_diffusion: bool = False,
    is_kinetic: bool = True
) -> LWE:
    """Setup individual gradient process."""
    name = "e9"
    system_options = knauer_system_options.copy()
    system_options['BindingModel'] = StericMassAction

    if include_pore_diffusion:
        model_name = f"{name}_grm"
        system_options['ColumnModel'] = GeneralRateModel
    else:
        model_name = f"{name}_lrmp"
        system_options['ColumnModel'] = LumpedRateModelWithPores

    return LWE(
        f"{model_name}_{gradient_length}_cv",
        component_system_lysozyme,
        system_options,
        c_buffer_a=[c_salt_low, 0],
        c_buffer_b=[c_salt_high, 0],
        c_sample=[c_salt_low, c_lysozyme],
        is_kinetic=is_kinetic,
        delta_t_wash=delta_t_wash,
        delta_t_elute=gradient_length*column_volume/flow_rate,
        delta_t_final_wash=delta_t_final_wash,
        flow_rate_wash=flow_rate,
    )


def setup_lwe_processes(
    include_pore_diffusion: bool = False,
    is_kinetic: bool = True,
):
    """Set up all gradient processes."""
    return [
        setup_lwe(gradient, include_pore_diffusion, is_kinetic)
        for gradient in gradient_lengths_cv
    ]


def run_optimization(
    lwe_processes,
    references_lysozyme,
    include_film_diffusion=False,
    start_times=None,
    end_times=None,
    prior_branch_name=None,
    debug=False,
):
    """Run optimization."""
    include_pore_diffusion = "grm" in lwe_processes[0].name
    is_kinetic = lwe_processes[0].flow_sheet.column.binding_model.is_kinetic

    characterization_options = {
        "include_film_diffusion": include_film_diffusion,
        "include_pore_diffusion": include_pore_diffusion,
        "is_kinetic": is_kinetic,
        "component_index": 1,
        "name": lwe_processes[0].name,
    }
    commit_message = "E9"
    if prior_branch_name is None:
        commit_message += "_synthetic"
    elif prior_branch_name == "parameters_lukas":
        commit_message += "_lukas"

    return optimize(
        lwe_processes,
        CharacterizationType=CharacterizeAdsorptionParameters,
        solution_path=solution_path_lysozyme,
        components=components,
        references=references_lysozyme,
        metrics=metrics,
        start_times=start_times,
        end_times=end_times,
        characterization_options=characterization_options,
        optimizer_options=optimizer_options,
        prior_branch_name=prior_branch_name,
        commit_message=commit_message,
        debug=debug,
    )


# %% Run optimization

def main(
    prior_branch_name=None,
    include_film_diffusion=False,
    include_pore_diffusion=False,
    is_kinetic=True,
    start_times=None,
    end_times=None,
    debug=False,
):
    # Setup reference data
    if prior_branch_name is None:
        references_lysozyme = None
    else:
        references_lysozyme, references_salt = setup_references()

    # Setup process
    lwe_processes = setup_lwe_processes(include_pore_diffusion, is_kinetic)

    # Run optimization
    return run_optimization(
        lwe_processes,
        references_lysozyme,
        include_film_diffusion,
        start_times,
        end_times,
        prior_branch_name,
        debug=debug,
    )


if __name__ == "__main__":
    include_film_diffusion = False
    include_pore_diffusion = False
    is_kinetic = False

    debug = False
    prior_branch_name = None
    start_times = None
    end_times = None

    e9_optimization_results, posteriour_branch_name = main(
        prior_branch_name,
        include_film_diffusion,
        include_pore_diffusion,
        is_kinetic,
        start_times,
        end_times,
        debug,
    )
