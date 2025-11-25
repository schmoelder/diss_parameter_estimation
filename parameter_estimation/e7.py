"""
# E7: Determine diffusion parameters (Lysozyme).

- Parameters: particle porosity, film diffusion, pore diffusion(, axial dispersion)
- Flow sheet: full system
- Injected component: Lysozyme
- Concentration: 0.2 mM
- Volume: 50e-9 m³
- Eluent: High Salt
- Measurement: UV

Status: done

"""

# %% Imports

from CADETProcess.processModel import ComponentSystem, GeneralRateModel, LumpedRateModelWithPores
from cadetrdm import Options, ProjectRepo, tracks_results

from knauer import PulseInjection, KnauerExperimentalData
from calibration import correct_baseline_and_normalize
from characterization import CharacterizeParticles, optimize
from parameters import (
    component_system_lysozyme,
    knauer_system_options,
    c_salt_high,
    c_lysozyme,
    n_sample_lysozyme,
    time_offset,
    cycle_time_bed,
    flow_rate,
    metrics,
    optimizer_options,
)
from utils import experimental_data_path


# %% Setup methods

solution_path = "tubing_post_column.outlet"
components = ["Lysozyme"]

knauer_system_options = knauer_system_options.copy()

DEFAULT_OPTIONS = Options({
    "include_axial_dispersion": True,
    "include_film_diffusion": False,
    "include_pore_diffusion": False,
    "prior_branch_name": None,
    "commit_message": "E7",
    "debug": False,
    "push": True,
})


def setup_reference() -> None:
    """Set up reference data."""
    file_path = experimental_data_path / "e7.csv"
    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = correct_baseline_and_normalize(
        knauer_data.uv_1, target_area=n_sample_lysozyme
    )
    reference.component_system = ComponentSystem(components)
    reference.update_solution()

    return reference


def setup_process(
    include_pore_diffusion=False,
):
    """Set up process."""
    name = "e7"

    if include_pore_diffusion:
        model_name = name + "_grm"
        knauer_system_options['ColumnModel'] = GeneralRateModel
    else:
        model_name = name + "_lrmp"
        knauer_system_options['ColumnModel'] = LumpedRateModelWithPores

    return PulseInjection(
        model_name,
        component_system_lysozyme,
        knauer_system_options,
        c_buffer_a=[c_salt_high, 0],
        c_sample=[c_salt_high, c_lysozyme],
        cycle_time=cycle_time_bed,
        flow_rate=flow_rate,
    )


def run_optimization(
    results_directory,
    process,
    reference,
    include_axial_dispersion=False,
    include_film_diffusion=False,
    prior_branch_name=None,
    debug=False,
):
    """Run optimization."""
    include_pore_diffusion = "grm" in process.name

    characterization_options = {
        "include_particle_porosity": True,
        "include_film_diffusion": include_film_diffusion,
        "include_pore_diffusion": include_pore_diffusion,
        "include_axial_dispersion": include_axial_dispersion,
        "component_index": 1,
        "name": process.name,
    }
    return optimize(
        results_directory,
        process,
        CharacterizationType=CharacterizeParticles,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        characterization_options=characterization_options,
        optimizer_options=optimizer_options,
        prior_branch_name=prior_branch_name,
    )


# %% Run optimization
@tracks_results
def main(repo:ProjectRepo, options: Options):
    # Setup process
    process = setup_process(options.include_pore_diffusion)

    # Setup reference data
    reference = None if options.use_synthetic_data else setup_reference()

    # Run optimization
    return run_optimization(
        repo.output_path,
        process,
        reference,
        include_film_diffusion=options.include_film_diffusion,
        include_axial_dispersion=options.include_axial_dispersion,
        prior_branch_name=options.prior_branch_name
    )


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.include_axial_dispersion = True
    options.include_film_diffusion = False
    options.include_pore_diffusion = False
    options.debug = True
    options.prior_branch_name = None

    e7_optimization_results, posteriour_branch_name = main(options)
