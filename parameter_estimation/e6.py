"""
# E6: Characterize particles.

- Parameters: particle porosity / total porosity, axial dispersion
- Flow sheet: full system
- Injected component: Acetone
- Concentration: 171.2 mM (1 %)
- Volume: 50e-9 m³
- Eluent: Low Salt
- Measurement: UV

Status: done

"""

# %% Imports

from CADETProcess.processModel import LumpedRateModelWithPores
from cadetrdm import Options, ProjectRepo, tracks_results

from knauer import PulseInjection, KnauerExperimentalData
from calibration import correct_baseline_and_normalize
from characterization import (
    CharacterizeParticles,
    optimize,
)
from parameters import (
    component_system_acetone,
    knauer_system_options,
    c_acetone,
    n_sample_acetone,
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

start_peak = 7*60
end_peak = 12*60

DEFAULT_OPTIONS = Options({
    "include_axial_dispersion": True,
    "prior_branch_name": None,
    "commit_message": "E6",
    "debug": False,
    "push": True,
})


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e6.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = correct_baseline_and_normalize(
        knauer_data.uv_1,
        target_area=n_sample_acetone,
        start_normalization=start_peak,
        end_normalization=end_peak,
    )

    return reference


def setup_process():
    """Set up process."""
    return PulseInjection(
        "e6",
        component_system_acetone,
        knauer_system_options,
        c_buffer_a=[0],
        c_sample=[c_acetone],
        cycle_time=cycle_time_bed,
        flow_rate=flow_rate,
    )


def run_optimization(
    results_directory,
    process,
    reference,
    include_axial_dispersion=True,
    prior_branch_name=None,
):
    """Run optimization."""
    characterization_options = {
        "include_particle_porosity": True,
        "include_film_diffusion": False,
        "include_pore_diffusion": False,
        "include_axial_dispersion": include_axial_dispersion,
        "component_index": 0,
    }

    return optimize(
        results_directory,
        process,
        CharacterizationType=CharacterizeParticles,
        solution_path=solution_path,
        components=components,
        references=reference,
        metrics=metrics,
        start_times=start_peak,
        end_times=end_peak,
        characterization_options=characterization_options,
        optimizer_options=optimizer_options,
        prior_branch_name=prior_branch_name,
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
        include_axial_dispersion=options.include_axial_dispersion,
        prior_branch_name=options.prior_branch_name
    )


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.include_axial_dispersion = True
    options.debug = True
    options.prior_branch_name = None

    e6_optimization_results, posteriour_branch_name = main(options)
