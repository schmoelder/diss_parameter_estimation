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

from knauer import PulseInjection, KnauerExperimentalData
from calibration import normalize_area
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


def setup_reference():
    """Set up reference data."""
    file_path = experimental_data_path / "e6.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
        time_offset=time_offset,
    )
    reference = normalize_area(
        knauer_data.uv_1,
        n_sample_acetone,
        start=start_peak,
        end=end_peak,
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
    process,
    reference,
    include_axial_dispersion=True,
    prior_branch_name=None,
    debug=False,
):
    """Run optimization."""
    characterization_options = {
        "include_particle_porosity": True,
        "include_film_diffusion": False,
        "include_pore_diffusion": False,
        "include_axial_dispersion": include_axial_dispersion,
        "component_index": 0,
    }

    commit_message = "E6"
    if prior_branch_name is None:
        commit_message += "_synthetic"
    elif prior_branch_name == "parameters_lukas":
        commit_message += "_lukas"

    return optimize(
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
        commit_message=commit_message,
        debug=debug,
    )

# %% Run optimization

def main(prior_branch_name=None, include_axial_dispersion=True, debug=False):
    # Setup process
    process = setup_process()

    # Setup reference data
    reference = None if prior_branch_name is None else setup_reference()

    # Run optimization
    return run_optimization(
        process,
        reference,
        include_axial_dispersion,
        prior_branch_name, debug
    )


if __name__ == "__main__":
    include_axial_dispersion = True
    debug = False
    prior_branch_name = None

    e6_optimization_results, posteriour_branch_name = main(
        prior_branch_name, include_axial_dispersion, debug
    )
