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
from CADETProcess.reference import ReferenceIO
from cadetrdm import Options, ProjectRepo, tracks_results

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
from utils import experimental_data_path, load_parameters, update_process_parameters


# %% Setup methods

solution_path_lysozyme = "tubing_post_column.outlet"
solution_path_salt = "tubing_detectors.outlet"
components = ["Lysozyme"]

DEFAULT_OPTIONS = Options({
    "pH": 4.0,
    "time_offset": 0.0,
    "use_peak_times": False,
    "include_film_diffusion": False,
    "include_pore_diffusion": False,
    "is_kinetic": False,
    "prior_branch_name": None,
    "commit_message": "E9",
    "debug": False,
    "push": True,
})

gradient_lengths_cv = [4, 8, 12, 16]

peaks = {
    4.0: {
        4: [50*60, 58*60],
        8: [68*60, 78*60],
        12: [85*60, 98*60],
        16: [100*60, 116*60],
    },
    4.25: {
        4: [48*60, 56*60],
        8: [64*60, 74*60],
        12: [80*60, 94*60],
        16: [95*60, 108*60],
    },
    4.5: {
        4: [46*60, 56*60],
        8: [61*60, 72*60],
        12: [75*60, 87*60],
        16: [90*60, 103*60],
    },
    4.75: {
        4: [45*60, 54*60],
        8: [58*60, 69*60],
        12: [71*60, 83*60],
        16: [85*60, 98*60],
    },
    5.0: {
        4: [44*60, 53*60],
        8: [55*60, 68*60],
        12: [70*60, 83*60],
        16: [80*60, 97*60],
    },
}


def get_peak_times(pH, time_offset=0.0):
    start_times = [peak[0]-time_offset for peak in peaks[pH].values()]
    end_times = [peak[1]-time_offset for peak in peaks[pH].values()]

    return start_times, end_times


def setup_references(pH, time_offset=0.0):
    """Set up reference data."""
    references_lysozyme: list[ReferenceIO] = []
    references_salt: list[ReferenceIO] = []
    start_times, end_times = get_peak_times(pH, time_offset)

    for gradient, start, end in zip(gradient_lengths_cv, start_times, end_times):
        file_path = experimental_data_path / "e9" / f"Lysozyme_pH_{pH}" / f"{gradient}_cv.csv"

        knauer_data = KnauerExperimentalData(
            file_path=file_path,
            flow_rate=flow_rate,
            time_offset=time_offset,
        )

        # Rescale Lysozyme
        reference_lysozyme = correct_baseline_and_normalize(
            knauer_data.uv_1,
            start_baseline=max(60, time_offset),  # Account for startup signal noise
            start_normalization=start,
            end_normalization=end,
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
    results_directory,
    lwe_processes,
    references_lysozyme,
    start_times,
    end_times,
    include_film_diffusion=False,
    prior_branch_name=None,
):
    """Run optimization."""
    include_pore_diffusion = "grm" in lwe_processes[0].name
    is_kinetic = lwe_processes[0].flow_sheet.column.binding_model.is_kinetic

    characterization_options = {
        "include_film_diffusion": include_film_diffusion,
        "include_pore_diffusion": include_pore_diffusion,
        "is_kinetic": is_kinetic,
        "component_index": 1,
        "name": lwe_processes[0].name[0:-5],
    }

    return optimize(
        results_directory,
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
    )


# %% Yamamoto

from CADETProcess.tools.yamamoto import (
    GradientExperiment, plot_experiments, fit_parameters, YamamotoResults
)

def setup_yamamoto_experiments(pH, column_volume, detector_offset):
    references_lysozyme, _ = setup_references(pH=pH)
    _, references_salt = setup_references(pH=pH, time_offset=detector_offset)

    def create_experiment(ref_salt, ref_protein, gradient_length_cv):
        time = ref_salt.time

        c_salt = ref_salt.solution
        c_protein = ref_protein.solution[0:len(time)]

        experiment = GradientExperiment(
            time,
            c_salt,
            c_protein,
            column_volume*gradient_length_cv,
            c_salt_start=20,
            c_salt_end=1020,
        )

        return experiment

    experiments = [
        create_experiment(ref_salt, ref_protein, gradient_length_cv)
        for ref_salt, ref_protein, gradient_length_cv
        in zip(references_salt, references_lysozyme, gradient_lengths_cv)
    ]

    return experiments


def yamamoto(
    pH,
    prior_branch_name=None,
) -> YamamotoResults:
    process = setup_lwe_processes()[0]
    prior_parameters = load_parameters(prior_branch_name)
    update_process_parameters(process, prior_parameters)
    column = process.flow_sheet.column
    tubing_detector = process.flow_sheet.tubing_detectors
    detector_offset = tubing_detector.volume / flow_rate

    experiments = setup_yamamoto_experiments(
        pH, column.volume, detector_offset
    )
    plot_experiments(experiments)

    yamamoto_results = fit_parameters(experiments, column)
    yamamoto_results.plot()

    return yamamoto_results


# %% Run optimization
@tracks_results
def main(repo:ProjectRepo, options: Options):
    # Setup process
    lwe_processes = setup_lwe_processes(
        options.include_pore_diffusion,
        options.is_kinetic,
    )

    # Setup reference data
    references_lysozyme = None
    if options.prior_branch_name:
        references_lysozyme, references_salt = setup_references(
            options.pH,
            options.time_offset,
        )

    start_times = None
    end_times = None
    if options.use_peak_times:
        start_times, end_times = get_peak_times(
            options.pH,
            options.time_offset,
        )

    # Run optimization
    return run_optimization(
        repo.output_path,
        lwe_processes,
        references_lysozyme,
        start_times,
        end_times,
        include_film_diffusion=options.include_film_diffusion,
        prior_branch_name=options.prior_branch_name
    )


if __name__ == "__main__":
    options = DEFAULT_OPTIONS.copy()
    options.pH = 4.0
    options.time_offset = 0.0
    options.use_peak_times = False
    options.include_film_diffusion = False
    options.include_pore_diffusion = False
    options.debug = True
    options.prior_branch_name = None

    e9_optimization_results, posteriour_branch_name = main(options)
