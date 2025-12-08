import os
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from CADETProcess import plotting
from CADETProcess.simulator import Cadet
from CADETProcess.simulationResults import SimulationResults
from CADETProcess.solution import SolutionIO, slice_solution
from CADETProcess.dataStructure import get_nested_value

from knauer import KnauerSystemProcess
from utils import update_process_parameters, load_parameters_from_previous_run


simulator = Cadet()


# %% Helper methods

def setup_figure(
    axes: str | list[str],
    style: Optional[Literal["single_column", "1.5_column", "double_column"]] = "1.5_column",
) -> tuple[plt.Figure, plt.Axes, ...]:
    fig, ax = plotting.setup_figure(style=style)

    ax.set_xlabel(r'$\text{time}~/~\text{min}$')

    if isinstance(axes, str):
        axes = [axes]

    return_axes = []

    for i, ax_i in enumerate(axes):
        if i == 0:
            ax_new = ax
        else:
            ax_new = ax.twinx()
            ax_new.spines.right.set_position(("axes", 1 + (i-1)*0.2))

        ax_new.set_ylabel(fr'$c_{{\text{{{ax_i}}}}}~/~\text{{mM}}$')

        return_axes.append(ax_new)

    fig.tight_layout()

    # Unpack the list of axes before returning
    return (fig, *return_axes)


def get_all_twin_handles_labels(ax: plt.Axes) -> tuple[list, list]:
    """
    Return handles and labels from an axes and all of its twins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The reference axes.

    Returns
    -------
    handles : list
        All line/patch artists from `ax` and its twin axes.
    labels : list of str
        Corresponding legend labels.

    Notes
    -----
    Traverses the shared-axes group to find all twins, including
    those created via `twinx()` and `twiny()`.
    """
    # Matplotlib groups axes that share x or y; twins belong to these groups.
    axs = set(ax.get_shared_x_axes().get_siblings(ax))

    handles = []
    labels = []

    for a in axs:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    return handles, labels


def update_parameters_and_simulate(
    process: KnauerSystemProcess,
    parameters: Optional[dict] = None,
) -> SimulationResults:
    update_process_parameters(process, parameters)
    simulation_results = simulator.simulate(process)

    return simulation_results


def extract_solution(
    simulation_results: SimulationResults,
    solution_path: str,
    components: Optional[str | list[str]] = None,
) -> SolutionIO:

    solution = get_nested_value(simulation_results.solution, solution_path)

    if components is not None:
        solution = slice_solution(solution, components)

    return solution

def simulate_and_get_solution(
    process: KnauerSystemProcess,
    solution_path: str,
    parameters: Optional[dict] = None,
    components: Optional[str | list[str]] = None,
) -> SolutionIO:

    simulation_results = update_parameters_and_simulate(process, parameters)
    solution = extract_solution(simulation_results, solution_path, components)

    return solution


def plot_comparison(ax, solution, reference, color, label=None):
    ax.plot(solution.time/60, solution.solution, color=color, linestyle="-", label=label)
    ax.plot(reference.time/60, reference.solution, color="k", linestyle=":")


# %% Comparison without column

def plot_comparison_without_column(parameters):
    with plotting.mpl_style_context():
        fig_system, ax_system_acetone, ax_system_salt = setup_figure(axes=["Acetone", "Salt"])

        from e1 import (
            setup_process as setup_process_e1,
            setup_reference as setup_reference_e1,
            solution_path as solution_path_e1,
        )
        process_e1 = setup_process_e1()
        reference_e1 = setup_reference_e1()
        solution_e1 = simulate_and_get_solution(process_e1, solution_path_e1, parameters)
        plot_comparison(ax_system_acetone, solution_e1, reference_e1, "darkblue", "E1")

        from e2 import (
            setup_process as setup_process_e2,
            setup_reference as setup_reference_e2,
            solution_path as solution_path_e2,
        )
        process_e2 = setup_process_e2()
        reference_e2 = setup_reference_e2()
        solution_e2 = simulate_and_get_solution(process_e2, solution_path_e2, parameters)

        plot_comparison(
            ax_system_acetone, solution_e2, reference_e2, "darkred", "E2")

        from e3 import (
            setup_process as setup_process_e3,
            setup_reference as setup_reference_e3,
            solution_path as solution_path_e3,
        )
        process_e3 = setup_process_e3()
        reference_e3 = setup_reference_e3()
        solution_e3 = simulate_and_get_solution(process_e3, solution_path_e3, parameters)

        plot_comparison(ax_system_salt, solution_e3, reference_e3, "darkorange", "E3")

        from e4 import (
            setup_process as setup_process_e4,
            setup_reference as setup_reference_e4,
            solution_path as solution_path_e4,
        )
        process_e4 = setup_process_e4()
        reference_e4 = setup_reference_e4()
        solution_e4 = simulate_and_get_solution(process_e4, solution_path_e4, parameters)

        plot_comparison(ax_system_salt, solution_e4, reference_e4, "darkgreen", "E4")

        ax_system_acetone.set_xlim(0, 6)
        handles, labels = get_all_twin_handles_labels(ax_system_acetone)
        ax_system_acetone.legend(handles, labels, loc="center right")
        fig_system.tight_layout()

    return fig_system, ax_system_acetone, ax_system_salt


# %% Characterization with column

def plot_comparison_with_column(
    parameters_e5,
    parameters_e6,
    parameters_e7,
):
    with plotting.mpl_style_context():
        fig_column, ax_column_bd, ax_column_acetone, ax_column_lysozyme = setup_figure(
            axes=["Blue Dextran", "Acetone", "Lysozyme"],
        )
        from e5 import (
            setup_process as setup_process_e5,
            setup_reference as setup_reference_e5,
            solution_path as solution_path_e5,
        )
        process_e5 = setup_process_e5()
        reference_e5 = setup_reference_e5()
        solution_e5 = simulate_and_get_solution(process_e5, solution_path_e5, parameters_e5)
        plot_comparison(ax_column_bd, solution_e5, reference_e5, "darkblue", "E5")

        from e6 import (
            setup_process as setup_process_e6,
            setup_reference as setup_reference_e6,
            solution_path as solution_path_e6,
        )
        process_e6 = setup_process_e6()
        reference_e6 = setup_reference_e6()
        solution_e6 = simulate_and_get_solution(process_e6, solution_path_e6, parameters_e6)
        plot_comparison(ax_column_acetone, solution_e6, reference_e6, "darkred", "E6")

        from e7 import (
            setup_process as setup_process_e7,
            setup_reference as setup_reference_e7,
            solution_path as solution_path_e7,
        )
        process_e7 = setup_process_e7()
        reference_e7 = setup_reference_e7()
        solution_e7 = simulate_and_get_solution(
            process_e7, solution_path_e7, parameters_e7, components="Lysozyme"
        )
        plot_comparison(ax_column_lysozyme, solution_e7, reference_e7, "darkorange", "E7")

        ax_column_bd.set_xlim(0, 12.5)
        handles, labels = get_all_twin_handles_labels(ax_column_bd)
        ax_column_bd.legend(handles, labels, loc="upper right")
        fig_column.tight_layout()

    return fig_column, ax_column_bd, ax_column_acetone, ax_column_lysozyme


# %% Resin titration

def plot_resin_titration(plot_single=False):
    from e8 import setup_references, determine_breakthrough, start_NaOH

    e8_references = setup_references(start_NaOH)
    breakthrough_10s = [determine_breakthrough(ref) for ref in e8_references]

    if plot_single:
        breakthrough_avg = np.mean(breakthrough_10s)
        reference = e8_references[0]
        fig_resin_titration, ax_resin_titration = reference.plot(end=120*60)
        ax_resin_titration.axvline(x=breakthrough_avg/60, color="darkred", linestyle=":")
        ax_resin_titration.set_ylabel("Conductivity / mS")
        ax_resin_titration.get_legend().remove()

        return fig_resin_titration, ax_resin_titration

    figs = []
    axs = []
    for breakthrough_10, reference in zip(breakthrough_10s, e8_references):
        fig_resin_titration, ax_resin_titration = reference.plot(end=120*60)
        ax_resin_titration.axvline(x=breakthrough_10/60, color="darkred", linestyle=":")
        ax_resin_titration.set_ylabel("Conductivity / mS")
        ax_resin_titration.get_legend().remove()
        figs.append(fig_resin_titration)
        axs.append(ax_resin_titration)

    return figs, axs


# %% Characterization Lysozyme

def plot_lysozyme(
    parameters_e9,
    pH,
    include_pore_diffusion,
    is_kinetic,
    use_validation=False,
):
    with plotting.mpl_style_context():
        fig_lysozyme, ax_lysozyme, ax_salt = setup_figure(axes=["Lysozyme", "Salt"])

        from e9 import (
            setup_references,
            setup_lwe_processes,
            solution_path_lysozyme,
            solution_path_salt,
        )

        references_lysozyme, references_salt = setup_references(
            pH,
            use_validation=use_validation,
        )
        lwe_processes = setup_lwe_processes(
            include_pore_diffusion=include_pore_diffusion,
            is_kinetic=is_kinetic,
            use_validation=use_validation,
        )

        colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        for lwe_process, reference_lysozyme, reference_salt in zip(
                lwe_processes, references_lysozyme, references_salt
        ):
            color = next(colors)

            cv = lwe_process.name.split("_")[2]
            simulation_results = update_parameters_and_simulate(lwe_process, parameters_e9)

            solution_lysozyme = extract_solution(
                simulation_results, solution_path_lysozyme, components=["Lysozyme"]
            )
            plot_comparison(ax_lysozyme, solution_lysozyme, reference_lysozyme, color, f"{cv} CV")

            solution_salt = extract_solution(
                simulation_results, solution_path_salt, components=["Salt"]
            )
            plot_comparison(ax_salt, solution_salt, reference_salt, color)

        ax_lysozyme.set_xlim(0, 125)
        handles, labels = get_all_twin_handles_labels(ax_lysozyme)
        ax_lysozyme.legend(handles, labels, loc="upper left")
        fig_lysozyme.tight_layout()

    return fig_lysozyme, ax_lysozyme, ax_salt


# %% Create tables

def embed_table_in_directive(
    table: str,
    caption: Optional[str] = None,
    name: Optional[str] = None,
    align: Optional[str] = "center"
) -> str:
    """Format table to embed it in MyST table directive."""
    formatted_table = "```{table}"
    if caption:
        formatted_table += f" {caption}"

    formatted_table += "\n"
    if name:
        formatted_table += f":name: {name}\n"
    formatted_table += ":widths: grid\n"
    formatted_table += f":align: {align}\n"
    formatted_table += "\n"
    formatted_table += f"{table}\n"
    formatted_table += "```"

    return formatted_table


def format_to_scientific_latex(arr, precision=3, wrap_math=True):
    """
    Convert an array of floats to scientific notation strings with LaTeX formatting.

    Parameters
    ----------
    arr : array_like
        Array of floats.
    precision : int, optional
        Number of significant figures (default is 3).
    wrap_math : bool, optional
        Whether to wrap the result in $...$ for LaTeX math mode (default is False).

    Returns
    -------
    np.ndarray
        Array of formatted strings.
    """
    arr = np.asarray(arr)
    flat = arr.flatten()
    latex_strings = []

    for x in flat:
        # Format the number in scientific notation
        s = np.format_float_scientific(
            x, precision=precision, unique=False, trim='k', exp_digits=1
        )
        base, exponent = s.split('e')
        exponent = exponent.replace('+', '')
        formatted = f"{base} \\times 10^{{{int(exponent)}}}"

        # Apply wrapping if requested
        if wrap_math:
            formatted = f"${formatted}$"

        latex_strings.append(formatted)

    return np.array(latex_strings).reshape(arr.shape)


def create_system_table(parameters):
    fmt = format_to_scientific_latex

    headers = ["Unit operation", "Parameter", "Value", "Unit"]

    rows = []

    tubing_pre_column = parameters["tubing_pre_column"]
    tubing_post_column = parameters["tubing_post_column"]
    tubing_detectors = parameters["tubing_detectors"]
    mixer = parameters["mixer"]
    tubing_pre_injection = parameters["tubing_pre_injection"]

    rows += [
        ["Tubing Pre Column", "Length", fmt(tubing_pre_column["length"]), r"$\text{m}$"],
        [" ", "Axial dispersion", fmt(tubing_pre_column["axial_dispersion"]), r"$\text{m}^2 \text{s}^{-1}$"],

        ["Tubing Post Column", "Length", fmt(tubing_post_column["length"]), r"$\text{m}$"],
        [" ", "Axial dispersion", fmt(tubing_post_column["axial_dispersion"]), r"$\text{m}^2 \text{s}^{-1}$"],

        ["Tubing Detectors", "Length", fmt(tubing_detectors["length"]), r"$m$"],
        [" ", "Axial dispersion", fmt(tubing_detectors["axial_dispersion"]["Salt"]), r"$\text{m}^2 \text{s}^{-1}$"],

        ["Mixer", "Volume", fmt(mixer["init_liquid_volume"]), r"$\text{m}^3$"],
        ["Tubing Pre Injection", "Length", fmt(tubing_pre_injection["length"]), r"$\text{m}$"],
    ]

    table = tabulate(rows, headers=headers, tablefmt="github")

    caption = "Fitted system periphery parameters."
    name = "tab_system_periphery"
    formatted_table = embed_table_in_directive(table, caption, name)

    return formatted_table


def create_column_table(parameters):
    fmt = format_to_scientific_latex

    headers = ["Parameter", "Component", "Value", "Unit"]

    rows = []

    column_parameters = parameters["column"]
    rows += [
        ["Bed porosity", "-", fmt(column_parameters["bed_porosity"]), r"$-$"],
        ["Particle porosity", "-", fmt(column_parameters["particle_porosity"]), r"$-$"],
        ["Axial dispersion", "Blue Dextran", fmt(column_parameters["axial_dispersion"]["Blue Dextran"]), r"$\text{m}^2 \text{s}^{-1}$"],
        ["", "Acetone", fmt(column_parameters["axial_dispersion"]["Acetone"]), r"$\text{m}^2 \text{s}^{-1}$"],
        ["", "Lysozyme", fmt(column_parameters["axial_dispersion"]["Lysozyme"]), r"$\text{m}^2 \text{s}^{-1}$"],
    ]

    table = tabulate(rows, headers=headers, tablefmt="github")

    caption = "Fitted column parameters."
    name = "tab_column_parameters"
    formatted_table = embed_table_in_directive(table, caption, name)

    return formatted_table


def create_lysozyme_table(parameters):
    fmt = format_to_scientific_latex

    headers = ["Parameter", "Component", "Value", "Unit"]

    rows = []

    column_parameters = parameters["column"]
    rows += [
        ["Characteristic charge", "Lysozyme", fmt(column_parameters["binding_model"]["characteristic_charge"]["Lysozyme"]), r"$-$"],
        ["Equilibrium constant", "Lysozyme", fmt(column_parameters["binding_model"]["adsorption_rate"]["Lysozyme"]), r"$\text{m}_\text{l}^3~\text{m}_\text{s}^{-3}$"],
    ]

    table = tabulate(rows, headers=headers, tablefmt="github")

    caption = "Fitted SMA parameters"
    name = "tab_lysozyme_parameters"
    formatted_table = embed_table_in_directive(table, caption, name)

    return formatted_table


def embed_figure_in_directive(
    study_root: os.PathLike,
    branch_name: str,
    figure_path: os.PathLike,
    name: None,
    caption: str,
    scale: Optional[int] = 100,
) -> str:
    """Format figure to embed it in MyST figure directive."""
    load_parameters_from_previous_run(branch_name)

    case_dir = Path(study_root) / 'output_cached' / branch_name
    results_dir_name = [
        d for d in os.listdir(case_dir)
        if os.path.isdir(case_dir / d) and d.startswith("results_")
    ][0]

    results_dir = case_dir / results_dir_name
    relative_results_dir = results_dir.relative_to(Path.cwd(), walk_up=True)

    figure_path = relative_results_dir / figure_path
    if not figure_path.exists():
        raise FileNotFoundError("Figure not found.")

    embedded_figure = f"```{{figure}} {str(figure_path)}"

    embedded_figure += "\n"
    if name:
        embedded_figure += f":name: {name}\n"
    embedded_figure += f":scale: {scale}\n"
    embedded_figure += "\n"
    embedded_figure += f"{caption}\n"
    embedded_figure += "```"

    return embedded_figure


# %% Plot objectives (b&w)

def plot_meta_score(
    study_root: os.PathLike,
    branch_name: str,
) -> tuple[plt.Figure, plt.Axes]:
    load_parameters_from_previous_run(branch_name)
    data = pd.read_csv(
        Path(study_root) / 'output_cached' / branch_name / "results_e9_lrmp/results_all.csv"
    )

    sum_nrmse = data.iloc[:, -4:].sum(axis=1)
    characteristic_charge = data["characteristic_charge"]

    fig, ax = plt.subplots()

    ax.plot(
        characteristic_charge, sum_nrmse,
        'k', marker="o", linestyle="None", markersize=1,
    )
    ax.set_xlabel("Characteristic charge / -")
    ax.set_ylabel('NRMSE / -')
    ax.set_yscale("log")

    fig.tight_layout()

    return fig, ax
