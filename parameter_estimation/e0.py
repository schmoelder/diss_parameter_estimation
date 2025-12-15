"""
# E0: Calibrate conductivity detector.

- Injected component: Salt
- Salt concentrations: [20 mM, 270 mM, 570 mM, 770 mM, 1020 mM]
- Eluent: A
- Measurement: Cond

Status: done

"""

# %% Imports

from cadetrdm import ProjectRepo
import matplotlib.pyplot as plt
import numpy as np

from calibration import polynomial_model, fit_polynomial
from knauer import KnauerExperimentalData
from parameters import flow_rate
from utils import experimental_data_path


# %% Setup reference data

salt_concentrations = [20, 270, 520, 770, 1020]

knauer_data_sets = []
mean_signals = []
for salt_concentration in salt_concentrations:
    file_path = experimental_data_path / "e0" / f"{salt_concentration}_mM.csv"

    knauer_data = KnauerExperimentalData(
        file_path=file_path,
        flow_rate=flow_rate,
    )
    knauer_data_sets.append(knauer_data)

    mean_value = np.mean(knauer_data.conductivity.solution)
    mean_signals.append(mean_value)


# %% Process data

coeffcients, r_squared = fit_polynomial(mean_signals, salt_concentrations, degree=2)


def plot():
    x = np.linspace(min(mean_signals), max(mean_signals), 1001)
    y = polynomial_model(x, coeffcients[0], coeffcients[1], coeffcients[2])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        mean_signals,
        salt_concentrations,
        'k',
        marker="o",
        linestyle="None",
        markersize=12
    )
    ax.plot(x, y, 'k')
    ax.set_xlabel(r'Conductivity / $\text{mS}~\text{cm}^{-1}$')
    ax.set_ylabel(r'Salt concentration / mM')

    # Add R² as a text box
    ax.text(
        0.05, 0.95,  # x, y position (relative to axes)
        f'$R^2 = {r_squared:.3f}$',  # Text to display
        transform=ax.transAxes,  # Use axes coordinates
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    fig.tight_layout()

    return fig, ax


# %% main

def main():
    repo = ProjectRepo(__file__)
    fig, ax = plot()
    fig.savefig(repo.output_path / "salt_calibration.png")


if __name__ == "__main__":
    main()
