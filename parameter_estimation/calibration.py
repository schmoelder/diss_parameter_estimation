import copy
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from CADETProcess.reference import ReferenceIO


def normalize_area(
        reference: ReferenceIO,
        target_area: float,
        start: Optional[float] = None,
        end: Optional[float] = None,
        ) -> ReferenceIO:
    """
    Detrend and normalize the peak area of a ReferenceIO object.

    This function finds the baseline by fitting a function to the lower-bound of the
    data and then applies a scaling factor to match the target peak area.

    Parameters
    ----------
    reference : ReferenceIO
        The input ReferenceIO object containing the peak data.
    target_area : float
        The desired integral area of the peak after calibration.
    start : Optional[float]
        The start of the integration range. Default is None, indicating that the first
        time point should be used.
    end : Optional[float]
        The end of the integration range. Default is None, indicating that the last time
        point should be used.

    Returns
    -------
    ReferenceIO
        A new ReferenceIO object with baseline correction and area normalization.
    """
    calibrated_reference = copy.deepcopy(reference)
    calibrated_reference.name = f"{reference.name}_calibrated"

    # Extract time and signal data
    time = reference.time
    signal = reference.solution

    # Step 1: Account for baseline drift
    def fit_baseline(
            time: np.ndarray,
            signal: np.ndarray,
            start: Optional[float] = None,
            end: Optional[float] = None,
            ) -> np.ndarray:
        """Estimate baseline by fitting a function to the lowest values."""
        # Account for peak window
        if start is None:
            start = time[0]
        if end is None:
            end = time[-1]

        window_indices = np.where((time >= start) & (time <= end))
        time_window = time[window_indices]
        signal_window = signal[window_indices]

        if len(time_window) == 0:
            raise ValueError("Not enough points to fit a baseline.")

        min_val = np.min(signal_window)
        max_val = np.max(signal_window)

        if max_val == min_val:  # Avoid division by zero
            return np.ones_like(signal_window, dtype=bool)

        normalized_signal = (signal_window - min_val) / (max_val - min_val)

        # Apply threshold in normalized space
        threshold = 2.5e-2
        baseline_indices = np.where((normalized_signal < threshold))

        baseline_time = time_window[baseline_indices[0]]
        baseline_signal = signal_window[baseline_indices[0]]

        # Fit a linear baseline
        coefficients = np.polyfit(baseline_time, baseline_signal, 1)
        baseline_fit = np.polyval(coefficients, time)

        return baseline_fit

    baseline = fit_baseline(time, signal, start, end).reshape(-1, 1)
    calibrated_reference.solution = reference.solution - baseline
    calibrated_reference.update_solution()

    # Step 2: Rescale peak to target area
    rescale = target_area / calibrated_reference.fraction_mass(start, end)
    calibrated_reference.solution = calibrated_reference.solution * rescale
    calibrated_reference.update_solution()

    return calibrated_reference


def polynomial_model(x, *coefficients):
    """
    General polynomial model for curve fitting.

    Parameters
    ----------
    x : array-like
        Independent variable.
    coefficients : array-like
        Polynomial coefficients in ascending order of degree.

    Returns
    -------
    array-like
        Evaluated polynomial values.
    """
    return np.polyval(coefficients, x)


def fit_polynomial(
        x_data,
        y_data,
        degree
        ) -> list:
    """
    Fit a polynomial model to the given data.

    Parameters
    ----------
    x_data : array-like
        Independent variable data.
    y_data : array-like
        Dependent variable data.
    degree : int
        Degree of the polynomial to fit.

    Returns
    -------
    list
        Fitted polynomial coefficients in ascending order of degree.
    """
    # Initial guess for the coefficients (can be zeros)
    initial_guess = np.zeros(degree + 1)

    # Fit the polynomial model to the data
    coefficients, _ = curve_fit(polynomial_model, x_data, y_data, p0=initial_guess)

    return coefficients.tolist()


def apply_polynomial_calibration(
        reference: ReferenceIO,
        coefficients: np.ndarray
        ) -> np.ndarray:
    """
    Apply a polynomial calibration to a data array using the provided coefficients.

    Parameters
    ----------
    reference : ReferenceIO
        The reference containing sensor data.
    coefficients : np.ndarray
        Polynomial coefficients in ascending order of degree.

    Returns
    -------
    ReferenceIO
        The calibrated reference.
    """
    calibrated_reference = copy.deepcopy(reference)
    calibrated_reference.name = f"{reference.name}_calibrated"

    # Apply the polynomial transformation to the data array
    calibrated_reference.solution = polynomial_model(
        reference.solution,
        *coefficients,
    )
    calibrated_reference.update_solution()

    return calibrated_reference
