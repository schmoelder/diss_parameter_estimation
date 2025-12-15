import copy
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from CADETProcess.reference import ReferenceIO


def fit_baseline(
    time: np.ndarray,
    signal: np.ndarray,
    start: Optional[float] = None,
    end: Optional[float] = None,
    threshold: float = 0.025,
) -> np.ndarray:
    """
    Estimate baseline by fitting a linear function to the lowest intensity values in a time window.

    Parameters
    ----------
    time : ndarray
        1D array of time points.
    signal : ndarray
        1D array of signal values.
    start : float, optional
        Start of the time window to consider for baseline fitting. Defaults to time[0].
    end : float, optional
        End of the time window. Defaults to time[-1].
    threshold : float, optional
        Relative intensity threshold (normalized 0–1) to select baseline points. Default is 0.025.

    Returns
    -------
    baseline : ndarray
        Linear baseline estimate over the entire `time` array.

    Raises
    ------
    ValueError
        If no points are available in the fitting window or baseline region.
    """
    if start is None:
        start = time[0]
    if end is None:
        end = time[-1]

    window_idx = np.where((time >= start) & (time <= end))
    time_window = time[window_idx]
    signal_window = signal[window_idx]

    if len(time_window) == 0:
        raise ValueError("Not enough points to fit a baseline.")

    min_val = np.min(signal_window)
    max_val = np.max(signal_window)

    if max_val == min_val:
        return np.full_like(signal, min_val)

    normalized = (signal_window - min_val) / (max_val - min_val)
    baseline_idx = np.where(normalized < threshold)[0]

    if len(baseline_idx) < 2:
        raise ValueError("Not enough baseline points found under threshold.")

    t_fit = time_window[baseline_idx]
    s_fit = signal_window[baseline_idx]

    coeffs = np.polyfit(t_fit, s_fit, 1)
    baseline = np.polyval(coeffs, time)

    return baseline


def correct_baseline(
    reference: ReferenceIO,
    start: Optional[float] = None,
    end: Optional[float] = None,
    threshold: Optional[float] = 0.025,
) -> ReferenceIO:
    """
    Correct baseline by fitting a linear function to the lowest intensity values in a time window.

    Parameters
    ----------
    time : ndarray
        1D array of time points.
    signal : ndarray
        1D array of signal values.
    start : float, optional
        Start of the time window to consider for baseline fitting. Defaults to time[0].
    end : float, optional
        End of the time window. Defaults to time[-1].
    threshold : float, optional
        Relative intensity threshold (normalized 0–1) to select baseline points.
        Default is 0.025.

    Returns
    -------
    ReferenceIO
        A new ReferenceIO object with baseline correction.
    """
    calibrated_reference = copy.deepcopy(reference)
    calibrated_reference.name = f"{reference.name}_calibrated"

    # Extract time and signal data
    time = reference.time
    signal = reference.solution

    if start is None:
        start = time[0]
    if end is None:
        end = time[-1]

    # Corect for baseline drift
    baseline = fit_baseline(time, signal, start, end, threshold).reshape(-1, 1)
    calibrated_reference.solution = reference.solution - baseline

    # Remove data outside of window
    window_idx = np.where((time < start) | (time > end))
    calibrated_reference.solution[window_idx] = 0

    calibrated_reference.update_solution()

    return calibrated_reference


def normalize_area(
    reference: ReferenceIO,
    target_area: float,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> ReferenceIO:
    """
    Normalize the peak area of a ReferenceIO object.

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
        A new ReferenceIO object with normalized area.
    """
    calibrated_reference = copy.deepcopy(reference)
    calibrated_reference.name = f"{reference.name}_calibrated"

    # Rescale peak to target area
    rescale = target_area / calibrated_reference.fraction_mass(start, end)
    calibrated_reference.solution = calibrated_reference.solution * rescale
    calibrated_reference.update_solution()

    return calibrated_reference


def correct_baseline_and_normalize(
    reference,
    target_area,
    start_baseline=None,
    end_baseline=None,
    threshold=0.025,
    start_normalization=None,
    end_normalization=None,
):
    reference = correct_baseline(
        reference, start_baseline, end_baseline, threshold,
    )
    reference = normalize_area(
        reference, target_area, start_normalization, end_normalization,
    )

    return reference


def polynomial_model(x, *coefficients):
    """
    General polynomial model for curve fitting.

    Parameters
    ----------
    x : array-like
        Independent variable.
    coefficients : array-like
        Polynomial coefficients in descending order of degree.

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
        Fitted polynomial coefficients in descending order of degree.
    float
        Coefficient of determination
    """
    # Initial guess for the coefficients (can be zeros)
    initial_guess = np.zeros(degree + 1)

    # Fit the polynomial model to the data
    coefficients, _ = curve_fit(polynomial_model, x_data, y_data, p0=initial_guess)
    y_fit = polynomial_model(x_data, *coefficients)

    ss_res = np.sum((y_data - y_fit)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - ss_res / ss_tot

    return coefficients.tolist(), r_squared


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
        Polynomial coefficients in descending order of degree.

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
