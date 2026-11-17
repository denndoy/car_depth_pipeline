"""Compact car-pipeline copy of constrained monotone quartic helpers.

The numerical method is adapted from the person's
``evaluate_final_monotone_quartics.py``. Only generic fitting/prediction
functions are included; person IDs, job IDs, paper plots, and output paths are
intentionally not vendored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize

DEGREE = 4
MONOTONE_TOL_Z = 1e-7


def design_matrix(area: np.ndarray, mean: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    z = (np.asarray(area, dtype=float) - mean) / scale
    return z, np.polynomial.polynomial.polyvander(z, DEGREE)


def predict(area: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    z, _ = design_matrix(
        np.asarray(area, dtype=float),
        float(model["normalization_mean_px2"]),
        float(model["normalization_scale_px2"]),
    )
    return np.polynomial.polynomial.polyval(z, model["coefficients_ascending"])


def exact_max_derivative(coefficients: np.ndarray, z_min: float, z_max: float) -> tuple[float, float]:
    """Find the exact maximum of the cubic derivative over the closed fit interval."""
    derivative = np.polynomial.polynomial.polyder(coefficients)
    second_derivative = np.polynomial.polynomial.polyder(derivative)
    candidates = [float(z_min), float(z_max)]
    for root in np.polynomial.polynomial.polyroots(second_derivative):
        if abs(float(np.imag(root))) <= 1e-9:
            value = float(np.real(root))
            if z_min < value < z_max:
                candidates.append(value)
    values = np.polynomial.polynomial.polyval(candidates, derivative)
    index = int(np.argmax(values))
    return float(values[index]), float(candidates[index])


def derivative_constraint_row(z: float) -> np.ndarray:
    return np.asarray([0.0, 1.0, 2.0 * z, 3.0 * z**2, 4.0 * z**3])


def make_model(
    area: np.ndarray,
    target: np.ndarray,
    mean: float,
    scale: float,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    z, matrix = design_matrix(area, mean, scale)
    residual = matrix @ coefficients - target
    z_min, z_max = float(z.min()), float(z.max())
    max_derivative_z, _ = exact_max_derivative(coefficients, z_min, z_max)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    extrema_z = [z_min, z_max]
    for root in np.polynomial.polynomial.polyroots(derivative):
        if abs(float(np.imag(root))) <= 1e-9:
            value = float(np.real(root))
            if z_min < value < z_max:
                extrema_z.append(value)
    extrema_values = np.polynomial.polynomial.polyval(extrema_z, coefficients)
    lo, hi = int(np.argmin(extrema_values)), int(np.argmax(extrema_values))
    return {
        "degree": DEGREE,
        "feature_name": "A_bb_px2",
        "target_name": "gt_distance_m",
        "normalization_mean_px2": mean,
        "normalization_scale_px2": scale,
        "coefficients_ascending": [float(value) for value in coefficients],
        "training_event_count": int(len(area)),
        "training_area_min_px2": float(np.min(area)),
        "training_area_max_px2": float(np.max(area)),
        "training_distance_min_m": float(np.min(target)),
        "training_distance_max_m": float(np.max(target)),
        "training_mae_m": float(np.mean(np.abs(residual))),
        "training_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "training_bias_m": float(np.mean(residual)),
        "training_prediction_min_m": float(extrema_values[lo]),
        "training_prediction_max_m": float(extrema_values[hi]),
        "max_derivative_m_per_normalized_area": max_derivative_z,
        "max_positive_derivative_m_per_px2": max(0.0, max_derivative_z / scale),
    }


def fit_monotone_quartic(area: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Fit a constrained least-squares quartic non-increasing in bbox area."""
    area = np.asarray(area, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(area) != len(target):
        raise ValueError("Features and targets must have the same number of events")
    if len(area) < DEGREE + 1 or len(np.unique(area)) < DEGREE + 1:
        raise ValueError("A quartic fit requires at least five distinct bounding-box areas")
    if not np.all(np.isfinite(area)) or not np.all(np.isfinite(target)):
        raise ValueError("Quartic inputs must be finite")
    mean, scale = float(area.mean()), float(area.std())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Bounding-box areas must have positive finite standard deviation")
    z, matrix = design_matrix(area, mean, scale)
    initial, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != DEGREE + 1:
        raise ValueError(f"Rank-deficient quartic design matrix ({rank}/{DEGREE + 1})")

    z_min, z_max = float(z.min()), float(z.max())
    nodes = list(np.linspace(z_min, z_max, 21))
    hessian = matrix.T @ matrix
    linear_term = matrix.T @ target
    coefficients = initial.copy()
    total_iterations = 0

    for exchange_iteration in range(30):
        constraints = np.vstack([derivative_constraint_row(node) for node in nodes])
        result = minimize(
            fun=lambda beta: 0.5 * float(beta @ hessian @ beta) - float(linear_term @ beta),
            x0=coefficients,
            jac=lambda beta: hessian @ beta - linear_term,
            hess=lambda _beta: hessian,
            method="trust-constr",
            constraints=[LinearConstraint(constraints, -np.inf, 0.0)],
            options={"maxiter": 5000, "gtol": 1e-10, "xtol": 1e-12,
                     "barrier_tol": 1e-12, "verbose": 0},
        )
        total_iterations += int(result.niter)
        if not result.success:
            raise RuntimeError(f"Monotone quartic optimizer failed: {result.message}")
        coefficients = np.asarray(result.x, dtype=float)
        max_derivative_z, max_location = exact_max_derivative(coefficients, z_min, z_max)
        if max_derivative_z <= MONOTONE_TOL_Z:
            model = make_model(area, target, mean, scale, coefficients)
            model["monotonicity"] = {
                "constraint": "d(distance)/d(area) <= 0 over the observed calibration area range",
                "verified_analytically": True,
                "max_derivative_m_per_normalized_area": max_derivative_z,
                "max_positive_derivative_m_per_px2": max(0.0, max_derivative_z / scale),
                "verification_tolerance_m_per_normalized_area": MONOTONE_TOL_Z,
                "exchange_iterations": exchange_iteration + 1,
                "optimizer_iterations": total_iterations,
                "constraint_nodes": len(nodes),
            }
            return model
        if any(abs(max_location - node) < 1e-10 for node in nodes):
            raise RuntimeError("Monotonicity refinement stalled at a constrained point")
        nodes.append(max_location)
    raise RuntimeError("Monotonicity constraints did not converge")
