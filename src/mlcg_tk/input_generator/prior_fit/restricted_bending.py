import torch
import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import trapezoid


def restricted_quartic_angle(x, a, b, c, d, k, v_0):
    r"""
    Quartic angle potential with repulsive 1/sin^2 term to avoid
    singularities at 0 and pi angles.

    .. math::

        V(\theta) = a cos(\theta)^4 + b cos(\theta)^3 + c cos(\theta)^2 + d cos(\theta) + \frac{k}{\sin^2(\theta)} + V0

    """
    cos = torch.cos(x)
    sin = torch.sin(x)

    quart = a * cos**4 + b * cos**3 + c * cos**2 + d * cos
    rep = k / (sin**2)
    V = quart + rep + v_0

    return V


def dx_restricted_quartic_angle_numpy(x, a, b, c, d, k):
    """Derivative of the restricted quartic angle potential with respect to x=theta (numpy version for fsolve)"""

    cos = np.cos(x)
    sin = np.sin(x)

    dquart = -4 * a * sin * cos**3 - 3 * b * sin * cos**2 - 2 * c * sin * cos - d * sin
    drep = -k * (2 * cos) / (sin**3)
    dV = dquart + drep

    return dV


def ddx2_restricted_quartic_angle_numpy(x, a, b, c, d, k):
    """Second derivative of the restricted quartic angle potential with respect to x=theta (numpy version for fsolve)"""

    cos = np.cos(x)
    sin = np.sin(x)

    ddquart = (
        - 4 * a * (cos**4 - 3 * sin**2 * cos**2)
        - 3 * b * (cos**3 - 2 * sin**2 * cos)
        - 2 * c * (cos**2 - sin**2)
        - d * cos
    )
    ddrep = 2 * k * (1 / sin**2 + 3 * cos**2 / sin**4)
    ddV = ddquart + ddrep

    return ddV


def find_extrema_in_range(
    a, b, c, d, k, x_range: Tuple[float, float]
) -> Tuple[List[float], List[float]]:
    """Find minima and maxima in a specified range.

    Args:
        a, b, c, d, k: potential parameters
        x_range: tuple (x_min, x_max) for search region

    Returns:
        tuple: (minima, maxima) - lists of x values where extrema occur
    """
    minima = []
    maxima = []
    search_min = max(x_range[0], 0.01)
    search_max = min(x_range[1], 3.13)

    if search_max <= search_min:
        return minima, maxima

    for x0 in np.linspace(search_min, search_max, 20):
        try:
            root = fsolve(
                dx_restricted_quartic_angle_numpy,
                x0,
                args=(a, b, c, d, k),
                full_output=True,
            )
            x_root = root[0][0]
            info = root[1]

            # Check if fsolve converged and root is in valid range
            if info["fvec"][0] ** 2 < 1e-6 and x_range[0] <= x_root <= x_range[1]:
                d2V = ddx2_restricted_quartic_angle_numpy(x_root, a, b, c, d, k)

                if d2V == 0:
                    continue
                elif d2V > 0:
                    if not any(abs(x_root - m) < 1e-4 for m in minima):
                        minima.append(x_root)
                elif d2V < 0:
                    if not any(abs(x_root - m) < 1e-4 for m in maxima):
                        maxima.append(x_root)
        except RuntimeError:
            print(f"Error finding extrema at x0={x0:.6f}")
            continue

    return sorted(minima), sorted(maxima)


def find_minmax(
    params, left_region: Tuple[float, float], right_region: Tuple[float, float]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Find minima and maxima of the potential in specified regions.

    Args:
        params: (a, b, c, d, k, v_0) parameters
        left_region: tuple (x_min, x_max) for left tail region
        right_region: tuple (x_min, x_max) for right tail region

    Returns:
        tuple: (left_minima, left_maxima, right_minima, right_maxima)
    """
    a, b, c, d, k, _ = params

    left_minima, left_maxima = find_extrema_in_range(a, b, c, d, k, left_region)
    right_minima, right_maxima = find_extrema_in_range(a, b, c, d, k, right_region)

    return left_minima, left_maxima, right_minima, right_maxima


def fit_rb_from_potential_estimates(
    bin_centers_nz: torch.Tensor, dG_nz: torch.Tensor, **kwargs
) -> Dict:
    r"""Fits restricted quartic angle potential. If minima or maxima are found in the tail regions
    (outside the non-zero data range), refits with a=0, b=0 to collapse to quadratic."""

    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )
    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)

    nonzero_indices = torch.where(mask)[0]

    first_nonzero_x = bin_centers_nz[nonzero_indices[0]].item()
    last_nonzero_x = bin_centers_nz[nonzero_indices[-1]].item()

    # Define tail regions (outside the data range)
    left_tail = (0.0, first_nonzero_x)
    right_tail = (last_nonzero_x, np.pi)

    try:
        popt, _ = curve_fit(
            restricted_quartic_angle,
            bin_centers_nz[mask],
            dG_nz[mask],
            p0=[1, 0, 0, 0, 1e-3, torch.argmin(dG_nz[mask])],
            bounds=(
                (0, -np.inf, -np.inf, -np.inf, 1e-5, -np.inf),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            ),
            maxfev=5000,
        )

        left_minima, left_maxima, right_minima, right_maxima = find_minmax(
            popt, left_tail, right_tail
        )

        has_tail_extrema = (
            len(left_minima) > 0
            or len(left_maxima) > 0
            or len(right_minima) > 0
            or len(right_maxima) > 0
        )

        if has_tail_extrema:
            extrema_info = []
            if left_minima:
                extrema_info.append(f"left minima: {left_minima}")
            if left_maxima:
                extrema_info.append(f"left maxima: {left_maxima}")
            if right_minima:
                extrema_info.append(f"right minima: {right_minima}")
            if right_maxima:
                extrema_info.append(f"right maxima: {right_maxima}")

            print(
                f"Extrema found in tail regions ({', '.join(extrema_info)}). Refitting with a=0, b=0"
            )

            # Refit with a=0 and b=0 (collapse to quadratic + repulsive term)
            def restricted_quartic_angle_constrained(x, c, d, k, v_0):
                return restricted_quartic_angle(x, 0, 0, c, d, k, v_0)

            popt_constrained, _ = curve_fit(
                restricted_quartic_angle_constrained,
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=[0, 0, 1e-3, torch.argmin(dG_nz[mask])],
                bounds=(
                    (-np.inf, -np.inf, 1e-5, -np.inf),
                    (np.inf, np.inf, np.inf, np.inf),
                ),
                maxfev=5000,
            )

            stat = {
                "a": torch.tensor(0.0),
                "b": torch.tensor(0.0),
                "c": popt_constrained[0],
                "d": popt_constrained[1],
                "k": popt_constrained[2],
                "v_0": popt_constrained[3],
            }
        else:
            stat = {
                "a": popt[0],
                "b": popt[1],
                "c": popt[2],
                "d": popt[3],
                "k": popt[4],
                "v_0": popt[5],
            }
    except Exception as e:
        print(f"Failed to fit potential estimate for RestrictedQuartic angle: {e}")
        stat = {
            "a": torch.tensor(float("nan")),
            "b": torch.tensor(float("nan")),
            "c": torch.tensor(float("nan")),
            "d": torch.tensor(float("nan")),
            "k": torch.tensor(float("nan")),
            "v_0": torch.tensor(float("nan")),
        }

    return stat
